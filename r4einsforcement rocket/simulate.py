import os
import sys
import time
import argparse
import subprocess

# Auto-detect local virtualenv if user executes with global python
venv_python = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
    try:
        import stable_baselines3
    except ImportError:
        sys.exit(subprocess.run([venv_python] + sys.argv).returncode)

import numpy as np
from stable_baselines3 import PPO
from rocket_env import RocketLandingEnv


def print_hud(step, t_sec, x, y, vx, vy, th_deg, w, fuel, throttle_pct, thrust_N, gimbal_deg, outcome=None):
    """
    Renders an ASCII mission control telemetry HUD in terminal.
    """
    fuel_bar_len = 16
    fuel_filled = int(fuel / 200.0 * fuel_bar_len)
    fuel_bar = "=" * fuel_filled + " " * (fuel_bar_len - fuel_filled)

    throt_bar_len = 12
    throt_filled = int(throttle_pct / 100.0 * throt_bar_len)
    throt_bar = "#" * throt_filled + " " * (throt_bar_len - throt_filled)

    orient = "|" if abs(th_deg) < 3 else ("/" if th_deg > 0 else "\\")
    flame = "^" if throttle_pct > 30 else ("." if throttle_pct > 0 else " ")

    status_str = f" [{outcome.upper()}]" if outcome else ""

    line = (
        f"\r[T+{t_sec:5.2f}s | Step {step:4d}] "
        f"Alt: {y:5.1f}m | Vy: {vy:+5.1f}m/s | X: {x:+5.1f}m | Vx: {vx:+4.1f}m/s | "
        f"Tilt: {th_deg:+5.1f}deg {orient}{flame} | Gimbal: {gimbal_deg:+4.1f}deg | "
        f"Throt: [{throt_bar}] {throttle_pct:3.0f}% | "
        f"Fuel: [{fuel_bar}] {fuel:5.1f}kg{status_str}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def run_simulation(checkpoint_path=None, seed=None, altitude=None, tilt=None, render_gui=False, delay=0.02):
    # Auto-find latest or final checkpoint if not specified
    if checkpoint_path is None:
        candidates = [
            os.path.join("outputs", "checkpoints", "checkpoint_trial_final.zip"),
            os.path.join("outputs", "checkpoints", "checkpoint_trial_500.zip"),
            os.path.join("outputs", "checkpoints", "checkpoint_trial_100.zip"),
            os.path.join("outputs", "checkpoints", "checkpoint_trial_20.zip"),
            os.path.join("outputs", "checkpoints", "checkpoint_trial_1.zip"),
        ]
        for c in candidates:
            if os.path.exists(c):
                checkpoint_path = c
                break

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found! Run train.py or run_all.py first.")
        sys.exit(1)

    print("=" * 80)
    print("           ROCKETLANDER-RL: AUTONOMOUS FLIGHT SIMULATION")
    print("=" * 80)
    print(f"Loading Model Checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, device="cpu")

    env = RocketLandingEnv()
    obs, _ = env.reset(seed=seed)

    # Override initial conditions if specified
    if altitude is not None:
        env.y = float(altitude)
    if tilt is not None:
        env.th = float(np.radians(tilt))

    print(f"Initial State: Altitude={env.y:.1f}m, X={env.x:.1f}m, Vy={env.vy:.1f}m/s, Vx={env.vx:.1f}m/s, Tilt={np.degrees(env.th):.1f}°")
    print("-" * 80)

    done = False
    truncated = False
    step = 0
    total_reward = 0.0

    trajectory = []

    # Simulation loop
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        total_reward += reward

        trajectory.append({
            "t": step,
            "time_sec": info["time_sec"],
            "x": info["x"],
            "y": info["y"],
            "vy": info["vy"],
            "vx": info["vx"],
            "th_deg": info["theta_deg"],
            "throttle": info["throttle_pct"],
            "thrust": info["thrust_N"],
            "gimbal": info["gimbal_deg"],
            "fuel": info["fuel_kg"]
        })

        outcome = info.get("outcome") if (done or truncated) else None
        print_hud(
            step=step,
            t_sec=info["time_sec"],
            x=info["x"],
            y=info["y"],
            vx=info["vx"],
            vy=info["vy"],
            th_deg=info["theta_deg"],
            w=info["omega"],
            fuel=info["fuel_kg"],
            throttle_pct=info["throttle_pct"],
            thrust_N=info["thrust_N"],
            gimbal_deg=info["gimbal_deg"],
            outcome=outcome
        )

        if delay > 0:
            time.sleep(delay)

    final_outcome = info.get("outcome", "Unknown")
    print("\n" + "=" * 80)
    print(f"MISSION TERMINATED: {final_outcome.upper()}")
    print("=" * 80)
    print(f"  * Total Steps:      {step} ({step * 0.02:.2f} seconds)")
    print(f"  * Cumulative Return: {total_reward:.2f}")
    print(f"  * Touchdown Vy:     {info['vy']:.2f} m/s (Safe landing limit: < 2.0 m/s)")
    print(f"  * Touchdown Vx:     {info['vx']:.2f} m/s (Safe landing limit: < 1.0 m/s)")
    print(f"  * Touchdown Tilt:   {info['theta_deg']:.2f} deg (Safe landing limit: < 5.7 deg)")
    print(f"  * Pad Offset (|x|): {abs(info['x']):.2f} m (Safe landing limit: < 5.0 m)")
    print(f"  * Fuel Remaining:   {info['fuel_kg']:.1f} kg ({info['fuel_pct']:.1f}%)")
    print("=" * 80)

    # Optional GUI trajectory plot
    if render_gui:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor("#0a0e17")
            ax1.set_facecolor("#0f172a")
            ax2.set_facecolor("#0f172a")

            xs = [p["x"] for p in trajectory]
            ys = [p["y"] for p in trajectory]
            ts = [p["time_sec"] for p in trajectory]
            vys = [p["vy"] for p in trajectory]

            # Trajectory plot
            ax1.plot(xs, ys, color="#38bdf8", linewidth=2.5, label="Flight Path")
            ax1.scatter([xs[0]], [ys[0]], color="#f59e0b", s=60, label="Start")
            ax1.scatter([xs[-1]], [ys[-1]], color="#22c55e", s=80, label=f"End: {final_outcome}")
            ax1.plot([-5, 5], [0, 0], color="#22c55e", linewidth=4, label="Landing Pad")
            ax1.axhline(0, color="#64748b", linestyle="--")
            ax1.set_title("Rocket Trajectory", color="#ffffff", fontweight="bold")
            ax1.set_xlabel("Horizontal X (m)", color="#94a3b8")
            ax1.set_ylabel("Altitude Y (m)", color="#94a3b8")
            ax1.grid(True, linestyle=":", alpha=0.3)
            ax1.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

            # Velocity plot
            ax2.plot(ts, vys, color="#f97316", linewidth=2, label="Vertical Speed (vy)")
            ax2.axhspan(-2, 0, color="#10b981", alpha=0.2, label="Safe Touchdown Band (|vy|<2)")
            ax2.axhline(0, color="#64748b", linestyle="--")
            ax2.set_title("Vertical Velocity Profile", color="#ffffff", fontweight="bold")
            ax2.set_xlabel("Time (s)", color="#94a3b8")
            ax2.set_ylabel("Velocity vy (m/s)", color="#94a3b8")
            ax2.grid(True, linestyle=":", alpha=0.3)
            ax2.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[Note] GUI display skipped: {e}")

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RocketLander-RL autonomous simulation")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint model zip")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for start condition")
    parser.add_argument("--altitude", type=float, default=None, help="Custom start altitude in meters (e.g. 100)")
    parser.add_argument("--tilt", type=float, default=None, help="Custom start tilt in degrees (e.g. 10)")
    parser.add_argument("--gui", action="store_true", help="Display Matplotlib trajectory plot after landing")
    parser.add_argument("--fast", action="store_true", help="Run at maximum speed without sleep delay")
    args = parser.parse_args()

    delay = 0.0 if args.fast else 0.015
    run_simulation(
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        altitude=args.altitude,
        tilt=args.tilt,
        render_gui=args.gui,
        delay=delay
    )
