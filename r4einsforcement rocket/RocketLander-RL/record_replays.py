import os
import glob
import json
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

from rocket_env import RocketLandingEnv


def generate_decision_explanation(step_info):
    """
    Generate intuitive plain-language explanation of what the AI is doing based on telemetry.
    Matches Section 7 item 3 of the prompt.
    """
    y = step_info["y"]
    vy = step_info["vy"]
    vx = step_info["vx"]
    th_deg = step_info["theta_deg"]
    throttle_pct = step_info["throttle_pct"]
    thrust_N = step_info["thrust_N"]
    gimbal_deg = step_info["gimbal_deg"]
    fuel_kg = step_info["fuel_kg"]

    if fuel_kg <= 0:
        return "Fuel exhausted: rocket falling ballistic under gravity"

    if y <= 0.2:
        return f"Touchdown: engine shutoff, landing gear in contact with pad"

    parts = []
    # Throttle rationale
    if throttle_pct > 80:
        if vy < -8.0:
            parts.append(f"Throttle {throttle_pct:.0f}%: falling fast ({vy:.1f} m/s), firing suicide burn to decelerate")
        else:
            parts.append(f"Throttle {throttle_pct:.0f}%: heavy thrust ({thrust_N:.0f} N) to fight gravity and arrest sink rate")
    elif throttle_pct > 40:
        if y < 20:
            parts.append(f"Throttle {throttle_pct:.0f}%: final terminal descent throttling for smooth touchdown")
        else:
            parts.append(f"Throttle {throttle_pct:.0f}%: hovering/braking with {thrust_N:.0f} N")
    elif throttle_pct > 5:
        parts.append(f"Throttle {throttle_pct:.0f}%: gentle station-keeping burn ({thrust_N:.0f} N)")
    else:
        parts.append(f"Throttle 0%: engine idle, coasting downward to conserve propellant")

    # Gimbal / steering rationale
    if abs(th_deg) > 4.0:
        direction = "right" if th_deg > 0 else "left"
        steer = "counter-torque" if np.sign(gimbal_deg) == np.sign(th_deg) else "tilt correction"
        parts.append(f"Gimbal {gimbal_deg:+.1f}°: tilted {abs(th_deg):.1f}° {direction}, applying {steer} to realign upright")
    elif abs(vx) > 1.5:
        drift = "right" if vx > 0 else "left"
        parts.append(f"Gimbal {gimbal_deg:+.1f}°: drifting {drift} at {abs(vx):.1f} m/s, vectoring thrust to null horizontal drift")
    elif abs(gimbal_deg) > 1.0:
        parts.append(f"Gimbal {gimbal_deg:+.1f}°: micro-attitude trim")
    else:
        parts.append("Gimbal centered: vertical orientation stabilized")

    return " | ".join(parts)


def evaluate_checkpoint_100_runs(model, seed_start=1000):
    """
    Run 100 random seeds (deterministic=True) and return outcomes and statistics.
    """
    env = RocketLandingEnv()
    outcomes_count = {
        "Perfect landing": 0,
        "Soft landing": 0,
        "Hard landing": 0,
        "Crash": 0,
        "Flipped": 0,
        "Flew away": 0,
        "Timeout": 0
    }

    final_vys = []
    final_vxs = []
    final_tilts = []
    final_xs = []
    final_fuels = []
    total_rewards = []

    for i in range(100):
        seed = seed_start + i
        obs, _ = env.reset(seed=seed)
        done = False
        truncated = False
        ep_reward = 0.0
        last_info = {}

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            last_info = info

        outcome = last_info.get("outcome", "Crash")
        if outcome in outcomes_count:
            outcomes_count[outcome] += 1
        else:
            outcomes_count["Crash"] += 1

        final_vys.append(last_info.get("vy", 0.0))
        final_vxs.append(last_info.get("vx", 0.0))
        final_tilts.append(abs(last_info.get("theta_rad", 0.0)))
        final_xs.append(abs(last_info.get("x", 0.0)))
        final_fuels.append(last_info.get("fuel_kg", 0.0))
        total_rewards.append(ep_reward)

    success_count = outcomes_count["Perfect landing"] + outcomes_count["Soft landing"]
    success_rate = float(success_count / 100.0 * 100.0)

    return {
        "success_rate_pct": success_rate,
        "outcomes": outcomes_count,
        "avg_vy": float(np.mean(final_vys)),
        "avg_vx": float(np.mean(final_vxs)),
        "avg_tilt_deg": float(np.degrees(np.mean(final_tilts))),
        "avg_x_error_m": float(np.mean(final_xs)),
        "avg_fuel_left_kg": float(np.mean(final_fuels)),
        "avg_reward": float(np.mean(total_rewards))
    }


def record_showcase_episode(model, is_trial_1=False, fixed_seed=123):
    """
    Run one episode on fixed seed 123.
    Trial-1 checkpoint: deterministic=False (stochastic).
    Later checkpoints: deterministic=True.
    Store every step.
    """
    env = RocketLandingEnv()
    obs, _ = env.reset(seed=fixed_seed)
    done = False
    truncated = False

    steps_data = []
    ep_reward = 0.0
    last_info = {}

    deterministic = not is_trial_1

    # Extract policy network device
    policy = model.policy
    device = policy.device

    while not (done or truncated):
        # Action distribution & critic value estimate
        obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
        action_mean = None
        action_std = None
        value_est = None

        with torch.no_grad():
            distribution = policy.get_distribution(obs_tensor)
            values = policy.predict_values(obs_tensor)
            if hasattr(distribution, "distribution"):
                dist_obj = distribution.distribution
                if hasattr(dist_obj, "mean"):
                    action_mean = dist_obj.mean.squeeze().cpu().tolist()
                if hasattr(dist_obj, "stddev"):
                    action_std = dist_obj.stddev.squeeze().cpu().tolist()
            value_est = float(values.squeeze().cpu().item())

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        last_info = info

        step_record = {
            "t": int(info["t"]),
            "time_sec": round(float(info["time_sec"]), 3),
            "x": round(float(info["x"]), 3),
            "y": round(float(info["y"]), 3),
            "vx": round(float(info["vx"]), 3),
            "vy": round(float(info["vy"]), 3),
            "theta_rad": round(float(info["theta_rad"]), 4),
            "theta_deg": round(float(info["theta_deg"]), 2),
            "omega": round(float(info["omega"]), 4),
            "fuel_kg": round(float(info["fuel_kg"]), 2),
            "fuel_pct": round(float(info["fuel_pct"]), 1),
            "raw_action": [round(float(a), 4) for a in info["raw_action"]],
            "throttle_pct": round(float(info["throttle_pct"]), 1),
            "thrust_N": round(float(info["thrust_N"]), 1),
            "gimbal_deg": round(float(info["gimbal_deg"]), 2),
            "step_reward": round(float(reward), 4),
            "cumulative_reward": round(float(info["cumulative_reward"]), 2),
            "mass_kg": round(float(info["mass_kg"]), 2),
            "action_mean": [round(float(m), 4) for m in action_mean] if action_mean else None,
            "action_std": [round(float(s), 4) for s in action_std] if action_std else None,
            "value_estimate": round(value_est, 3) if value_est is not None else None,
        }
        step_record["explanation"] = generate_decision_explanation(info)
        steps_data.append(step_record)

    showcase = {
        "deterministic": deterministic,
        "seed": fixed_seed,
        "total_steps": len(steps_data),
        "total_reward": round(ep_reward, 2),
        "outcome": last_info.get("outcome", "Unknown"),
        "final_y": round(float(last_info.get("y", 0.0)), 2),
        "final_vy": round(float(last_info.get("vy", 0.0)), 2),
        "final_vx": round(float(last_info.get("vx", 0.0)), 2),
        "final_tilt_deg": round(float(np.degrees(last_info.get("theta_rad", 0.0))), 2),
        "final_x_error": round(float(abs(last_info.get("x", 0.0))), 2),
        "fuel_left_kg": round(float(last_info.get("fuel_kg", 0.0)), 2),
        "steps": steps_data
    }
    return showcase


def record_all_replays(checkpoint_dir="outputs/checkpoints", output_path="outputs/replays.json"):
    """
    Find all checkpoints, record fixed-start showcase and 100-seed stats, save to replays.json.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_trial_*.zip"))

    # Sort checkpoints logically: trial 1, 20, 100, 500, 2000, 10000, final
    def sort_key(path):
        base = os.path.basename(path).replace("checkpoint_trial_", "").replace(".zip", "")
        if base == "final":
            return 999999999
        try:
            return int(base)
        except ValueError:
            return 999999

    checkpoint_files = sorted(checkpoint_files, key=sort_key)
    print(f"Found {len(checkpoint_files)} checkpoints to evaluate: {[os.path.basename(p) for p in checkpoint_files]}")

    results = {}
    hero_story_parts = []

    for ckpt_path in checkpoint_files:
        name = os.path.basename(ckpt_path).replace("checkpoint_trial_", "").replace(".zip", "")
        is_trial_1 = (name == "1")
        display_label = f"Trial {name}" if name != "final" else "Final Model"

        print(f"\nProcessing {display_label} ({ckpt_path})...")
        model = PPO.load(ckpt_path)

        # 1. Run showcase episode on fixed start (seed 123)
        showcase = record_showcase_episode(model, is_trial_1=is_trial_1, fixed_seed=123)
        print(f"  Showcase (Seed 123): Outcome = {showcase['outcome']}, Steps = {showcase['total_steps']}, Reward = {showcase['total_reward']}")

        # 2. Run 100 random seeds for rigorous statistical evaluation
        print(f"  Running 100 random test seeds for statistical validation...")
        stats_100 = evaluate_checkpoint_100_runs(model, seed_start=1000)
        print(f"  100-Run Success Rate: {stats_100['success_rate_pct']:.1f}% | Outcomes: {stats_100['outcomes']}")

        results[name] = {
            "checkpoint_name": name,
            "label": display_label,
            "is_trial_1": is_trial_1,
            "showcase": showcase,
            "eval_100": stats_100
        }

        hero_story_parts.append(f"{display_label}: {showcase['outcome']} ({stats_100['success_rate_pct']:.0f}% win)")

    # Formulate Hero Summary story
    hero_summary = " -> ".join(hero_story_parts)

    output_payload = {
        "hero_summary": hero_summary,
        "milestones": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[Replays Recorded] Saved full evaluation and telemetry to {output_path}")
    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record replays and evaluations for checkpoints")
    parser.add_argument("--checkpoints", default="outputs/checkpoints", help="Path to checkpoints directory")
    parser.add_argument("--output", default="outputs/replays.json", help="Path to output replays.json")
    args = parser.parse_args()

    record_all_replays(checkpoint_dir=args.checkpoints, output_path=args.output)
