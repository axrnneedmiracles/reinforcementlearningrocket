import os
import csv
import argparse
from collections import deque
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from rocket_env import RocketLandingEnv

# Configurable trial milestones at the top of the file
DEFAULT_MILESTONES = [1, 20, 100, 500, 2000, 10000]
QUICK_MILESTONES = [1, 20, 100, 500]


class RocketTrainingCallback(BaseCallback):
    """
    Custom callback that tracks completed episodes across all parallel envs,
    saves checkpoints at exact trial milestones, logs to training_log.csv,
    and terminates early when rolling success rate >= 90% and final milestone reached.
    """
    def __init__(self, milestones, log_path, checkpoint_dir, max_timesteps, verbose=1):
        super().__init__(verbose)
        self.milestones = sorted(list(milestones))
        self.log_path = log_path
        self.checkpoint_dir = checkpoint_dir
        self.max_timesteps = max_timesteps

        self.episode_count = 0
        self.saved_milestones = set()
        self.rolling_outcomes = deque(maxlen=200)

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize CSV log file with header
        with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_number",
                "return",
                "length",
                "outcome",
                "final_vy",
                "final_vx",
                "final_tilt",
                "final_x",
                "fuel_left"
            ])

    def _on_step(self) -> bool:
        # Check done signals from parallel envs
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done:
                    self.episode_count += 1
                    info = infos[i]

                    # Extract episode stats
                    ep_return = info.get("cumulative_reward", 0.0)
                    ep_length = info.get("t", 0)
                    outcome = info.get("outcome", "Unknown")
                    final_vy = info.get("vy", 0.0)
                    final_vx = info.get("vx", 0.0)
                    final_tilt = info.get("theta_rad", 0.0)
                    final_x = info.get("x", 0.0)
                    fuel_left = info.get("fuel_kg", 0.0)

                    is_success = outcome in ["Perfect landing", "Soft landing"]
                    self.rolling_outcomes.append(1 if is_success else 0)

                    # Log to CSV
                    with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            self.episode_count,
                            f"{ep_return:.4f}",
                            ep_length,
                            outcome,
                            f"{final_vy:.4f}",
                            f"{final_vx:.4f}",
                            f"{final_tilt:.4f}",
                            f"{final_x:.4f}",
                            f"{fuel_left:.2f}"
                        ])

                    # Check milestones to save
                    for m in self.milestones:
                        if m not in self.saved_milestones and self.episode_count >= m:
                            ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_trial_{m}.zip")
                            self.model.save(ckpt_path)
                            self.saved_milestones.add(m)
                            print(f"\n[Checkpoint Saved] Milestone {m} saved at episode {self.episode_count} -> {ckpt_path}")

                    # Check rolling success rate
                    rolling_success = (sum(self.rolling_outcomes) / len(self.rolling_outcomes)) if self.rolling_outcomes else 0.0

                    if self.episode_count % 10 == 0:
                        curriculum = min(1.0, 0.2 + 0.8 * (self.num_timesteps / max(1, self.max_timesteps * 0.4)))
                        try:
                            self.training_env.env_method("set_curriculum_level", curriculum)
                        except Exception:
                            pass

                    if self.episode_count % 50 == 0:
                        recent_n = len(self.rolling_outcomes)
                        print(f"Episode {self.episode_count:5d} | Timesteps: {self.num_timesteps:7d} | "
                              f"Outcome: {outcome:15s} | Ret: {ep_return:7.1f} | "
                              f"Rolling Success ({recent_n} eps): {rolling_success * 100:5.1f}%", flush=True)

                    # Stopping criteria:
                    # Rolling success rate (last 200 eps) >= 90% AND all milestones reached
                    all_milestones_saved = all(m in self.saved_milestones for m in self.milestones)
                    if len(self.rolling_outcomes) >= 100 and rolling_success >= 0.90 and all_milestones_saved:
                        print(f"\n>>> Target reached: Rolling success rate = {rolling_success*100:.1f}% >= 90% "
                              f"after {self.episode_count} episodes and all milestones saved!")
                        return False

        return True


def make_env():
    return RocketLandingEnv()


def train(quick=False, total_timesteps=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RocketLander-RL Training ===")
    print(f"Device: {device.upper()} (CUDA Available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    milestones = QUICK_MILESTONES if quick else DEFAULT_MILESTONES
    if total_timesteps is None:
        total_timesteps = 200_000 if quick else 5_000_000

    print(f"Mode: {'QUICK' if quick else 'FULL'}")
    print(f"Max Timesteps: {total_timesteps:,}")
    print(f"Milestones: {milestones}")

    # 8 parallel environments using DummyVecEnv
    n_envs = 8
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

    # Neural network policy architecture: 2x128
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    # Initialize PPO with prompt hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0
    )

    log_path = os.path.join("outputs", "training_log.csv")
    checkpoint_dir = os.path.join("outputs", "checkpoints")

    callback = RocketTrainingCallback(
        milestones=milestones,
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        max_timesteps=total_timesteps,
        verbose=1
    )

    print("\nStarting PPO training...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final model...")

    # Ensure all milestones are accounted for
    for m in milestones:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_trial_{m}.zip")
        if not os.path.exists(ckpt_path):
            model.save(ckpt_path)
            print(f"[Fallback Saved] Checkpoint for milestone {m} saved at end of training.")

    # Always save final checkpoint
    final_path = os.path.join(checkpoint_dir, "checkpoint_trial_final.zip")
    model.save(final_path)
    print(f"[Final Checkpoint Saved] -> {final_path}")
    print(f"Total episodes completed: {callback.episode_count}")
    print(f"Training finished successfully.")

    vec_env.close()
    return callback.episode_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RocketLander-RL agent with PPO on GPU/CPU")
    parser.add_argument("--quick", action="store_true", help="Run quick 200k timesteps mode with milestones [1, 20, 100, 500]")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    args = parser.parse_args()

    train(quick=args.quick, total_timesteps=args.timesteps)
