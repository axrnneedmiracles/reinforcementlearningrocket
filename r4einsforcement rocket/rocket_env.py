import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RocketLandingEnv(gym.Env):
    """
    2D rocket landing environment with real physics.
    Units: m, s, kg, N, rad. Angle 0 = upright.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, curriculum_level: float = 1.0):
        super().__init__()
        self.g = self.g0 = 9.81
        self.dt = 0.02  # 50 Hz physics
        self.m_dry, self.fuel0 = 800.0, 200.0
        self.T_max, self.Isp = 20000.0, 300.0
        self.L, self.H = 5.0, 10.0  # engine arm, rocket height
        self.max_gimbal = float(np.radians(15))
        self.gamma = 0.995

        # Action: continuous [-1, 1]^2 -> [throttle control, gimbal control]
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        # Observation (normalised):
        # [x/50, y/100, vx/10, vy/10, sin(th), cos(th), w, fuel/fuel0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)

        # Curriculum factor between 0.0 (easier) and 1.0 (full random range)
        self.curriculum_level = float(np.clip(curriculum_level, 0.0, 1.0))

        # Internal state
        self.x = 0.0
        self.y = 100.0
        self.vx = 0.0
        self.vy = -10.0
        self.th = 0.0
        self.w = 0.0
        self.fuel = self.fuel0
        self.t = 0
        self.prev_pot = 0.0
        self.cumulative_reward = 0.0

    def set_curriculum_level(self, level: float):
        self.curriculum_level = float(np.clip(level, 0.1, 1.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random

        # Randomised start each episode scaled by curriculum_level
        c = self.curriculum_level
        if c >= 0.999:
            # Full prompt bounds: x +-20, y 80..120, vx +-3, vy -15..-5, tilt +-0.2
            self.x = float(r.uniform(-20.0, 20.0))
            self.y = float(r.uniform(80.0, 120.0))
            self.vx = float(r.uniform(-3.0, 3.0))
            self.vy = float(r.uniform(-15.0, -5.0))
            self.th = float(r.uniform(-0.2, 0.2))
        else:
            # Curriculum progression: start close to ground so policy discovers touchdown bonus (+100)
            # Scales linearly to full bounds as c approaches 1.0
            self.x = float(r.uniform(-1.0 - 19.0 * c, 1.0 + 19.0 * c))
            self.y = float(r.uniform(8.0 + 72.0 * c, 16.0 + 104.0 * c))
            self.vx = float(r.uniform(-0.4 - 2.6 * c, 0.4 + 2.6 * c))
            self.vy = float(r.uniform(-2.5 - 12.5 * c, -0.5 - 4.5 * c))
            self.th = float(r.uniform(-0.02 - 0.18 * c, 0.02 + 0.18 * c))
        self.w = 0.0
        self.fuel = float(self.fuel0)
        self.t = 0
        self.prev_pot = self._potential()
        self.cumulative_reward = 0.0

        return self._obs(), {}

    def _obs(self):
        return np.array([
            self.x / 50.0,
            self.y / 100.0,
            self.vx / 10.0,
            self.vy / 10.0,
            np.sin(self.th),
            np.cos(self.th),
            self.w,
            self.fuel / self.fuel0
        ], dtype=np.float32)

    def _potential(self):
        dist = np.hypot(self.x, self.y)
        speed = np.hypot(self.vx, self.vy)
        return -(dist / 10.0 + speed / 2.0 + 2.0 * abs(self.th))

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        u = float((a[0] + 1.0) / 2.0)  # throttle 0..1
        phi = float(a[1] * self.max_gimbal)  # gimbal angle in radians

        T = float(u * self.T_max if self.fuel > 0 else 0.0)
        m = float(self.m_dry + self.fuel)
        I = float(m * (self.H ** 2) / 12.0)  # uniform rod about center

        ax = float(-T / m * np.sin(self.th + phi))
        ay = float(T / m * np.cos(self.th + phi) - self.g)
        alpha = float(-self.L * T * np.sin(phi) / I)

        dm = float(T / (self.Isp * self.g0) * self.dt)  # fuel burned this step
        self.fuel = float(max(0.0, self.fuel - dm))

        # semi-implicit Euler: update velocities first, then positions
        self.vx += float(ax * self.dt)
        self.vy += float(ay * self.dt)
        self.w += float(alpha * self.dt)
        self.x += float(self.vx * self.dt)
        self.y += float(self.vy * self.dt)
        self.th += float(self.w * self.dt)
        self.t += 1

        pot = self._potential()
        reward = float(self.gamma * pot - self.prev_pot)  # potential-based shaping
        self.prev_pot = pot
        reward -= float(0.02 * u + 0.01)  # fuel cost + time cost

        terminated = False
        truncated = bool(self.t >= 1000)
        outcome = None

        if abs(self.th) > 1.0:
            reward -= 100.0
            terminated = True
            outcome = "Flipped"
        elif abs(self.x) > 100.0 or self.y > 200.0:
            reward -= 100.0
            terminated = True
            outcome = "Flew away"
        elif self.y <= 0.0:
            terminated = True
            # Check landing outcome criteria
            is_perfect = (
                abs(self.vy) < 0.5 and
                abs(self.vx) < 0.3 and
                abs(self.th) < 0.03 and
                abs(self.x) < 1.0
            )
            is_soft = (
                abs(self.vy) < 2.0 and
                abs(self.vx) < 1.0 and
                abs(self.th) < 0.1 and
                abs(self.x) < 5.0
            )

            if is_perfect:
                outcome = "Perfect landing"
                reward += 100.0
            elif is_soft:
                outcome = "Soft landing"
                reward += 100.0
            elif abs(self.vy) < 6.0:
                outcome = "Hard landing"
                reward -= 100.0
            else:
                outcome = "Crash"
                reward -= 100.0
        elif truncated:
            # User requirement: Don't treat timeout as free. Give timeout a penalty at least as bad as crash!
            outcome = "Timeout"
            reward -= 100.0
            terminated = True

        self.cumulative_reward += reward

        info = {
            "t": self.t,
            "time_sec": self.t * self.dt,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "theta_rad": self.th,
            "theta_deg": float(np.degrees(self.th)),
            "omega": self.w,
            "fuel_kg": self.fuel,
            "fuel_pct": float(self.fuel / self.fuel0 * 100.0),
            "mass_kg": m,
            "raw_action": [float(a[0]), float(a[1])],
            "throttle_pct": float(u * 100.0),
            "thrust_N": T,
            "gimbal_deg": float(np.degrees(phi)),
            "step_reward": reward,
            "cumulative_reward": self.cumulative_reward,
            "outcome": outcome,
            "is_success": bool(outcome in ["Perfect landing", "Soft landing"])
        }

        return self._obs(), float(reward), terminated, truncated, info
