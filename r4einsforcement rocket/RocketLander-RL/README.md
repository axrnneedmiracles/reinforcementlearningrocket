# RocketLander-RL: Autonomous Rocket Landing with Deep RL (PPO) & Interactive Visual Replay Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CUDA%20(RTX%20GPU)-EE4C2C.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0%2B-green.svg)](https://gymnasium.farama.org/)
[![Stable--Baselines3](https://img.shields.io/badge/SB3-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Dashboard](https://img.shields.io/badge/Dashboard-100%25%20Offline%20HTML5%2FCanvas-00f.svg)](#interactive-dashboard)

A complete, presentable deep reinforcement learning system simulating realistic 2D rocket landing physics (dynamics, gravity, thrust vectoring, fuel mass depletion, and inertia). Using Proximal Policy Optimization (PPO), the agent learns to control throttle and gimbal angles from an initial tumbling freefall at ~100m altitude to a precision touchdown on the landing pad.

---

## 1. Quick Start & Reproducibility

You can run the entire pipeline with a single command:

### Smoke Test Mode (~2-3 minutes)
```bash
python run_all.py --quick
```
Runs the physics sanity test suite, trains for 200k timesteps saving milestone checkpoints `[1, 20, 100, 500, final]`, evaluates all checkpoints on a fixed showcase seed (`seed=123`) and 100 randomized test seeds, and builds the offline `dashboard.html` and `comparison.png`.

### Full Training Mode (Target: >=90% Success Rate)
```bash
python run_all.py
```
Runs full PPO training up to 5M timesteps or until a rolling success rate >= 90% is achieved across 200 consecutive episodes. Saves checkpoints at `[1, 20, 100, 500, 2000, 10000, final]`.

---

## 2. Project Architecture

```
RocketLander-RL/
│
├── rocket_env.py       # 2D rigid-body rocket landing dynamics & Gymnasium environment
├── tests_physics.py    # 5 physics sanity tests (freefall, thrust, gimbal, fuel, PD controller)
├── train.py            # PPO training with GPU acceleration & milestone checkpoint callbacks
├── record_replays.py   # Records multi-milestone fixed-seed showcase + 100-seed benchmark
├── build_dashboard.py  # Builds self-contained offline dashboard.html & comparison.png
├── run_all.py          # End-to-end pipeline orchestrator (supports --quick)
├── requirements.txt    # Project dependencies
├── README.md           # Documentation & reproduce guide
│
└── outputs/
    ├── checkpoints/    # Model weights (checkpoint_trial_1.zip, ..., checkpoint_trial_final.zip)
    ├── replays.json    # Complete step telemetry for all milestone showcase flights & 100-run stats
    ├── training_log.csv# Episode-by-episode training returns, outcomes, and final telemetry
    ├── dashboard.html  # Self-contained, zero-CDN offline visual replay dashboard
    └── comparison.png  # Multi-milestone trajectory and learning curve plots
```

---

## 3. Physics & Environment Specifications

The simulation runs at **50 Hz** ($dt = 0.02\text{ s}$) with semi-implicit Euler integration:

### Rocket Properties
- **Dry Mass ($m_{\text{dry}}$):** $800.0\text{ kg}$
- **Initial Propellant ($m_{\text{fuel}}$):** $200.0\text{ kg}$
- **Total Initial Mass ($m_0$):** $1000.0\text{ kg}$
- **Height ($H$):** $10.0\text{ m}$ (uniform cylinder about center, $I = \frac{1}{12} m H^2$)
- **Engine Moment Arm ($L$):** $5.0\text{ m}$
- **Max Thrust ($T_{\text{max}}$):** $20,000\text{ N}$ (Thrust-to-weight ratio $\approx 2.04$ at start, accelerating to $2.55$ when fuel depleted)
- **Specific Impulse ($I_{\text{sp}}$):** $300.0\text{ s}$ ($g_0 = 9.81\text{ m/s}^2$)
- **Max Gimbal Angle ($\phi_{\text{max}}$):** $\pm 15^\circ$ ($\approx \pm 0.2618\text{ rad}$)

### Equations of Motion
$$\begin{aligned}
a_x &= -\frac{T}{m} \sin(\theta + \phi) \\
a_y &= \frac{T}{m} \cos(\theta + \phi) - g \\
\alpha &= -\frac{L \cdot T \cdot \sin(\phi)}{I} \\
\dot{m} &= \frac{T}{I_{\text{sp}} g_0}
\end{aligned}$$

---

## 4. Observation & Action Spaces

### Observation Space (8D Normalized Continuous)
$$\mathbf{s} = \left[ \frac{x}{50},\; \frac{y}{100},\; \frac{v_x}{10},\; \frac{v_y}{10},\; \sin(\theta),\; \cos(\theta),\; \omega,\; \frac{\text{fuel}}{\text{fuel}_0} \right] \in \mathbb{R}^8$$

### Action Space (2D Continuous in $[-1, 1]$)
- $a_0 \in [-1, 1] \implies \text{Throttle } u = \frac{a_0 + 1}{2} \in [0, 1]$
- $a_1 \in [-1, 1] \implies \text{Gimbal } \phi = a_1 \times 15^\circ$

---

## 5. Reward Design & Potential-Based Shaping

To prevent reward hacking while ensuring rapid gradient propagation:
- **Potential function:**
  $$\Phi(s) = -\left( \frac{\text{dist}}{10} + \frac{\text{speed}}{2} + 2 |\theta| \right)$$
- **Step Reward:**
  $$r_t = \gamma \Phi(s_{t+1}) - \Phi(s_t) - 0.02 \cdot u - 0.01$$
  where $\gamma = 0.995$.
- **Terminal Outcomes & Rewards:**
  - **Perfect Landing (+100 bonus):** $|v_y| < 0.5\text{ m/s}$, $|v_x| < 0.3\text{ m/s}$, $|\theta| < 0.03\text{ rad}$, $|x| < 1.0\text{ m}$.
  - **Soft Landing (+100 bonus):** $|v_y| < 2.0\text{ m/s}$, $|v_x| < 1.0\text{ m/s}$, $|\theta| < 0.1\text{ rad}$, $|x| < 5.0\text{ m}$.
  - **Hard Landing (-100 penalty):** Touched down ($y \le 0$) with $|v_y| < 6.0\text{ m/s}$ but failed soft landing bounds.
  - **Crash (-100 penalty):** Impacted ground ($y \le 0$) at $|v_y| \ge 6.0\text{ m/s}$ or tilted.
  - **Flipped (-100 penalty):** Severe tilt $|\theta| > 1.0\text{ rad}$ ($\approx 57.3^\circ$).
  - **Flew Away (-100 penalty):** Lateral drift $|x| > 100\text{ m}$ or altitude $y > 200\text{ m}$.
  - **Timeout:** Step limit reached ($t \ge 1000$ steps / $20\text{ s}$).

---

## 6. Physics Sanity Tests (`tests_physics.py`)

Prior to RL training, 5 automated physics sanity checks verify the dynamics:
1. **Zero throttle free-fall:** Confirms $v_y$ decreases by $\approx 9.81\text{ m/s}$ per second.
2. **Full throttle upward acceleration:** Confirms upward net acceleration ($a_y = \frac{20000}{1000} - 9.81 = +10.19\text{ m/s}^2$).
3. **Gimbal torque direction:** Positive $\phi$ yields negative angular acceleration ($\alpha < 0$), negative $\phi$ yields $\alpha > 0$.
4. **Fuel depletion & engine cut-off:** Fuel decrements by exact mass-flow equation and thrust clamps to 0 once empty.
5. **Hand-written PD controller:** Closed-loop controller ($\phi = k_1 \theta + k_2 \omega$) successfully dampens initial tilt of $14.3^\circ$ to $< 2.8^\circ$ upright.

---

## 7. Interactive Offline Dashboard (`outputs/dashboard.html`)

The dashboard is completely **self-contained** into a single HTML file with **zero external CDN links or internet requirements**. It opens by double-clicking in any web browser.

### Features
1. **Hero Summary:** Multi-milestone progress line and key outcome metrics.
2. **Side-by-Side Canvas Replay:** Real-time synchronized animation of all checkpoint flights playing the identical initial condition (`seed=123`). Features realistic Falcon-9 style rocket rendering, dynamic gimbaled exhaust flames, particle trails, altitude/speed HUD, and outcome badges.
3. **Replay Controls:** Play/Pause, Restart, variable speed playback ($0.25\times, 0.5\times, 1\times, 2\times$), and interactive timeline scrubber.
4. **Decision Log:** Explanatory table detailing altitude, velocity, throttle, thrust (N), gimbal angle, and plain-language interpretation of the AI's control intent for each time step.
5. **Multi-Milestone Telemetry Charts:** Interactive overlay charts for altitude, vertical speed, tilt, throttle, gimbal angle, fuel consumption, and cumulative return.
6. **Learning Curve:** Rolling success rate and mean episode return across trials with vertical milestone markers.
7. **Milestone Comparison Table:** Statistical breakdown of 100 randomized test flights with outcome breakdown stacked bars.
8. **Physics & RL Reference:** Complete equations, reward structures, and loop diagrams for non-expert viewers.
