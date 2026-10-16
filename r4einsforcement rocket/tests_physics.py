import sys
import numpy as np
from rocket_env import RocketLandingEnv


def test_zero_throttle_freefall():
    """
    Zero throttle: rocket free-falls, vy decreases by about 9.81 m/s per second.
    """
    env = RocketLandingEnv()
    env.reset(seed=42)
    # Set known state
    env.y = 150.0
    env.vy = 0.0
    env.th = 0.0
    env.w = 0.0
    env.x = 0.0
    env.vx = 0.0

    # 1 second of simulation = 50 steps (dt = 0.02)
    # Action [-1, 0] corresponds to throttle = 0, gimbal = 0
    zero_action = np.array([-1.0, 0.0], dtype=np.float32)
    for _ in range(50):
        env.step(zero_action)

    expected_vy = -9.81
    delta_vy = abs(env.vy - expected_vy)
    passed = delta_vy < 0.1
    print(f"Test 1 [Zero Throttle Free-fall]: vy = {env.vy:.4f} m/s (expected ~{expected_vy:.2f} m/s, diff = {delta_vy:.4f}) -> {'PASS' if passed else 'FAIL'}")
    return passed


def test_full_throttle_acceleration():
    """
    Full throttle, gimbal 0, upright: rocket accelerates upward (thrust-to-weight > 1).
    """
    env = RocketLandingEnv()
    env.reset(seed=42)
    env.y = 100.0
    env.vy = 0.0
    env.th = 0.0
    env.w = 0.0
    env.x = 0.0
    env.vx = 0.0

    # Action [1, 0] corresponds to throttle = 1.0 (20000 N), gimbal = 0
    # Mass = 800 + 200 = 1000 kg. ay = 20000 / 1000 - 9.81 = 10.19 m/s^2 > 0
    full_action = np.array([1.0, 0.0], dtype=np.float32)
    initial_vy = env.vy
    for _ in range(10):
        env.step(full_action)

    passed = env.vy > initial_vy
    print(f"Test 2 [Full Throttle Acceleration]: initial vy = {initial_vy:.2f}, new vy = {env.vy:.2f} m/s (ay > 0) -> {'PASS' if passed else 'FAIL'}")
    return passed


def test_gimbal_angular_acceleration():
    """
    Positive gimbal phi must produce NEGATIVE angular acceleration (torque = -L*T*sin(phi));
    negative phi the opposite.
    """
    env = RocketLandingEnv()
    env.reset(seed=42)
    env.th = 0.0
    env.w = 0.0

    # Positive gimbal: a1 = 1.0 -> phi = +15 deg
    # Throttle on (a0 = 1.0)
    pos_action = np.array([1.0, 1.0], dtype=np.float32)
    env.step(pos_action)
    alpha_pos = env.w  # starting from w = 0, delta w is alpha * dt
    pass_pos = alpha_pos < 0.0

    # Reset for negative gimbal: a1 = -1.0 -> phi = -15 deg
    env.reset(seed=42)
    env.th = 0.0
    env.w = 0.0
    neg_action = np.array([1.0, -1.0], dtype=np.float32)
    env.step(neg_action)
    alpha_neg = env.w
    pass_neg = alpha_neg > 0.0

    passed = pass_pos and pass_neg
    print(f"Test 3 [Gimbal Angular Acceleration]: pos phi alpha = {alpha_pos:.4f} (< 0: {pass_pos}), neg phi alpha = {alpha_neg:.4f} (> 0: {pass_neg}) -> {'PASS' if passed else 'FAIL'}")
    return passed


def test_fuel_consumption_and_exhaustion():
    """
    Fuel decreases by T/(Isp*g0)*dt per step and never goes below 0; with fuel = 0 thrust is 0.
    """
    env = RocketLandingEnv()
    env.reset(seed=42)
    initial_fuel = env.fuel
    T = env.T_max
    expected_dm = T / (env.Isp * env.g0) * env.dt

    # Full throttle 1 step
    full_action = np.array([1.0, 0.0], dtype=np.float32)
    env.step(full_action)
    actual_dm = initial_fuel - env.fuel
    dm_match = abs(actual_dm - expected_dm) < 1e-4

    # Deplete fuel completely
    env.fuel = 0.0
    env.vy = 0.0
    env.step(full_action)
    # With fuel = 0, thrust is 0, ay = -9.81 m/s^2, vy should decrease
    zero_thrust_ok = (env.fuel == 0.0) and (env.vy < 0.0)

    passed = dm_match and zero_thrust_ok
    print(f"Test 4 [Fuel Consumption & Depletion]: dm = {actual_dm:.6f} kg (expected {expected_dm:.6f} kg), zero fuel thrust = 0 -> {'PASS' if passed else 'FAIL'}")
    return passed


def test_pd_controller_stabilization():
    """
    A hand-written PD controller (gimbal = k1*theta + k2*omega) stabilises the tilt to near zero.
    """
    env = RocketLandingEnv()
    env.reset(seed=42)
    env.y = 120.0
    env.vy = 0.0
    env.th = 0.25  # initial tilt ~14.3 degrees
    env.w = 0.0
    env.x = 0.0
    env.vx = 0.0

    # PD gains for gimbal action a1 in [-1, 1]
    # Remember: positive phi produces negative angular acceleration.
    # So if th > 0 (tilted right), positive phi creates negative torque, turning rocket left toward upright.
    # gimbal_action = clip(k1 * th + k2 * w, -1, 1)
    k1 = 2.5
    k2 = 0.8

    # We provide enough throttle to maintain thrust for control
    # hover throttle ~ (m * g) / T_max = 1000 * 9.81 / 20000 = ~0.49
    # a0 = 2 * u - 1 = 2 * 0.5 - 1 = 0.0
    for step_i in range(150):  # 3.0 seconds
        gimbal_action = np.clip(k1 * env.th + k2 * env.w, -1.0, 1.0)
        action = np.array([0.0, gimbal_action], dtype=np.float32)
        env.step(action)
        if env.y <= 0:
            break

    final_tilt = abs(env.th)
    final_omega = abs(env.w)
    passed = (final_tilt < 0.05) and (final_omega < 0.05)
    print(f"Test 5 [PD Controller Stabilization]: final |tilt| = {final_tilt:.4f} rad ({np.degrees(final_tilt):.2f} deg), |omega| = {final_omega:.4f} rad/s -> {'PASS' if passed else 'FAIL'}")
    return passed


def run_all_tests():
    print("=" * 60)
    print("RUNNING ROCKET LANDER PHYSICS SANITY TESTS")
    print("=" * 60)
    results = [
        test_zero_throttle_freefall(),
        test_full_throttle_acceleration(),
        test_gimbal_angular_acceleration(),
        test_fuel_consumption_and_exhaustion(),
        test_pd_controller_stabilization()
    ]
    print("=" * 60)
    all_passed = all(results)
    if all_passed:
        print("ALL PHYSICS TESTS PASSED! Ready for RL training.")
    else:
        print("SOME PHYSICS TESTS FAILED! Fix issues before training.")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
