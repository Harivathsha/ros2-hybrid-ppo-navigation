"""Unit tests for the deterministic math in nav_learning.ros_nav_env."""

import math

import numpy as np
import pytest

from nav_learning.nav_math import (
    COLLISION_DISTANCE_M,
    GOAL_DISTANCE_CAP_M,
    LIDAR_BINS,
    MAX_ANGULAR_SPEED_RADPS,
    MAX_LINEAR_SPEED_MPS,
    OBSERVATION_SIZE,
    RewardSettings,
    InvalidScanError,
    build_observation,
    calculate_reward,
    determine_episode_outcome,
    policy_action_to_velocity,
    preprocess_scan,
    quaternion_from_yaw,
    relative_goal,
    velocity_to_policy_action,
    wrap_to_pi,
    yaw_from_quaternion,
)


def test_stationary_turns_have_an_accessible_speed_action_range():
    from nav_learning.nav_math import ActionSettings
    settings = ActionSettings(stop_fraction=0.15)
    for speed_action in (-0.99, -0.90, -0.75, -0.70):
        for turn in (-0.5, 0.5):
            linear, angular = policy_action_to_velocity([speed_action, turn], settings)
            assert linear == 0
            assert angular == pytest.approx(turn * MAX_ANGULAR_SPEED_RADPS)
    assert policy_action_to_velocity([1, 0], settings)[0] == pytest.approx(MAX_LINEAR_SPEED_MPS)
    assert policy_action_to_velocity([0, 0], settings)[0] > 0


def test_new_speed_mapping_is_continuous_and_round_trips_velocities():
    from nav_learning.nav_math import ActionSettings
    settings = ActionSettings(stop_fraction=0.15)
    speeds = [policy_action_to_velocity([a, 0], settings)[0] for a in np.linspace(-1, 1, 100)]
    assert np.all(np.diff(speeds) >= 0)
    assert policy_action_to_velocity([-0.7 + 1e-6, 0], settings)[0] < 1e-6
    for velocity in (0, 0.01, 0.10, MAX_LINEAR_SPEED_MPS):
        action = velocity_to_policy_action(velocity, -0.4, settings)
        np.testing.assert_allclose(policy_action_to_velocity(action, settings), [velocity, -0.4], atol=1e-7)


def test_detour_progress_rewards_sideways_escape_from_a_blocked_goal():
    from nav_learning.nav_math import DetourDistance, BOX_NAMES
    goal = (1.5, 0)
    field = DetourDistance({BOX_NAMES[0]: (0, 0, 0.4, 0)}, goal)
    previous, current = (-1.5, 0), (-1.5, 0.1)
    assert math.dist(current, goal) > math.dist(previous, goal)
    assert field.distance(current) < field.distance(previous)
    settings = RewardSettings(progress_mode="detour_potential")
    _, components = calculate_reward(math.dist(previous, goal), math.dist(current, goal),
        0, 1, 0.1, "running", reward_settings=settings,
        previous_detour_distance_m=field.distance(previous),
        current_detour_distance_m=field.distance(current))
    assert components["progress_reward"] > 0


def test_potential_shaping_telescopes_and_keeps_timeout_bootstrapping():
    settings = RewardSettings(progress_mode="detour_potential", discount_factor=0.99)
    distances = [4.0, 4.2, 3.0, 0.1]
    shaped = []
    for index in range(3):
        outcome = "success" if index == 2 else "running"
        _, parts = calculate_reward(2, 2, 0, 1, 0, outcome, reward_settings=settings,
            previous_detour_distance_m=distances[index], current_detour_distance_m=distances[index + 1])
        shaped.append(parts["progress_reward"])
    assert sum(settings.discount_factor ** i * r for i, r in enumerate(shaped)) == pytest.approx(4.0)
    _, timeout = calculate_reward(2, 2, 0, 1, 0, "timeout", reward_settings=settings,
        previous_detour_distance_m=4, current_detour_distance_m=3)
    assert timeout["progress_reward"] == pytest.approx(4 - 0.99 * 3)
    _, collision = calculate_reward(2, 2, 0, 0.21, 0, "collision", reward_settings=settings,
        previous_detour_distance_m=1000, current_detour_distance_m=1000)
    assert collision["total_reward"] < 0  # The bounded shaping cannot pay for crashing.


def test_route_distance_matches_single_box_geometry_and_reflection():
    from nav_learning.nav_math import DetourDistance, BOX_NAMES
    layout = {BOX_NAMES[0]: (0, 0, 0.4, 0)}
    field = DetourDistance(layout, (1.5, 0))
    offset = 0.801
    expected = 2 * math.hypot(1.5 - offset, offset) + 2 * offset
    assert field.distance((-1.5, 0)) == pytest.approx(expected)
    assert field.distance((-1.5, 0.4)) == pytest.approx(field.distance((-1.5, -0.4)))
    assert field.distance((1.5, 0)) == 0
    assert DetourDistance({}, (1, 1)).distance((0, 0)) == pytest.approx(math.sqrt(2))
    for point in ((-0.75, 0), (0, 0), (4.4, 0)):
        assert 0 <= field.distance(point) <= 20
    # Outside the circular collision margin, even a diagonal corner approach
    # must have a finite escape potential rather than jumping to the fallback.
    assert field.distance((-0.70, 0.70)) < 20


@pytest.mark.parametrize("section, values", [
    ("action", {"stop_fraction": -0.1}), ("action", {"stop_fraction": True}),
    ("action", {"stop_fraction": float("nan")}), ("reward", {"progress_mode": "typo"}),
    ("reward", {"discount_factor": 1.0}), ("reward", {"detour_clearance_m": 0.4})])
def test_invalid_new_settings_fail_early(section, values):
    from nav_learning.nav_math import ActionSettings
    with pytest.raises(ValueError):
        (ActionSettings if section == "action" else RewardSettings).from_mapping(values)


def test_tuned_obstacle_penalty_starts_earlier_and_discourages_close_approaches():
    """At 0.30 m the warning must outweigh a maximum-speed progress reward."""

    tuned = RewardSettings.from_mapping({"safety_distance_m": 0.50,
                                         "near_obstacle_penalty_scale": 0.20})
    def reward(clearance, settings=RewardSettings(), outcome="running"):
        return calculate_reward(2, 2 - MAX_LINEAR_SPEED_MPS * 0.1, 0, clearance,
                                MAX_LINEAR_SPEED_MPS, outcome, reward_settings=settings)
    old, old_parts = reward(0.45)
    new, new_parts = reward(0.45, tuned)
    assert old_parts["obstacle_penalty"] == 0
    assert new_parts["obstacle_penalty"] < 0
    assert new < old
    assert reward(0.30, tuned)[0] < 0
    assert reward(1.0, tuned) == reward(1.0)
    assert reward(0.21, tuned, "collision")[1]["terminal_reward"] == -25
    assert reward(1.0, tuned, "success")[1]["terminal_reward"] == 25


@pytest.mark.parametrize("values", [{"safety_distance_m": 0.22},
    {"safety_distance_m": float("nan")}, {"near_obstacle_penalty_scale": -0.1},
    {"saftey_distance_m": 0.5}])
def test_invalid_reward_settings_rejected(values):
    with pytest.raises(ValueError):
        RewardSettings.from_mapping(values)


def test_collision_threshold_is_no_longer_overly_conservative():
    collision = determine_episode_outcome(
        goal_distance_m=1.0,
        nearest_obstacle_m=COLLISION_DISTANCE_M,
        step_count=1,
        max_episode_steps=300,
    )
    running = determine_episode_outcome(
        goal_distance_m=1.0,
        nearest_obstacle_m=COLLISION_DISTANCE_M + 0.001,
        step_count=1,
        max_episode_steps=300,
    )
    assert collision == (True, False, "collision")
    assert running == (False, False, "running")


def test_wrap_to_pi_keeps_equivalent_angle():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    assert wrap_to_pi(2.0 * math.pi) == pytest.approx(0.0, abs=1e-7)
    assert abs(wrap_to_pi(3.0 * math.pi)) == pytest.approx(math.pi)


@pytest.mark.parametrize("yaw", [-2.5, -1.0, 0.0, 0.7, 2.5])
def test_yaw_quaternion_round_trip(yaw):
    x, y, z, w = quaternion_from_yaw(yaw)
    recovered = yaw_from_quaternion(x, y, z, w)
    assert recovered == pytest.approx(yaw, abs=1e-7)


def test_relative_goal_distance_and_bearing():
    distance, bearing = relative_goal(0.0, 0.0, 0.0, 3.0, 4.0)
    assert distance == pytest.approx(5.0)
    assert bearing == pytest.approx(math.atan2(4.0, 3.0))


def test_relative_goal_is_expressed_in_robot_frame():
    distance, bearing = relative_goal(
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=math.pi / 2.0,
        goal_x=2.0,
        goal_y=0.0,
    )
    assert distance == pytest.approx(2.0)
    assert bearing == pytest.approx(-math.pi / 2.0)


def test_preprocess_scan_output_contract():
    processed, nearest = preprocess_scan(np.full(360, 8.0, dtype=np.float32))
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (LIDAR_BINS,)
    assert processed.dtype == np.float32
    np.testing.assert_allclose(processed, 1.0)
    assert isinstance(nearest, float)
    assert nearest == pytest.approx(8.0)


def test_preprocess_scan_uses_minimum_pooling():
    scan = np.full(360, 8.0, dtype=np.float32)
    scan[7] = 0.50
    processed, nearest = preprocess_scan(scan)

    # With 360 rays and 72 bins, rays 5..9 form bin 1.
    expected = (0.50 - 0.15) / (8.0 - 0.15)
    assert processed[1] == pytest.approx(expected)
    assert nearest == pytest.approx(0.50)


def test_positive_infinity_means_no_lidar_return():
    processed, nearest = preprocess_scan(np.full(360, np.inf, dtype=np.float32))
    np.testing.assert_allclose(processed, 1.0)
    assert nearest == pytest.approx(8.0)


def test_small_number_of_invalid_rays_is_sanitised():
    scan = np.full(360, 4.0, dtype=np.float32)
    scan[0] = np.nan
    scan[1] = -np.inf
    processed, nearest = preprocess_scan(scan)
    assert np.isfinite(processed).all()
    assert nearest == pytest.approx(0.15)


def test_too_many_invalid_rays_are_rejected():
    scan = np.full(360, 4.0, dtype=np.float32)
    scan[:37] = np.nan  # More than 10 percent of 360 rays.
    with pytest.raises(InvalidScanError):
        preprocess_scan(scan)


def test_scan_with_too_few_rays_is_rejected():
    with pytest.raises(InvalidScanError):
        preprocess_scan(np.ones(LIDAR_BINS - 1, dtype=np.float32))


def test_build_observation_exact_order_and_type():
    lidar = np.linspace(0.0, 1.0, LIDAR_BINS, dtype=np.float32)
    observation = build_observation(
        lidar_normalised=lidar,
        goal_distance_m=GOAL_DISTANCE_CAP_M / 2.0,
        relative_goal_bearing_rad=math.pi / 2.0,
        linear_velocity_mps=MAX_LINEAR_SPEED_MPS / 2.0,
        angular_velocity_radps=-MAX_ANGULAR_SPEED_RADPS / 2.0,
    )

    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    np.testing.assert_allclose(observation[:LIDAR_BINS], lidar)
    assert observation[72] == pytest.approx(0.5)
    assert observation[73] == pytest.approx(1.0)
    assert observation[74] == pytest.approx(0.0, abs=1e-7)
    assert observation[75] == pytest.approx(0.5)
    assert observation[76] == pytest.approx(-0.5)


def test_build_observation_rejects_bad_shape_and_nonfinite_values():
    with pytest.raises(ValueError):
        build_observation(np.ones(71), 1.0, 0.0, 0.0, 0.0)

    lidar = np.ones(LIDAR_BINS, dtype=np.float32)
    lidar[0] = np.nan
    with pytest.raises(ValueError):
        build_observation(lidar, 1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("action", "expected_linear", "expected_angular"),
    [
        ([-1.0, 0.0], 0.0, 0.0),
        ([0.0, 0.5], MAX_LINEAR_SPEED_MPS / 2.0, MAX_ANGULAR_SPEED_RADPS / 2.0),
        ([1.0, -1.0], MAX_LINEAR_SPEED_MPS, -MAX_ANGULAR_SPEED_RADPS),
        ([2.0, -2.0], MAX_LINEAR_SPEED_MPS, -MAX_ANGULAR_SPEED_RADPS),
    ],
)
def test_policy_action_to_velocity(action, expected_linear, expected_angular):
    linear, angular = policy_action_to_velocity(action)
    assert linear == pytest.approx(expected_linear)
    assert angular == pytest.approx(expected_angular)


def test_policy_action_to_velocity_rejects_bad_input():
    with pytest.raises(ValueError):
        policy_action_to_velocity([0.0])
    with pytest.raises(ValueError):
        policy_action_to_velocity([np.nan, 0.0])


@pytest.mark.parametrize(
    ("linear", "angular", "expected"),
    [
        (0.0, 0.0, [-1.0, 0.0]),
        (MAX_LINEAR_SPEED_MPS / 2.0, MAX_ANGULAR_SPEED_RADPS / 2.0, [0.0, 0.5]),
        (MAX_LINEAR_SPEED_MPS, -MAX_ANGULAR_SPEED_RADPS, [1.0, -1.0]),
        (5.0, -5.0, [1.0, -1.0]),
    ],
)
def test_velocity_to_policy_action(linear, angular, expected):
    action = velocity_to_policy_action(linear, angular)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    np.testing.assert_allclose(action, expected, atol=1e-7)


def test_velocity_to_policy_action_rejects_nonfinite_input():
    with pytest.raises(ValueError):
        velocity_to_policy_action(np.nan, 0.0)
    with pytest.raises(ValueError):
        velocity_to_policy_action(0.0, np.inf)


@pytest.mark.parametrize(
    "original_action",
    [
        np.array([-1.0, -1.0], dtype=np.float32),
        np.array([-0.4, 0.7], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    ],
)
def test_action_velocity_round_trip(original_action):
    linear, angular = policy_action_to_velocity(original_action)
    recovered_action = velocity_to_policy_action(linear, angular)
    np.testing.assert_allclose(recovered_action, original_action, atol=1e-6)
