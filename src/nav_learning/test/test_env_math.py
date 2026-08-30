"""Unit tests for the deterministic math in nav_learning.ros_nav_env."""

import math

import numpy as np
import pytest

from nav_learning.ros_nav_env import (
    COLLISION_DISTANCE_M,
    GOAL_DISTANCE_CAP_M,
    LIDAR_BINS,
    MAX_ANGULAR_SPEED_RADPS,
    MAX_LINEAR_SPEED_MPS,
    OBSERVATION_SIZE,
    InvalidScanError,
    build_observation,
    build_route_waypoints,
    calculate_reward,
    determine_episode_outcome,
    navigation_path_exists,
    policy_action_to_velocity,
    preprocess_scan,
    quaternion_from_yaw,
    relative_goal,
    velocity_to_policy_action,
    wrap_to_pi,
    yaw_from_quaternion,
)


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


def test_route_waypoints_cross_each_required_doorway():
    assert build_route_waypoints("office", "office") == ()
    np.testing.assert_allclose(
        build_route_waypoints("office", "laboratory"),
        (
            (-2.65, 0.0),
            (-1.95, 0.0),
            (1.95, 0.0),
            (2.65, 0.0),
        ),
    )
    np.testing.assert_allclose(
        build_route_waypoints("laboratory", "office"),
        (
            (2.65, 0.0),
            (1.95, 0.0),
            (-1.95, 0.0),
            (-2.65, 0.0),
        ),
    )


def test_navigation_connectivity_detects_a_blocked_doorway():
    assert navigation_path_exists(
        start_xy=(-4.0, 0.0),
        goal_xy=(0.0, 0.0),
        layout={},
    )
    assert not navigation_path_exists(
        start_xy=(-4.0, 0.0),
        goal_xy=(0.0, 0.0),
        layout={
            "office_coffee_table": (-2.30, 0.0, 0.0, 0.0),
        },
    )


def test_waypoint_and_stuck_reward_components_sum():
    reward, components = calculate_reward(
        previous_goal_distance_m=1.0,
        current_goal_distance_m=1.0,
        relative_goal_bearing_rad=0.0,
        nearest_obstacle_m=1.0,
        commanded_linear_velocity_mps=0.0,
        outcome="running",
        waypoints_reached=1,
        stuck=True,
    )
    assert components["waypoint_reward"] == pytest.approx(2.0)
    assert components["stuck_penalty"] == pytest.approx(-0.20)
    assert reward == pytest.approx(
        components["progress_reward"]
        + components["heading_reward"]
        + components["step_penalty"]
        + components["obstacle_penalty"]
        + components["waypoint_reward"]
        + components["stuck_penalty"]
        + components["terminal_reward"]
    )


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
