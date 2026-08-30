"""Common ROS 2 navigation environment for PPO and Behavior Cloning.

This module provides deterministic observation/action utilities, ROS 2 sensor
integration, connected seeded resets, doorway route targets, reward shaping,
and Gym-style episode termination for the three-room training world.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence

import numpy as np

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import LaserScan


LIDAR_BINS = 72
OBSERVATION_SIZE = 77

LIDAR_MIN_M = 0.15
LIDAR_CAP_M = 8.0
MAX_INVALID_SCAN_FRACTION = 0.10
MAX_INVALID_SCAN_RETRIES = 3
# The world interior is approximately 14 m × 10 m.
# Its corner-to-corner distance is roughly 17.2 m.
GOAL_DISTANCE_CAP_M = 17.0

MAX_LINEAR_SPEED_MPS = 0.22

MAX_ANGULAR_SPEED_RADPS = 1.20

SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/cmd_vel"

SET_POSE_SERVICE = "/world/rl_office_training_world/set_pose"

STARTUP_TIMEOUT_SEC = 8.0
FRESH_SENSOR_TIMEOUT_SEC = 2.0

# One policy action remains active for this much simulation time.
CONTROL_INTERVAL_SEC = 0.10
STEP_SENSOR_TIMEOUT_SEC = 2.0
NANOSECONDS_PER_SECOND = 1_000_000_000

# ------------------------- Reward/termination contract ---------------------

GOAL_REACHED_DISTANCE_M = 0.25

# LiDAR is mounted near the chassis centre.  The chassis corner is about
# 0.173 m from that centre, so 0.22 m retains a small physical and timing
# margin without ending episodes at the old, overly conservative 0.30 m.
COLLISION_DISTANCE_M = 0.22

# Start discouraging the robot before reaching collision distance.
SAFETY_DISTANCE_M = 0.50

MAX_EPISODE_STEPS = {
    0: 300,
    1: 600,
    2: 1200,
}

PROGRESS_REWARD_SCALE = 5.0
HEADING_REWARD_SCALE = 0.02
STEP_PENALTY = -0.01
NEAR_OBSTACLE_PENALTY_SCALE = 0.20
ROUTE_WAYPOINT_REWARD = 2.0
STUCK_PENALTY = -0.20

SUCCESS_REWARD = 25.0
COLLISION_PENALTY = -25.0
TIMEOUT_PENALTY = -5.0


# --------------------------- Episode reset contract -------------------------

ROBOT_NAME = "hv_bot"
GOAL_MARKER_NAME = "goal_marker"

ROBOT_RESET_Z_M = 0.20
GOAL_MARKER_Z_M = 0.018

SET_POSE_TIMEOUT_SEC = 2.0
RESET_SENSOR_TIMEOUT_SEC = 2.0

MAX_RESET_RETRIES = 3
MAX_SAMPLING_ATTEMPTS = 200
MAX_LAYOUT_SAMPLING_ATTEMPTS = 30

ROBOT_SAMPLE_RADIUS_M = 0.32
SAMPLING_CLEARANCE_M = 0.08

RESET_POSITION_TOLERANCE_M = 0.20
RESET_YAW_TOLERANCE_RAD = 0.30


# Number of objects moved according to curriculum difficulty.
RANDOMIZED_OBJECT_COUNT = {
    0: 2,  # early training
    1: 4,  # intermediate training
    2: 6,  # full difficulty
}


# Allowed start and goal distance for each curriculum level.
CURRICULUM_DISTANCE_LIMITS_M = {
    0: (0.80, 2.50),
    1: (1.50, 5.50),
    2: (2.50, math.inf),
}


ROOM_BOUNDS = {
    "office": (-6.55, -2.70, -4.55, 4.55),
    "meeting": (-1.90, 1.90, -4.55, 4.55),
    "laboratory": (2.70, 6.55, -4.55, 4.55),
}

OBJECT_ROOM_BOUNDS = {
    "office": (-6.15, -3.15, -4.10, 4.10),
    "meeting": (-1.55, 1.55, -4.10, 4.10),
    "laboratory": (3.15, 6.15, -4.10, 4.10),
}

# Format: (name, x, y, approximate safety radius)
STATIC_OBSTACLES = (
    # Office
    ("office_desk", -5.35, 2.85, 0.95),
    ("office_bookshelf", -6.60, -2.55, 1.25),
    ("office_sofa", -4.70, -4.05, 1.05),

    # Meeting room
    ("meeting_round_table", 0.85, 3.05, 0.85),
    ("meeting_sofa", 1.05, -4.02, 1.05),
    ("meeting_side_cabinet", -1.72, -3.65, 0.75),

    # Laboratory
    ("lab_workbench", 5.35, 3.85, 1.15),
    ("lab_storage_rack", 6.55, 0.80, 1.12),
    ("lab_tall_cabinet", 5.85, -0.85, 0.68),
)

# Format: (x, y, keep-out radius)
DOORWAY_KEEP_OUTS = (
    (-2.30, 0.0, 0.80),
    (2.30, 0.0, 0.80),
)

# The three rooms form a line.  Each partition has a 2.0 m doorway centred
# at y=0.  Approach/exit waypoints make the local PPO controller target the
# opening instead of being attracted to a wall corner by the final goal.
ROOM_SEQUENCE = ("office", "meeting", "laboratory")
PARTITION_X_M = (-2.30, 2.30)
DOORWAY_HALF_WIDTH_M = 1.0
PARTITION_HALF_THICKNESS_M = 0.075
DOORWAY_CROSSING_OFFSET_M = 0.35
ROUTE_WAYPOINT_REACHED_DISTANCE_M = 0.30

# A compact grid reachability check rejects randomized layouts with no safe
# route.  Obstacles are inflated by the LiDAR collision threshold plus a
# small discretization margin.
PATH_GRID_RESOLUTION_M = 0.10
PATH_EXTRA_MARGIN_M = 0.03
OUTER_WALL_INNER_X_M = 6.925
OUTER_WALL_INNER_Y_M = 4.925

# Eight seconds of negligible translation is treated as being stuck.  This
# is a reward signal only; the episode remains active so PPO can recover.
STUCK_WINDOW_STEPS = 80
STUCK_TRANSLATION_M = 0.15

RANDOMIZABLE_OBJECTS = {
    # Office
    "office_chair": {
        "room": "office",
        "home_pose": (-4.15, 2.85, 0.0, math.pi),
        "radius": 0.42,
    },
    "office_coffee_table": {
        "room": "office",
        "home_pose": (-4.65, -2.75, 0.0, 0.0),
        "radius": 0.65,
    },
    "office_plant": {
        "room": "office",
        "home_pose": (-3.15, 3.95, 0.0, 0.0),
        "radius": 0.45,
    },

    # Meeting room
    "meeting_chair_east": {
        "room": "meeting",
        "home_pose": (1.85, 3.05, 0.0, math.pi),
        "radius": 0.42,
    },
    "meeting_chair_south": {
        "room": "meeting",
        "home_pose": (0.85, 1.95, 0.0, math.pi / 2.0),
        "radius": 0.42,
    },
    "moving_service_cart": {
        "room": "meeting",
        "home_pose": (-0.65, -3.40, 0.30, 0.0),
        "radius": 0.48,
    },

    # Laboratory
    "lab_crate_a": {
        "room": "laboratory",
        "home_pose": (3.40, 3.85, 0.32, 0.18),
        "radius": 0.55,
    },
    "lab_crate_b": {
        "room": "laboratory",
        "home_pose": (4.05, 4.15, 0.24, -0.22),
        "radius": 0.45,
    },
    "lab_stool": {
        "room": "laboratory",
        "home_pose": (4.20, 2.85, 0.0, 0.0),
        "radius": 0.35,
    },
    "moving_cleaner": {
        "room": "laboratory",
        "home_pose": (3.35, -3.25, 0.13, 0.0),
        "radius": 0.40,
    },
}


def build_route_waypoints(
    start_room: str,
    goal_room: str,
) -> tuple[tuple[float, float], ...]:
    """Return ordered doorway approach/exit targets between two rooms."""

    if start_room not in ROOM_SEQUENCE:
        raise ValueError(f"Unknown start room: {start_room}")
    if goal_room not in ROOM_SEQUENCE:
        raise ValueError(f"Unknown goal room: {goal_room}")

    start_index = ROOM_SEQUENCE.index(start_room)
    goal_index = ROOM_SEQUENCE.index(goal_room)
    if start_index == goal_index:
        return ()

    direction = 1 if goal_index > start_index else -1
    current_index = start_index
    waypoints: list[tuple[float, float]] = []

    while current_index != goal_index:
        boundary_index = (
            current_index
            if direction > 0
            else current_index - 1
        )
        partition_x = PARTITION_X_M[boundary_index]
        approach_x = (
            partition_x
            - direction * DOORWAY_CROSSING_OFFSET_M
        )
        exit_x = (
            partition_x
            + direction * DOORWAY_CROSSING_OFFSET_M
        )
        waypoints.extend(
            (
                (float(approach_x), 0.0),
                (float(exit_x), 0.0),
            )
        )
        current_index += direction

    return tuple(waypoints)


def _layout_obstacle_circles(
    layout: Mapping[str, tuple[float, float, float, float]],
) -> tuple[tuple[float, float, float], ...]:
    """Convert randomized-object poses to circular planning obstacles."""

    circles = []
    for name, pose in layout.items():
        if name not in RANDOMIZABLE_OBJECTS:
            raise ValueError(f"Unknown randomized object: {name}")
        x, y, _, _ = pose
        radius = float(RANDOMIZABLE_OBJECTS[name]["radius"])
        circles.append((float(x), float(y), radius))
    return tuple(circles)


def navigation_path_exists(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    layout: Mapping[str, tuple[float, float, float, float]],
    resolution_m: float = PATH_GRID_RESOLUTION_M,
) -> bool:
    """Check start-goal reachability on an inflated four-connected grid."""

    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("resolution_m must be positive and finite.")

    clearance_m = COLLISION_DISTANCE_M + PATH_EXTRA_MARGIN_M
    minimum_x = -OUTER_WALL_INNER_X_M + clearance_m
    maximum_x = OUTER_WALL_INNER_X_M - clearance_m
    minimum_y = -OUTER_WALL_INNER_Y_M + clearance_m
    maximum_y = OUTER_WALL_INNER_Y_M - clearance_m

    x_coordinates = np.arange(
        minimum_x,
        maximum_x + resolution_m * 0.5,
        resolution_m,
        dtype=np.float32,
    )
    y_coordinates = np.arange(
        minimum_y,
        maximum_y + resolution_m * 0.5,
        resolution_m,
        dtype=np.float32,
    )
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
    blocked = np.zeros(grid_x.shape, dtype=bool)

    obstacle_circles = [
        (float(x), float(y), float(radius))
        for _, x, y, radius in STATIC_OBSTACLES
    ]
    obstacle_circles.extend(_layout_obstacle_circles(layout))

    for obstacle_x, obstacle_y, obstacle_radius in obstacle_circles:
        inflated_radius = obstacle_radius + clearance_m
        blocked |= (
            (grid_x - obstacle_x) ** 2
            + (grid_y - obstacle_y) ** 2
            <= inflated_radius ** 2
        )

    wall_half_width = PARTITION_HALF_THICKNESS_M + clearance_m
    usable_doorway_half_width = max(
        DOORWAY_HALF_WIDTH_M - clearance_m,
        0.0,
    )
    outside_doorway = np.abs(grid_y) >= usable_doorway_half_width
    for partition_x in PARTITION_X_M:
        blocked |= (
            (np.abs(grid_x - partition_x) <= wall_half_width)
            & outside_doorway
        )

    def point_to_cell(
        point_xy: tuple[float, float],
    ) -> tuple[int, int] | None:
        point_x, point_y = point_xy
        if not np.isfinite([point_x, point_y]).all():
            raise ValueError("Path endpoints must be finite.")
        column = int(round((point_x - minimum_x) / resolution_m))
        row = int(round((point_y - minimum_y) / resolution_m))
        if not (
            0 <= row < blocked.shape[0]
            and 0 <= column < blocked.shape[1]
        ):
            return None
        return row, column

    start_cell = point_to_cell(start_xy)
    goal_cell = point_to_cell(goal_xy)
    if start_cell is None or goal_cell is None:
        return False
    if blocked[start_cell] or blocked[goal_cell]:
        return False
    if start_cell == goal_cell:
        return True

    visited = np.zeros(blocked.shape, dtype=bool)
    visited[start_cell] = True
    pending = deque([start_cell])
    neighbour_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while pending:
        row, column = pending.popleft()
        for row_offset, column_offset in neighbour_offsets:
            neighbour_row = row + row_offset
            neighbour_column = column + column_offset
            if not (
                0 <= neighbour_row < blocked.shape[0]
                and 0 <= neighbour_column < blocked.shape[1]
            ):
                continue
            if blocked[neighbour_row, neighbour_column]:
                continue
            if visited[neighbour_row, neighbour_column]:
                continue
            if (neighbour_row, neighbour_column) == goal_cell:
                return True
            visited[neighbour_row, neighbour_column] = True
            pending.append((neighbour_row, neighbour_column))

    return False


class InvalidScanError(ValueError):
    """Raised when a LaserScan cannot safely form an observation."""


def wrap_to_pi(angle: float) -> float:

    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(
    x: float,
    y: float,
    z: float,
    w: float,
) -> float:

    numerator = 2 * (w * z + x * y)
    denominator = 1 - 2 * (y * y + z * z)
    yaw = float(math.atan2(numerator, denominator))

    return yaw


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:

    x = 0.0
    y = 0.0
    z = float(math.sin(yaw / 2))
    w = float(math.cos(yaw / 2))

    return x, y, z, w


def relative_goal(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    goal_x: float,
    goal_y: float,
) -> tuple[float, float]:


    dx = goal_x - robot_x
    dy = goal_y - robot_y

    distance = math.hypot(dx, dy)
    world_bearing = math.atan2(dy, dx)
    relative_bearing = wrap_to_pi(world_bearing - robot_yaw)

    return distance, relative_bearing

def preprocess_scan(
    ranges: Sequence[float],
    range_min_m: float = LIDAR_MIN_M,
    range_cap_m: float = LIDAR_CAP_M,
    output_bins: int = LIDAR_BINS,
) -> tuple[np.ndarray, float]:
    ranges_array = np.array(ranges, dtype=np.float32)

    if ranges_array.ndim != 1:
        raise InvalidScanError("Scan is not one-dimensional.")
    if ranges_array.size < output_bins: 
        raise InvalidScanError("Scan has fewer values than output_bins.")
    if range_cap_m <= range_min_m:
        raise InvalidScanError("range_cap_m must be greater than range_min_m.") 

    invalid_mask = np.isnan(ranges_array) | np.isneginf(ranges_array)
    invalid_fraction = float(np.mean(invalid_mask))

    if invalid_fraction > MAX_INVALID_SCAN_FRACTION:
        raise InvalidScanError("Invalid scan fraction exceeds maximum allowed.")    

    ranges_array = np.nan_to_num(ranges_array, nan=range_cap_m, posinf=range_cap_m, neginf=range_min_m) 
    ranges_array = np.clip(ranges_array, range_min_m, range_cap_m)


    nearest_obstacle_m = np.min(ranges_array)

    sectors = np.array_split(ranges_array, output_bins)

    pooled_distances = np.asarray([np.min(sector) for sector in sectors])

    lidar_normalised = (
        (pooled_distances - range_min_m)
        / (range_cap_m - range_min_m)
    )

    
    return lidar_normalised.astype(np.float32), float(nearest_obstacle_m)

def build_observation(
    lidar_normalised: np.ndarray,
    goal_distance_m: float,
    relative_goal_bearing_rad: float,
    linear_velocity_mps: float,
    angular_velocity_radps: float,
) -> np.ndarray:

    if lidar_normalised.shape != (LIDAR_BINS,):
        raise ValueError(f"lidar_normalised must have shape ({LIDAR_BINS},)")
    
    if not np.isfinite(goal_distance_m):
        raise ValueError("goal_distance_m must be finite")
    
    if not np.isfinite(relative_goal_bearing_rad):
        raise ValueError("relative_goal_bearing_rad must be finite")
    
    goal_distance_norm = np.clip(goal_distance_m / GOAL_DISTANCE_CAP_M, 0, 1)

    linear_norm = np.clip(linear_velocity_mps / MAX_LINEAR_SPEED_MPS, -1, 1)
    angular_norm = np.clip(angular_velocity_radps / MAX_ANGULAR_SPEED_RADPS, -1, 1)

    concatenate = np.concatenate([
        lidar_normalised,
        [goal_distance_norm],
        [np.sin(relative_goal_bearing_rad)],
        [np.cos(relative_goal_bearing_rad)],
        [linear_norm],
        [angular_norm]
    ])

    if concatenate.shape != (OBSERVATION_SIZE,) or not np.isfinite(concatenate).all():
        raise ValueError(f"Observation must have shape ({OBSERVATION_SIZE},)")
    
    return concatenate.astype(np.float32)

def policy_action_to_velocity(
    action: Sequence[float],
) -> tuple[float, float]:

    action_array = np.asarray(action, dtype=np.float32)

    if action_array.shape != (2,):
        raise ValueError("Action must have shape (2,)")

    if not np.isfinite(action_array).all():
        raise ValueError("Action must contain finite values")

    clipped_action = np.clip(action_array, -1, 1)

    linear = (clipped_action[0] + 1) / 2 * MAX_LINEAR_SPEED_MPS
    angular = clipped_action[1] * MAX_ANGULAR_SPEED_RADPS

    return float(linear), float(angular)

def velocity_to_policy_action(
    linear_velocity_mps: float,
    angular_velocity_radps: float,
) -> np.ndarray:

    if not np.isfinite(
        [linear_velocity_mps, angular_velocity_radps]
    ).all():
        raise ValueError("Velocities must contain finite values")
    clipped_linear = np.clip(linear_velocity_mps, 0, MAX_LINEAR_SPEED_MPS)
    clipped_angular = np.clip(angular_velocity_radps, -MAX_ANGULAR_SPEED_RADPS, MAX_ANGULAR_SPEED_RADPS)

    linear_action = 2 * clipped_linear / MAX_LINEAR_SPEED_MPS - 1
    angular_action = clipped_angular / MAX_ANGULAR_SPEED_RADPS

    return np.array([linear_action, angular_action], dtype=np.float32)

def determine_episode_outcome(
    goal_distance_m: float,
    nearest_obstacle_m: float,
    step_count: int,
    max_episode_steps: int,
) -> tuple[bool, bool, str]:
    """Return Gym-style termination flags and one outcome name."""

    if not np.isfinite(
        [goal_distance_m, nearest_obstacle_m]
    ).all():
        raise ValueError("Outcome inputs must be finite.")

    if goal_distance_m < 0.0 or nearest_obstacle_m < 0.0:
        raise ValueError("Distances cannot be negative.")

    if step_count < 1 or max_episode_steps < 1:
        raise ValueError(
            "Step limits must be positive integers."
        )

    # Collision receives priority for safety.
    if nearest_obstacle_m <= COLLISION_DISTANCE_M:
        return True, False, "collision"

    if goal_distance_m <= GOAL_REACHED_DISTANCE_M:
        return True, False, "success"

    if step_count >= max_episode_steps:
        return False, True, "timeout"

    return False, False, "running"

def calculate_reward(
    previous_goal_distance_m: float,
    current_goal_distance_m: float,
    relative_goal_bearing_rad: float,
    nearest_obstacle_m: float,
    commanded_linear_velocity_mps: float,
    outcome: str,
    waypoints_reached: int = 0,
    stuck: bool = False,
) -> tuple[float, dict[str, float]]:
    """Calculate transparent reward components for one transition."""

    values = [
        previous_goal_distance_m,
        current_goal_distance_m,
        relative_goal_bearing_rad,
        nearest_obstacle_m,
        commanded_linear_velocity_mps,
    ]

    if not np.isfinite(values).all():
        raise ValueError("Reward inputs must be finite.")

    if previous_goal_distance_m < 0.0:
        raise ValueError(
            "Previous goal distance cannot be negative."
        )

    if current_goal_distance_m < 0.0:
        raise ValueError(
            "Current goal distance cannot be negative."
        )

    if nearest_obstacle_m < 0.0:
        raise ValueError(
            "Nearest-obstacle distance cannot be negative."
        )

    if isinstance(waypoints_reached, bool) or not isinstance(
        waypoints_reached,
        int,
    ):
        raise TypeError("waypoints_reached must be an integer.")
    if waypoints_reached < 0:
        raise ValueError("waypoints_reached cannot be negative.")
    if not isinstance(stuck, bool):
        raise TypeError("stuck must be a bool.")

    terminal_rewards = {
        "running": 0.0,
        "success": SUCCESS_REWARD,
        "collision": COLLISION_PENALTY,
        "timeout": TIMEOUT_PENALTY,
    }

    if outcome not in terminal_rewards:
        raise ValueError(
            f"Unknown episode outcome: {outcome}"
        )

    # Positive when the robot moved closer to the goal.
    progress_m = (
        previous_goal_distance_m
        - current_goal_distance_m
    )

    progress_reward = (
        PROGRESS_REWARD_SCALE * progress_m
    )

    forward_fraction = float(
        np.clip(
            commanded_linear_velocity_mps
            / MAX_LINEAR_SPEED_MPS,
            0.0,
            1.0,
        )
    )

    # Reward alignment only while moving forward.
    # This prevents the robot from earning reward by standing still.
    heading_reward = (
        HEADING_REWARD_SCALE
        * forward_fraction
        * math.cos(relative_goal_bearing_rad)
    )

    obstacle_penalty = 0.0

    if nearest_obstacle_m < SAFETY_DISTANCE_M:
        unsafe_fraction = float(
            np.clip(
                (
                    SAFETY_DISTANCE_M
                    - nearest_obstacle_m
                )
                / (
                    SAFETY_DISTANCE_M
                    - COLLISION_DISTANCE_M
                ),
                0.0,
                1.0,
            )
        )

        obstacle_penalty = (
            -NEAR_OBSTACLE_PENALTY_SCALE
            * unsafe_fraction
        )

    terminal_reward = terminal_rewards[outcome]
    waypoint_reward = ROUTE_WAYPOINT_REWARD * waypoints_reached
    stuck_penalty = (
        STUCK_PENALTY
        if stuck and outcome == "running"
        else 0.0
    )

    total_reward = float(
        progress_reward
        + heading_reward
        + STEP_PENALTY
        + obstacle_penalty
        + waypoint_reward
        + stuck_penalty
        + terminal_reward
    )

    components = {
        "progress_m": float(progress_m),
        "progress_reward": float(progress_reward),
        "heading_reward": float(heading_reward),
        "step_penalty": float(STEP_PENALTY),
        "obstacle_penalty": float(obstacle_penalty),
        "waypoint_reward": float(waypoint_reward),
        "stuck_penalty": float(stuck_penalty),
        "terminal_reward": float(terminal_reward),
        "total_reward": total_reward,
    }

    return total_reward, components

class RosNavEnv(Node):
    """Synchronous learning environment around asynchronous ROS callbacks."""

    def __init__(self) -> None:
        super().__init__("ros_nav_env")

        # Condition protects the sensor messages and counters.
        self._sensor_condition = threading.Condition()

        self._latest_scan: LaserScan | None = None
        self._latest_odom: Odometry | None = None

        self._scan_count = 0
        self._odom_count = 0

        self._scan_stamp_ns: int | None = None
        self._odom_stamp_ns: int | None = None

        self._closed = False
        self.goal_xy: tuple[float, float] | None = None
        self._previous_goal_distance_m: float | None = None
        self._route_waypoints_xy: tuple[
            tuple[float, float],
            ...,
        ] = ()
        self._route_waypoint_index = 0
        self._position_history: deque[
            tuple[float, float]
        ] = deque(maxlen=STUCK_WINDOW_STEPS)

        self._episode_seed: int | None = None
        self._step_count = 0

        self._current_layout: dict[
            str,
            tuple[float, float, float, float],
        ] = {}

        self._randomized_objects: tuple[str, ...] = ()
        self._episode_active = False
        self._curriculum_level: int | None = None

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            CMD_VEL_TOPIC,
            10,
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self._scan_callback,
            qos_profile_sensor_data,
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self._odom_callback,
            qos_profile_sensor_data,
        )

        self.set_pose_client = self.create_client(
            SetEntityPose,
            SET_POSE_SERVICE,
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)

        self._executor_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
        )
        self._executor_thread.start()

        try:
            self._startup_check()
        except Exception:
            self.close()
            raise




    @staticmethod
    def _stamp_to_nanoseconds(stamp) -> int:
        """Convert a ROS timestamp into one integer."""

        if not hasattr(stamp, "sec") or not hasattr(stamp, "nanosec"):
            raise ValueError("Stamp must have 'sec' and 'nanosec' attributes")

        if not isinstance(stamp.sec, int) or not isinstance(stamp.nanosec, int):
            raise ValueError("'sec' and 'nanosec' must be integers")

        return stamp.sec * 1_000_000_000 + stamp.nanosec
    

    def _scan_callback(self, message: LaserScan) -> None:
        """Cache the newest LiDAR message."""

        with self._sensor_condition:
            self._latest_scan = message
            self._scan_count += 1
            self._scan_stamp_ns = self._stamp_to_nanoseconds(message.header.stamp)
            self._sensor_condition.notify_all()

    def _odom_callback(self, message: Odometry) -> None:
        """Cache the newest odometry message."""

        with self._sensor_condition:
            self._latest_odom = message
            self._odom_count += 1
            self._odom_stamp_ns = self._stamp_to_nanoseconds(message.header.stamp)
            self._sensor_condition.notify_all()

    def _wait_for_first_sensors(
        self,
        timeout_sec: float,
    ) -> tuple[LaserScan, Odometry]:
        """Wait until both sensor streams have produced a message."""

        with self._sensor_condition:

            success = self._sensor_condition.wait_for(
                lambda: self._latest_scan is not None and self._latest_odom is not None,
                timeout=timeout_sec
            )
            if not success:
                raise TimeoutError("Timeout waiting for first scan and odometry messages.")

            return self._latest_scan, self._latest_odom
        
    def _wait_for_sensor_time(
        self,
        target_stamp_ns: int,
        timeout_sec: float,
    ) -> tuple[LaserScan, Odometry]:
        """Wait until both sensor streams reach a simulation timestamp."""

        with self._sensor_condition:
            success = self._sensor_condition.wait_for(
                lambda: (
                    self._latest_scan is not None
                    and self._latest_odom is not None
                    and self._scan_stamp_ns is not None
                    and self._odom_stamp_ns is not None
                    and self._scan_stamp_ns >= target_stamp_ns
                    and self._odom_stamp_ns >= target_stamp_ns
                ),
                timeout=timeout_sec,
            )

            if not success:
                raise TimeoutError(
                    "Timeout waiting for the action control interval."
                )

            return self._latest_scan, self._latest_odom
        
    def _wait_for_fresh_sensors(
        self,
        previous_scan_count: int,
        previous_odom_count: int,
        timeout_sec: float,
    ) -> tuple[LaserScan, Odometry]:
        """Wait for scan and odometry newer than captured counters."""

        with self._sensor_condition:
            success = self._sensor_condition.wait_for(
                lambda: self._scan_count > previous_scan_count and self._odom_count > previous_odom_count,
                timeout=timeout_sec
            )
            if not success:
                raise TimeoutError("Timeout waiting for fresh scan and odometry messages.")

            return self._latest_scan, self._latest_odom

    def _startup_check(self) -> None:
        """Verify essential simulation interfaces before training."""

        if not self.set_pose_client.wait_for_service(
            timeout_sec=STARTUP_TIMEOUT_SEC
        ):
            raise TimeoutError(
                f"Timeout waiting for set-pose service {SET_POSE_SERVICE}."
            )

        first_scan, _ = self._wait_for_first_sensors(STARTUP_TIMEOUT_SEC)
        if len(first_scan.ranges) < LIDAR_BINS:
            raise InvalidScanError(
                f"Initial scan has fewer than {LIDAR_BINS} rays."
            )
        # Do not validate scan contents before reset(). A previous interrupted
        # run may have left the robot at a geometrically invalid pose.
        # reset() teleports the robot to a valid start and then performs the
        # complete LiDAR validation.

        with self._sensor_condition:
            scan_count = self._scan_count
            odom_count = self._odom_count
            scan_stamp_ns = self._scan_stamp_ns
            odom_stamp_ns = self._odom_stamp_ns

        fresh_scan, fresh_odom = self._wait_for_fresh_sensors(
            scan_count,
            odom_count,
            FRESH_SENSOR_TIMEOUT_SEC,
        )

        fresh_scan_stamp_ns = self._stamp_to_nanoseconds(fresh_scan.header.stamp)
        fresh_odom_stamp_ns = self._stamp_to_nanoseconds(fresh_odom.header.stamp)
        if fresh_scan_stamp_ns == scan_stamp_ns:
            raise RuntimeError("Fresh scan timestamp did not change.")
        if fresh_odom_stamp_ns == odom_stamp_ns:
            raise RuntimeError("Fresh odometry timestamp did not change.")

        self.get_logger().info(
            "Navigation environment startup check passed: set-pose service, "
            "LiDAR, and odometry are ready."
        )

    def _set_entity_pose(
        self,
        entity_name: str,
        x: float,
        y: float,
        z: float,
        yaw: float,
    ) -> None:
        """Teleport one Gazebo model and verify the service response."""

        request = SetEntityPose.Request()

        request.entity.name = entity_name
        request.entity.type = Entity.MODEL

        request.pose.position.x = float(x)
        request.pose.position.y = float(y)
        request.pose.position.z = float(z)

        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        request.pose.orientation.x = qx
        request.pose.orientation.y = qy
        request.pose.orientation.z = qz
        request.pose.orientation.w = qw

        future = self.set_pose_client.call_async(request)
        deadline = time.monotonic() + SET_POSE_TIMEOUT_SEC
        while not future.done():
            if time.monotonic() >= deadline:
                future.cancel()
                raise TimeoutError(
                    f"Timeout waiting for set-pose service response for {entity_name}."
                )
            time.sleep(0.01)

        service_exception = future.exception()
        if service_exception is not None:
            raise RuntimeError(
                f"SetEntityPose service call failed for {entity_name}: "
                f"{service_exception}"
            ) from service_exception

        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(
                f"Gazebo rejected the pose requested for '{entity_name}'."
            )

    @staticmethod
    def _sample_from_bounds(
        rng: np.random.Generator,
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """Sample one continuous x-y position from rectangular bounds."""

        minimum_x, maximum_x, minimum_y, maximum_y = bounds

        x = float(rng.uniform(minimum_x, maximum_x))
        y = float(rng.uniform(minimum_y, maximum_y))

        return x, y

    @staticmethod
    def _is_point_clear(
        x: float,
        y: float,
        candidate_radius: float,
        occupied: Sequence[tuple[float, float, float]],
    ) -> bool:
        """Check clearance from static and already-positioned objects."""

        # STATIC_OBSTACLES entries contain: (name, x, y, radius).
        for _, obstacle_x, obstacle_y, obstacle_radius in STATIC_OBSTACLES:
            separation = math.hypot(x - obstacle_x, y - obstacle_y)

            required_separation = (
                candidate_radius + obstacle_radius + SAMPLING_CLEARANCE_M
            )

            if separation < required_separation:
                return False

        # Occupied entries contain only: (x, y, radius). This includes
        # doorway keep-outs, placed objects, and the sampled start pose.
        for occupied_x, occupied_y, occupied_radius in occupied:
            separation = math.hypot(x - occupied_x, y - occupied_y)
            required_separation = (
                candidate_radius + occupied_radius + SAMPLING_CLEARANCE_M
            )

            if separation < required_separation:
                return False

        return True

    def _sample_clear_point(
        self,
        rng: np.random.Generator,
        bounds: tuple[float, float, float, float],
        candidate_radius: float,
        occupied: Sequence[tuple[float, float, float]],
    ) -> tuple[float, float]:
        """Sample a clear point within the given bounds."""

        for _ in range(MAX_SAMPLING_ATTEMPTS):
            x, y = self._sample_from_bounds(rng, bounds)
            if self._is_point_clear(x, y, candidate_radius, occupied):
                return x, y

        raise RuntimeError(
            "Unable to sample a collision-free point after "
            f"{MAX_SAMPLING_ATTEMPTS} attempts."
        )


    def _sample_object_layout(
        self,
        rng: np.random.Generator,
        curriculum_level: int,
    ) -> tuple[
        dict[str, tuple[float, float, float, float]],
        tuple[str, ...],
        list[tuple[float, float, float]],
    ]:
        """Choose objects and generate their final episode poses."""

        if curriculum_level not in RANDOMIZED_OBJECT_COUNT:
            raise ValueError(f"Invalid curriculum level: {curriculum_level}")

        randomized_count = RANDOMIZED_OBJECT_COUNT[curriculum_level]
        object_names = tuple(RANDOMIZABLE_OBJECTS)

        selected_array = rng.choice(
            object_names,
            size=randomized_count,
            replace=False,
        )

        randomized_objects = tuple(
            str(name) for name in selected_array.tolist()
        )
        randomized_set = set(randomized_objects)

        layout: dict[str, tuple[float, float, float, float]] = {}
        occupied: list[tuple[float, float, float]] = list(
            DOORWAY_KEEP_OUTS
        )

        for name, specification in RANDOMIZABLE_OBJECTS.items():
            if name in randomized_set:
                continue

            home_pose = specification["home_pose"]
            radius = float(specification["radius"])

            home_x, home_y, home_z, home_yaw = home_pose
            layout[name] = (
                float(home_x),
                float(home_y),
                float(home_z),
                float(home_yaw),
            )
            occupied.append((float(home_x), float(home_y), radius))

        for name in randomized_objects:
            specification = RANDOMIZABLE_OBJECTS[name]
            room_name = str(specification["room"])
            radius = float(specification["radius"])
            _, _, z, _ = specification["home_pose"]

            x, y = self._sample_clear_point(
                rng=rng,
                bounds=OBJECT_ROOM_BOUNDS[room_name],
                candidate_radius=radius,
                occupied=occupied,
            )

            yaw = float(rng.uniform(-math.pi, math.pi))
            layout[name] = (x, y, float(z), yaw)
            occupied.append((x, y, radius))

        return layout, randomized_objects, occupied

    def _sample_start_and_goal(
        self,
        rng: np.random.Generator,
        occupied: Sequence[tuple[float, float, float]],
        curriculum_level: int,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float],
        str,
        str,
    ]:
        """Sample a safe robot and navigation goal."""

        if curriculum_level not in RANDOMIZED_OBJECT_COUNT:
            raise ValueError(f"Invalid curriculum level: {curriculum_level}")

        minimum_distance, maximum_distance = CURRICULUM_DISTANCE_LIMITS_M[
            curriculum_level
        ]

        room_names = tuple(ROOM_BOUNDS)

        for _ in range(MAX_SAMPLING_ATTEMPTS):
            if curriculum_level in (0, 1):
                start_room = str(rng.choice(room_names))
                goal_room = start_room
            else:
                chosen_rooms = rng.choice(
                    room_names,
                    size=2,
                    replace=False,
                ).tolist()

                start_room = str(chosen_rooms[0])
                goal_room = str(chosen_rooms[1])

            try:
                start_x, start_y = self._sample_clear_point(
                    rng=rng,
                    bounds=ROOM_BOUNDS[start_room],
                    candidate_radius=ROBOT_SAMPLE_RADIUS_M,
                    occupied=occupied,
                )
                occupied_for_goal = [
                    *occupied,
                    (
                        start_x,
                        start_y,
                        ROBOT_SAMPLE_RADIUS_M,
                    ),
                ]

                goal_x, goal_y = self._sample_clear_point(
                    rng=rng,
                    bounds=ROOM_BOUNDS[goal_room],
                    candidate_radius=ROBOT_SAMPLE_RADIUS_M,
                    occupied=occupied_for_goal,
                )
            except RuntimeError:
                continue

            start_goal_distance = math.hypot(
                goal_x - start_x,
                goal_y - start_y,
            )
            if not (minimum_distance <= start_goal_distance <= maximum_distance):
                continue

            start_yaw = float(rng.uniform(-math.pi, math.pi))

            return (
                (start_x, start_y, start_yaw),
                (goal_x, goal_y),
                start_room,
                goal_room,
            )

        raise RuntimeError("Failed to sample a valid start and goal pair.")

    def _sample_connected_episode(
        self,
        rng: np.random.Generator,
        curriculum_level: int,
    ) -> tuple[
        dict[str, tuple[float, float, float, float]],
        tuple[str, ...],
        tuple[float, float, float],
        tuple[float, float],
        str,
        str,
        int,
    ]:
        """Sample object, start, and goal poses with a safe grid path."""

        for layout_attempt in range(
            1,
            MAX_LAYOUT_SAMPLING_ATTEMPTS + 1,
        ):
            try:
                layout, randomized_objects, occupied = (
                    self._sample_object_layout(
                        rng=rng,
                        curriculum_level=curriculum_level,
                    )
                )
                start, goal, start_room, goal_room = (
                    self._sample_start_and_goal(
                        rng=rng,
                        occupied=occupied,
                        curriculum_level=curriculum_level,
                    )
                )
            except RuntimeError:
                continue

            if navigation_path_exists(
                start_xy=(start[0], start[1]),
                goal_xy=goal,
                layout=layout,
            ):
                return (
                    layout,
                    randomized_objects,
                    start,
                    goal,
                    start_room,
                    goal_room,
                    layout_attempt,
                )

        raise RuntimeError(
            "Failed to sample a connected navigation episode after "
            f"{MAX_LAYOUT_SAMPLING_ATTEMPTS} attempts."
        )

    def _active_target_xy(self) -> tuple[float, float]:
        """Return the next doorway waypoint or the final goal."""

        if self.goal_xy is None:
            raise RuntimeError("Goal position is not set.")
        if self._route_waypoint_index < len(self._route_waypoints_xy):
            return self._route_waypoints_xy[self._route_waypoint_index]
        return self.goal_xy

    def _advance_route_waypoints(
        self,
        robot_x: float,
        robot_y: float,
    ) -> int:
        """Consume every doorway waypoint reached by the robot."""

        reached = 0
        while self._route_waypoint_index < len(
            self._route_waypoints_xy
        ):
            waypoint_x, waypoint_y = self._route_waypoints_xy[
                self._route_waypoint_index
            ]
            waypoint_distance = math.hypot(
                waypoint_x - robot_x,
                waypoint_y - robot_y,
            )
            if waypoint_distance > ROUTE_WAYPOINT_REACHED_DISTANCE_M:
                break
            self._route_waypoint_index += 1
            reached += 1
        return reached

    def _update_stuck_state(
        self,
        robot_x: float,
        robot_y: float,
    ) -> bool:
        """Return true after the robot remains in one small area."""

        self._position_history.append((robot_x, robot_y))
        if len(self._position_history) < STUCK_WINDOW_STEPS:
            return False
        maximum_displacement = max(
            math.hypot(robot_x - x, robot_y - y)
            for x, y in self._position_history
        )
        return maximum_displacement < STUCK_TRANSLATION_M

    def _observation_from_messages(
        self,
        scan: LaserScan,
        odom: Odometry,
    ) -> tuple[np.ndarray, dict]:
        """Construct the 77-value observation from ROS messages."""

        if self.goal_xy is None:
            raise RuntimeError("Goal position is not set.")

        sensor_range_min = float(scan.range_min)

        if not math.isfinite(sensor_range_min) or sensor_range_min <= 0.0:
            sensor_range_min = LIDAR_MIN_M

        range_min_m = max(
            sensor_range_min,
            LIDAR_MIN_M,
        )
        lidar_normalised, nearest_obstacle_m = preprocess_scan(
            ranges=scan.ranges,
            range_min_m=range_min_m,
            range_cap_m=LIDAR_CAP_M,
            output_bins=LIDAR_BINS,
        )

        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        robot_x = float(position.x)
        robot_y = float(position.y)
        robot_yaw = yaw_from_quaternion(
            x=float(orientation.x),
            y=float(orientation.y),
            z=float(orientation.z),
            w=float(orientation.w),
        )

        waypoints_reached = self._advance_route_waypoints(
            robot_x=robot_x,
            robot_y=robot_y,
        )
        active_target_x, active_target_y = self._active_target_xy()
        target_distance_m, relative_target_bearing_rad = relative_goal(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
            goal_x=active_target_x,
            goal_y=active_target_y,
        )

        goal_x, goal_y = self.goal_xy
        goal_distance_m, relative_goal_bearing_rad = relative_goal(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
            goal_x=goal_x,
            goal_y=goal_y,
        )
        linear_velocity_mps = float(odom.twist.twist.linear.x)
        angular_velocity_radps = float(odom.twist.twist.angular.z)

        observation = build_observation(
            lidar_normalised=lidar_normalised,
            goal_distance_m=target_distance_m,
            relative_goal_bearing_rad=relative_target_bearing_rad,
            linear_velocity_mps=linear_velocity_mps,
            angular_velocity_radps=angular_velocity_radps,
        )

        stuck = self._update_stuck_state(
            robot_x=robot_x,
            robot_y=robot_y,
        )
        active_target_kind = (
            "doorway"
            if self._route_waypoint_index < len(self._route_waypoints_xy)
            else "goal"
        )

        metrics = {
            "robot_x": robot_x,
            "robot_y": robot_y,
            "robot_yaw": robot_yaw,
            "distance_to_goal_m": goal_distance_m,
            "relative_goal_bearing_rad": relative_goal_bearing_rad,
            "distance_to_target_m": target_distance_m,
            "relative_target_bearing_rad": (
                relative_target_bearing_rad
            ),
            "active_target_x": active_target_x,
            "active_target_y": active_target_y,
            "active_target_kind": active_target_kind,
            "route_waypoint_index": self._route_waypoint_index,
            "route_waypoint_count": len(self._route_waypoints_xy),
            "remaining_route_waypoints": (
                len(self._route_waypoints_xy)
                - self._route_waypoint_index
            ),
            "waypoints_reached_this_step": waypoints_reached,
            "stuck": stuck,
            "nearest_obstacle_m": nearest_obstacle_m,
        }

        return observation, metrics

    def reset(
        self,
        seed: int,
        curriculum_level: int = 0,
    ) -> tuple[np.ndarray, dict]:

        """Create one deterministic randomized episode."""
        if curriculum_level not in RANDOMIZED_OBJECT_COUNT:
            raise ValueError(f"Invalid curriculum level: {curriculum_level}")

        # A failed reset must never leave the previous episode usable.
        self._episode_active = False
        self._curriculum_level = None
        self.goal_xy = None
        self._previous_goal_distance_m = None
        self._route_waypoints_xy = ()
        self._route_waypoint_index = 0
        self._position_history.clear()

        episode_seed = int(seed)
        last_error: Exception | None = None

        for attempt in range(1, MAX_RESET_RETRIES + 1):
            try:
                self.publish_stop()
                time.sleep(0.05)

                rng = np.random.default_rng(episode_seed)

                (
                    layout,
                    randomized_objects,
                    start,
                    goal,
                    start_room,
                    goal_room,
                    layout_sampling_attempt,
                ) = self._sample_connected_episode(
                    rng=rng,
                    curriculum_level=curriculum_level,
                )
                start_x, start_y, start_yaw = start
                goal_x, goal_y = goal

                # Reset every randomizable object.
                for entity_name, pose in layout.items():
                    x, y, z, yaw = pose

                    self._set_entity_pose(
                        entity_name=entity_name,
                        x=x,
                        y=y,
                        z=z,
                        yaw=yaw,
                    )

                # Move the goal after all objects have been positioned.
                self._set_entity_pose(
                    entity_name=GOAL_MARKER_NAME,
                    x=goal_x,
                    y=goal_y,
                    z=GOAL_MARKER_Z_M,
                    yaw=0.0,
                )

                # Save the goal before constructing an observation.
                self.goal_xy = (goal_x, goal_y)
                self._route_waypoints_xy = build_route_waypoints(
                    start_room=start_room,
                    goal_room=goal_room,
                )
                self._route_waypoint_index = 0
                self._position_history.clear()

                # Move the robot last.
                self._set_entity_pose(
                    entity_name=ROBOT_NAME,
                    x=start_x,
                    y=start_y,
                    z=ROBOT_RESET_Z_M,
                    yaw=start_yaw,
                )

                self.publish_stop()
                time.sleep(0.05)

                # Capture counters only after every teleport has finished.
                with self._sensor_condition:
                    previous_scan_count = self._scan_count
                    previous_odom_count = self._odom_count

                fresh_scan, fresh_odom = self._wait_for_fresh_sensors(
                    previous_scan_count=previous_scan_count,
                    previous_odom_count=previous_odom_count,
                    timeout_sec=RESET_SENSOR_TIMEOUT_SEC,
                )

                observed_position = fresh_odom.pose.pose.position
                observed_orientation = fresh_odom.pose.pose.orientation

                observed_x = float(observed_position.x)
                observed_y = float(observed_position.y)

                observed_yaw = yaw_from_quaternion(
                    observed_orientation.x,
                    observed_orientation.y,
                    observed_orientation.z,
                    observed_orientation.w,
                )

                observed_position_error = math.hypot(
                    observed_x - start_x,
                    observed_y - start_y,
                )

                observed_yaw_error = abs(
                    wrap_to_pi(observed_yaw - start_yaw)
                )

                if observed_position_error > RESET_POSITION_TOLERANCE_M:
                    raise RuntimeError(
                        "Observed robot position error "
                        f"{observed_position_error:.3f} m exceeds tolerance "
                        f"{RESET_POSITION_TOLERANCE_M:.3f} m."
                    )

                if observed_yaw_error > RESET_YAW_TOLERANCE_RAD:
                    raise RuntimeError(
                        "Observed robot yaw error "
                        f"{observed_yaw_error:.3f} rad exceeds tolerance "
                        f"{RESET_YAW_TOLERANCE_RAD:.3f} rad."
                    )

                observation, metrics = self._observation_from_messages(
                    scan=fresh_scan,
                    odom=fresh_odom,
                )

                self._episode_seed = episode_seed
                self._step_count = 0
                self._current_layout = dict(layout)
                self._randomized_objects = tuple(randomized_objects)
                self._previous_goal_distance_m = metrics[
                    "distance_to_target_m"
                ]

                self._curriculum_level = curriculum_level
                self._episode_active = True

                info = {
                    "seed": episode_seed,
                    "curriculum_level": curriculum_level,
                    "randomized_object_count": len(
                        randomized_objects
                    ),
                    "randomized_objects": randomized_objects,
                    "start_room": start_room,
                    "goal_room": goal_room,
                    "requested_start": start,
                    "goal": goal,
                    "route_waypoints": self._route_waypoints_xy,
                    "route_waypoint_count": len(
                        self._route_waypoints_xy
                    ),
                    "layout_sampling_attempt": (
                        layout_sampling_attempt
                    ),
                    "object_layout": dict(layout),
                    "reset_attempt": attempt,
                }

                info.update(metrics)

                return observation, info

            except Exception as error:
                last_error = error
                self.get_logger().warning(
                    f"Reset attempt {attempt}/{MAX_RESET_RETRIES} "
                    f"failed for seed {episode_seed}: {error}"
                )

        raise RuntimeError(
            f"Episode reset failed after {MAX_RESET_RETRIES} attempts."
        ) from last_error



    def _publish_velocity_command(
        self,
        linear_velocity_mps: float,
        angular_velocity_radps: float,
    ) -> None:
        """Publish one already validated differential-drive command."""

        if not np.isfinite(
            [linear_velocity_mps, angular_velocity_radps]
        ).all():
            raise ValueError(
                "Velocity command must contain finite values."
            )

        message = Twist()
        message.linear.x = float(linear_velocity_mps)
        message.angular.z = float(angular_velocity_radps)

        self.cmd_vel_publisher.publish(message)

    def _apply_action_and_observe(
        self,
        action: Sequence[float],
    ) -> tuple[np.ndarray, dict]:
        """Execute one policy action and return the resulting fresh state."""

        if not self._episode_active:
            raise RuntimeError(
                "reset() must succeed before applying an action."
            )

        try:
            linear_velocity_mps, angular_velocity_radps = (
                policy_action_to_velocity(action)
            )

            with self._sensor_condition:
                if (
                    self._scan_stamp_ns is None
                    or self._odom_stamp_ns is None
                ):
                    raise RuntimeError(
                        "Sensor timestamps are unavailable."
                    )

                reference_stamp_ns = max(
                    self._scan_stamp_ns,
                    self._odom_stamp_ns,
                )

            interval_ns = int(
                CONTROL_INTERVAL_SEC * NANOSECONDS_PER_SECOND
            )
            target_stamp_ns = reference_stamp_ns + interval_ns

            self._publish_velocity_command(
                linear_velocity_mps=linear_velocity_mps,
                angular_velocity_radps=angular_velocity_radps,
            )

            fresh_scan, fresh_odom = self._wait_for_sensor_time(
                target_stamp_ns=target_stamp_ns,
                timeout_sec=STEP_SENSOR_TIMEOUT_SEC,
            )

            for invalid_attempt in range(
                MAX_INVALID_SCAN_RETRIES + 1
            ):
                try:
                    observation, metrics = (
                        self._observation_from_messages(
                            scan=fresh_scan,
                            odom=fresh_odom,
                        )
                    )
                    break

                except InvalidScanError as error:
                    if invalid_attempt == MAX_INVALID_SCAN_RETRIES:
                        raise

                    self.get_logger().warning(
                        "Rejected invalid LiDAR frame; "
                        "waiting for a fresh frame "
                        f"({invalid_attempt + 1}/"
                        f"{MAX_INVALID_SCAN_RETRIES}): {error}"
                    )

                    with self._sensor_condition:
                        previous_scan_count = self._scan_count
                        previous_odom_count = self._odom_count

                    fresh_scan, fresh_odom = (
                        self._wait_for_fresh_sensors(
                            previous_scan_count=previous_scan_count,
                            previous_odom_count=previous_odom_count,
                            timeout_sec=STEP_SENSOR_TIMEOUT_SEC,
                        )
                    )

        except Exception:
            # Never allow the previous command to continue after an error.
            self.publish_stop()
            self._episode_active = False
            raise

        self._step_count += 1

        executed_action = velocity_to_policy_action(
            linear_velocity_mps=linear_velocity_mps,
            angular_velocity_radps=angular_velocity_radps,
        )

        info = {
            "step_count": self._step_count,
            "executed_action": (
                float(executed_action[0]),
                float(executed_action[1]),
            ),
            "commanded_linear_velocity_mps": (
                linear_velocity_mps
            ),
            "commanded_angular_velocity_radps": (
                angular_velocity_radps
            ),
        }

        info.update(metrics)

        return observation, info
    
    def step(
        self,
        action: Sequence[float],
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one RL transition using the five-value API."""

        if self._previous_goal_distance_m is None:
            raise RuntimeError(
                "Previous goal distance is unavailable; "
                "call reset()."
            )

        if self._curriculum_level is None:
            raise RuntimeError(
                "Curriculum level is unavailable; call reset()."
            )

        previous_target_distance_m = (
            self._previous_goal_distance_m
        )

        observation, info = (
            self._apply_action_and_observe(action)
        )

        current_goal_distance_m = float(
            info["distance_to_goal_m"]
        )

        current_target_distance_m = float(
            info["distance_to_target_m"]
        )

        waypoints_reached = int(
            info["waypoints_reached_this_step"]
        )

        # A target switch changes the distance reference.  Zero the progress
        # term for that one transition and use the explicit waypoint bonus.
        reward_previous_distance_m = (
            current_target_distance_m
            if waypoints_reached > 0
            else previous_target_distance_m
        )

        nearest_obstacle_m = float(
            info["nearest_obstacle_m"]
        )

        max_episode_steps = MAX_EPISODE_STEPS[
            self._curriculum_level
        ]

        terminated, truncated, outcome = (
            determine_episode_outcome(
                goal_distance_m=current_goal_distance_m,
                nearest_obstacle_m=nearest_obstacle_m,
                step_count=self._step_count,
                max_episode_steps=max_episode_steps,
            )
        )

        reward, reward_components = calculate_reward(
            previous_goal_distance_m=reward_previous_distance_m,
            current_goal_distance_m=current_target_distance_m,
            relative_goal_bearing_rad=float(
                info["relative_target_bearing_rad"]
            ),
            nearest_obstacle_m=nearest_obstacle_m,
            commanded_linear_velocity_mps=float(
                info["commanded_linear_velocity_mps"]
            ),
            outcome=outcome,
            waypoints_reached=waypoints_reached,
            stuck=bool(info["stuck"]),
        )

        self._previous_goal_distance_m = (
            current_target_distance_m
        )

        if terminated or truncated:
            self.publish_stop()
            self._episode_active = False

        info.update(
            {
                "outcome": outcome,
                "reached_goal": outcome == "success",
                "collision": outcome == "collision",
                "time_limit_reached": outcome == "timeout",
                "max_episode_steps": max_episode_steps,
                "reward_components": reward_components,
            }
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    
    def publish_stop(self) -> None:
        """Immediately request zero robot velocity."""

        self.cmd_vel_publisher.publish(Twist())

    def close(self) -> None:
        """Stop the robot and release ROS resources safely."""

        if self._closed:
            return

        # Mark first so repeated close calls do nothing.
        self._closed = True
        self._episode_active = False

        self.publish_stop()
        time.sleep(0.05)

        self._executor.shutdown(timeout_sec=1.0)
        self._executor_thread.join(timeout=1.0)

        self.destroy_node()


def main(args=None) -> None:
    """Run the complete step-contract smoke test."""

    outcome_cases = (
        (1.0, 1.0, 1, 300, (False, False, "running")),
        (0.20, 1.0, 1, 300, (True, False, "success")),
        (1.0, 0.20, 1, 300, (True, False, "collision")),
        (1.0, 1.0, 300, 300, (False, True, "timeout")),
    )

    for (
        goal_distance_m,
        nearest_obstacle_m,
        step_count,
        max_steps,
        expected,
    ) in outcome_cases:
        actual = determine_episode_outcome(
            goal_distance_m=goal_distance_m,
            nearest_obstacle_m=nearest_obstacle_m,
            step_count=step_count,
            max_episode_steps=max_steps,
        )

        if actual != expected:
            raise RuntimeError(
                f"Outcome test failed: expected {expected}, "
                f"got {actual}."
            )

    rclpy.init(args=args)
    environment = None

    try:
        environment = RosNavEnv()

        environment.reset(
            seed=19,
            curriculum_level=0,
        )

        (
            observation,
            running_reward,
            terminated,
            truncated,
            running_info,
        ) = environment.step(
            action=(-1.0, 0.0)
        )

        if terminated or truncated:
            raise RuntimeError(
                "Normal transition ended unexpectedly."
            )

        if running_info["outcome"] != "running":
            raise RuntimeError(
                "Expected a running transition."
            )

        # Place the visual and logical goal safely 0.10 m away.
        test_goal_x = running_info["robot_x"] + 0.10
        test_goal_y = running_info["robot_y"]

        environment._set_entity_pose(
            entity_name=GOAL_MARKER_NAME,
            x=test_goal_x,
            y=test_goal_y,
            z=GOAL_MARKER_Z_M,
            yaw=0.0,
        )

        environment.goal_xy = (
            test_goal_x,
            test_goal_y,
        )

        environment._previous_goal_distance_m = 0.10

        (
            observation,
            success_reward,
            terminated,
            truncated,
            success_info,
        ) = environment.step(
            action=(-1.0, 0.0)
        )

        if not terminated or truncated:
            raise RuntimeError(
                "Success did not produce the expected flags."
            )

        if success_info["outcome"] != "success":
            raise RuntimeError(
                "Expected a success outcome."
            )

        if success_reward <= running_reward:
            raise RuntimeError(
                "Success reward must exceed running reward."
            )

        if observation.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(
                "Unexpected step observation shape: "
                f"{observation.shape}"
            )

        environment.get_logger().info(
            "Step contract passed: "
            f"running_reward={running_reward:.3f}, "
            f"success_reward={success_reward:.3f}, "
            f"outcome={success_info['outcome']}, "
            f"terminated={terminated}, "
            f"truncated={truncated}, "
            f"observation_shape={observation.shape}"
        )

    finally:
        if environment is not None:
            environment.close()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
