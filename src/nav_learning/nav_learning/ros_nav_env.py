"""ROS 2 final-goal PPO environment with staged random navigation tasks."""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Sequence
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from nav_learning.nav_math import (
    BOX_NAMES, ENVIRONMENT_ID, WORLD_NAME, STAGES, RewardSettings,
    goal_region_reached, parked_box_pose, sample_episode,
)
from nav_learning.nav_math import (
    COLLISION_DISTANCE_M, CONTROL_INTERVAL_SEC, GOAL_REACHED_DISTANCE_M,
    LIDAR_BINS, LIDAR_CAP_M, LIDAR_MIN_M, MAX_EPISODE_STEPS,
    MAX_LINEAR_SPEED_MPS, MAX_ANGULAR_SPEED_RADPS,
    OBSERVATION_SIZE, InvalidScanError, build_observation, calculate_reward,
    determine_episode_outcome, policy_action_to_velocity, preprocess_scan,
    quaternion_from_yaw, relative_goal, velocity_to_policy_action, wrap_to_pi,
    yaw_from_quaternion,
    ActionSettings, DetourDistance,
)
SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/cmd_vel"
SET_POSE_SERVICE = f"/world/{WORLD_NAME}/set_pose"
CONTROL_WORLD_SERVICE = f"/world/{WORLD_NAME}/control"
CONTROL_MODE = "gazebo_run_to_sim_time_v1"
MAX_SENSOR_AGE_SEC = 0.05  # The existing scan and odometry publishers run at 30 Hz.
CLOCK_TOLERANCE_NS = 2_000_000
STARTUP_TIMEOUT_SEC = 8.0
FRESH_SENSOR_TIMEOUT_SEC = 2.0
STEP_SENSOR_TIMEOUT_SEC = 5.0
NANOSECONDS_PER_SECOND = 1_000_000_000
ROBOT_NAME = "hv_bot"
GOAL_MARKER_NAME = "goal_marker"
ROBOT_RESET_Z_M = 0.20
GOAL_MARKER_Z_M = 0.012
SET_POSE_TIMEOUT_SEC = 2.0
RESET_SENSOR_TIMEOUT_SEC = 2.0
MAX_RESET_RETRIES = 3
RESET_POSITION_TOLERANCE_M = 0.20
RESET_YAW_TOLERANCE_RAD = 0.30
RANDOMIZED_OBJECT_COUNT = {level: stage.box_count_limits[1] for level, stage in STAGES.items()}
STUCK_WINDOW_STEPS = 80
STUCK_TRANSLATION_M = 0.15

class RosNavEnv(Node):
    """Synchronous learning environment around asynchronous ROS callbacks."""

    def __init__(self, reward_settings: RewardSettings | None = None,
                 action_settings: ActionSettings | None = None) -> None:
        super().__init__("ros_nav_env")
        self._reward_settings = reward_settings or RewardSettings()
        self._action_settings = action_settings or ActionSettings()
        self._detour_distance = None
        self._previous_detour_distance_m = None

        # Condition protects the sensor messages and counters.
        self._sensor_condition = threading.Condition()

        self._latest_scan: LaserScan | None = None
        self._latest_odom: Odometry | None = None

        self._scan_count = 0
        self._odom_count = 0

        self._scan_stamp_ns: int | None = None
        self._odom_stamp_ns: int | None = None
        self._clock_stamp_ns: int | None = None
        self._paused_stamp_ns: int | None = None
        self._last_step_timing: dict[str, float] = {}

        self._closed = False
        self.goal_xy: tuple[float, float] | None = None
        self._previous_goal_distance_m: float | None = None
        self._previous_nearest_obstacle_m: float | None = None
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
        self.clock_subscription = self.create_subscription(
            Clock, "/clock", self._clock_callback, qos_profile_sensor_data,
        )

        self.set_pose_client = self.create_client(
            SetEntityPose,
            SET_POSE_SERVICE,
        )
        self.control_world_client = self.create_client(ControlWorld, CONTROL_WORLD_SERVICE)

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

    def _clock_callback(self, message: Clock) -> None:
        with self._sensor_condition:
            self._clock_stamp_ns = self._stamp_to_nanoseconds(message.clock)
            self._sensor_condition.notify_all()

    @property
    def simulation_time_ns(self) -> int:
        with self._sensor_condition:
            if self._clock_stamp_ns is None:
                raise RuntimeError("No /clock message. Check the Gazebo clock bridge.")
            return self._clock_stamp_ns

    def _control_world(self, *, pause: bool, run_to_ns: int | None = None) -> None:
        """Request native Gazebo control; the response acknowledges acceptance."""
        request = ControlWorld.Request()
        request.world_control.pause = pause
        if run_to_ns is not None:
            request.world_control.run_to_sim_time.sec = run_to_ns // NANOSECONDS_PER_SECOND
            request.world_control.run_to_sim_time.nanosec = run_to_ns % NANOSECONDS_PER_SECOND
        future = self.control_world_client.call_async(request)
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        if not ready.wait(STEP_SENSOR_TIMEOUT_SEC):
            future.cancel()
            raise TimeoutError(f"Gazebo control request timed out: {CONTROL_WORLD_SERVICE}")
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("Gazebo rejected the world-control request.")

    def pause_simulation(self) -> None:
        """Pause physics and drain the last in-flight /clock publications."""
        self._control_world(pause=True)
        deadline = time.monotonic() + STEP_SENSOR_TIMEOUT_SEC
        previous = None
        while time.monotonic() < deadline:
            time.sleep(0.02)
            current = self.simulation_time_ns
            if current == previous:
                self._paused_stamp_ns = current
                return
            previous = current
        raise TimeoutError("Gazebo did not stop advancing after pause.")

    def assert_simulation_time_unchanged(self, expected_ns: int) -> None:
        """Catch accidental Play clicks or a broken pause/step bridge."""
        if abs(self.simulation_time_ns - expected_ns) > CLOCK_TOLERANCE_NS:
            raise RuntimeError(
                "Simulation advanced outside an environment step. Leave Gazebo paused "
                "while training; run --timing-check before restarting."
            )

    def _advance_simulation(self, duration_sec: float) -> tuple[LaserScan, Odometry]:
        """Advance to an absolute sim timestamp, where Gazebo pauses itself.

        run_to_sim_time is enforced by Gazebo, not a wall-clock sleep. Sensor
        delivery may lag; timestamps must be recent and belong to this step.
        """
        if not math.isfinite(duration_sec) or duration_sec <= MAX_SENSOR_AGE_SEC:
            raise ValueError("Step duration must exceed the sensor age allowance.")
        if self._paused_stamp_ns is None:
            raise RuntimeError("Simulation has not been paused and synchronized.")
        self.assert_simulation_time_unchanged(self._paused_stamp_ns)
        start_ns = self._paused_stamp_ns
        target_ns = start_ns + int(round(duration_sec * NANOSECONDS_PER_SECOND))
        self._control_world(pause=False, run_to_ns=target_ns)
        with self._sensor_condition:
            arrived = self._sensor_condition.wait_for(
                lambda: self._clock_stamp_ns is not None and self._clock_stamp_ns >= target_ns,
                timeout=STEP_SENSOR_TIMEOUT_SEC,
            )
        if not arrived:
            raise TimeoutError("Gazebo failed to reach the requested simulation time.")
        scan, odom = self._wait_for_sensor_time(
            target_ns - int(MAX_SENSOR_AGE_SEC * NANOSECONDS_PER_SECOND),
            STEP_SENSOR_TIMEOUT_SEC,
        )
        end_ns = self.simulation_time_ns
        if abs(end_ns - target_ns) > CLOCK_TOLERANCE_NS:
            raise RuntimeError("Gazebo overshot the requested step; check world-control support.")
        sensor_ns = [self._stamp_to_nanoseconds(m.header.stamp) for m in (scan, odom)]
        if min(sensor_ns) <= start_ns or max(sensor_ns) > end_ns + CLOCK_TOLERANCE_NS:
            raise RuntimeError("Sensor timestamps do not belong to the completed action step.")
        self._paused_stamp_ns = end_ns
        self._last_step_timing = {
            "simulation_interval_sec": (end_ns - start_ns) / NANOSECONDS_PER_SECOND,
            "maximum_sensor_age_sec": max(0.0, (end_ns - min(sensor_ns)) / NANOSECONDS_PER_SECOND),
        }
        return scan, odom

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

        if not self.control_world_client.wait_for_service(timeout_sec=STARTUP_TIMEOUT_SEC):
            raise TimeoutError(
                f"Missing {CONTROL_WORLD_SERVICE}. Rebuild description_new and restart its launch."
            )
        if not self.set_pose_client.wait_for_service(
            timeout_sec=STARTUP_TIMEOUT_SEC
        ):
            raise TimeoutError(
                f"Timeout waiting for set-pose service {SET_POSE_SERVICE}."
            )

        # A previous trainer intentionally leaves physics paused. Bootstrap only
        # while no episode is active, then retain ownership of simulation time.
        self.publish_stop()
        self._control_world(pause=False)
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
        with self._sensor_condition:
            if not self._sensor_condition.wait_for(
                lambda: self._clock_stamp_ns is not None, timeout=STARTUP_TIMEOUT_SEC,
            ):
                raise TimeoutError("Missing /clock. Check the clock bridge.")
        self.pause_simulation()

        self.get_logger().info(
            "Navigation environment startup check passed: set-pose service, "
            "world control, /clock, LiDAR, and odometry are ready; physics is paused."
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
            goal_distance_m=goal_distance_m,
            relative_goal_bearing_rad=relative_goal_bearing_rad,
            linear_velocity_mps=linear_velocity_mps,
            angular_velocity_radps=angular_velocity_radps,
        )

        stuck = self._update_stuck_state(
            robot_x=robot_x,
            robot_y=robot_y,
        )
        metrics = {
            "robot_x": robot_x,
            "robot_y": robot_y,
            "robot_yaw": robot_yaw,
            "distance_to_goal_m": goal_distance_m,
            "relative_goal_bearing_rad": relative_goal_bearing_rad,
            "goal_region_reached": goal_region_reached((robot_x, robot_y), self.goal_xy),
            "stuck": stuck,
            "nearest_obstacle_m": nearest_obstacle_m,
        }

        return observation, metrics

    def reset(
        self,
        seed: int,
        curriculum_level: int = 0,
        practice_phase: int | None = None,
        corner_probability: float = 0.0,
        mirror_pair: bool = False,
    ) -> tuple[np.ndarray, dict]:

        """Create one deterministic randomized episode."""
        if curriculum_level not in RANDOMIZED_OBJECT_COUNT:
            raise ValueError(f"Invalid curriculum level: {curriculum_level}")

        # A failed reset must never leave the previous episode usable.
        self._episode_active = False
        self._curriculum_level = None
        self.goal_xy = None
        self._previous_goal_distance_m = None
        self._previous_nearest_obstacle_m = None
        self._detour_distance = None
        self._previous_detour_distance_m = None
        self._position_history.clear()

        episode_seed = int(seed)
        last_error: Exception | None = None

        for attempt in range(1, MAX_RESET_RETRIES + 1):
            try:
                self.pause_simulation()
                self.publish_stop()

                episode = sample_episode(episode_seed, curriculum_level, practice_phase,
                                         corner_probability=corner_probability, mirror_pair=mirror_pair)
                layout, start, goal = episode.layout, episode.start, episode.goal
                randomized_objects = tuple(layout)
                # Stage the robot outside the arena so moving boxes cannot land on it.
                self._set_entity_pose(ROBOT_NAME, 0.0, -6.0, ROBOT_RESET_Z_M, 0.0)
                start_x, start_y, start_yaw = start
                goal_x, goal_y = goal

                # Reset every randomizable object.
                for box_index, entity_name in enumerate(BOX_NAMES):
                    pose = layout.get(entity_name, parked_box_pose(box_index))
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
                # Let gravity and zero-velocity commands settle the teleported
                # robot, then pause before forming the first observation.
                fresh_scan, fresh_odom = self._advance_simulation(0.40)

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

                if metrics["nearest_obstacle_m"] <= COLLISION_DISTANCE_M:
                    raise RuntimeError("Reset returned a robot already inside the collision margin.")
                if abs(fresh_odom.twist.twist.linear.x) > 0.03 or abs(fresh_odom.twist.twist.angular.z) > 0.08:
                    raise RuntimeError("Robot velocity did not settle after reset.")

                self._episode_seed = episode_seed
                self._step_count = 0
                self._current_layout = dict(layout)
                self._randomized_objects = tuple(randomized_objects)
                self._previous_goal_distance_m = metrics[
                    "distance_to_goal_m"
                ]
                self._previous_nearest_obstacle_m = metrics["nearest_obstacle_m"]
                if self._reward_settings.progress_mode == "detour_potential":
                    # This helper is used ONLY to calculate rewards. Observation
                    # construction and the action command never read its output.
                    self._detour_distance = DetourDistance(layout, goal,
                        self._reward_settings.detour_clearance_m)
                    self._previous_detour_distance_m = self._detour_distance.distance(
                        (metrics["robot_x"], metrics["robot_y"]))

                self._curriculum_level = curriculum_level
                self._episode_active = True

                info = {
                    "seed": episode_seed,
                    "curriculum_level": curriculum_level,
                    "randomized_object_count": len(
                        randomized_objects
                    ),
                    "randomized_objects": randomized_objects,
                    "environment_id": ENVIRONMENT_ID,
                    "requested_start": start,
                    "goal": goal,
                    "straight_path_blocked": episode.straight_path_blocked,
                    "straight_path_blocker_count": episode.straight_path_blocker_count,
                    "layout_sampling_attempt": episode.layout_sampling_attempt,
                    "object_layout": dict(layout),
                    "region_center": episode.region_center,
                    "region_size_m": episode.region_size_m,
                    "reset_attempt": attempt,
                    "practice_phase": episode.practice_phase,
                    "control_mode": CONTROL_MODE,
                    "task_kind": episode.task_kind,
                    "mirrored": episode.mirrored,
                    "detour_distance_m": self._previous_detour_distance_m,
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
                policy_action_to_velocity(action, getattr(self, "_action_settings", ActionSettings()))
            )

            self._publish_velocity_command(
                linear_velocity_mps=linear_velocity_mps,
                angular_velocity_radps=angular_velocity_radps,
            )
            # Allow the ROS/Gazebo command bridge to drain while physics is
            # paused. This wall wait cannot move the robot.
            time.sleep(0.002)
            fresh_scan, fresh_odom = self._advance_simulation(CONTROL_INTERVAL_SEC)
            observation, metrics = self._observation_from_messages(fresh_scan, fresh_odom)

        except Exception:
            # Never allow the previous command to continue after an error.
            self.publish_stop()
            self._episode_active = False
            try:
                self.pause_simulation()
            except Exception:
                pass  # Preserve the original transport/sensor failure.
            raise

        self._step_count += 1

        executed_action = velocity_to_policy_action(
            linear_velocity_mps=linear_velocity_mps,
            angular_velocity_radps=angular_velocity_radps,
            action_settings=getattr(self, "_action_settings", ActionSettings()),
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
        info.update(self._last_step_timing)

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

        previous_goal_distance_m = self._previous_goal_distance_m
        observation, info = self._apply_action_and_observe(action)
        current_goal_distance_m = float(info["distance_to_goal_m"])

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
                goal_reached=bool(info["goal_region_reached"]),
            )
        )

        current_detour = None
        if getattr(self, "_detour_distance", None) is not None:
            current_detour = self._detour_distance.distance((info["robot_x"], info["robot_y"]))
        reward, reward_components = calculate_reward(
            previous_goal_distance_m=previous_goal_distance_m,
            current_goal_distance_m=current_goal_distance_m,
            relative_goal_bearing_rad=float(
                info["relative_goal_bearing_rad"]
            ),
            nearest_obstacle_m=nearest_obstacle_m,
            commanded_linear_velocity_mps=float(
                info["commanded_linear_velocity_mps"]
            ),
            outcome=outcome,
            stuck=bool(info["stuck"]),
            reward_settings=getattr(self, "_reward_settings", RewardSettings()),
            previous_detour_distance_m=getattr(self, "_previous_detour_distance_m", None),
            current_detour_distance_m=current_detour,
            previous_nearest_obstacle_m=getattr(self, "_previous_nearest_obstacle_m", None),
        )

        self._previous_goal_distance_m = (
            current_goal_distance_m
        )
        self._previous_detour_distance_m = current_detour
        self._previous_nearest_obstacle_m = nearest_obstacle_m

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
                "detour_distance_m": current_detour,
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
        try:
            self.pause_simulation()
        except Exception:
            pass  # The ROS context or Gazebo may already have stopped.

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
