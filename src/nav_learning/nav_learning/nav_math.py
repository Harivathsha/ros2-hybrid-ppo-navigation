"""Navigation math: curriculum layouts, observations, actions and rewards.

This file has no ROS or PyTorch dependency. Start reading with STAGES,
then sample_episode(), build_observation(), and calculate_reward().
"""
from __future__ import annotations

import math
import heapq
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace

import numpy as np

ENVIRONMENT_ID = 'square_arena_random_3_6_9m_curriculum_circle_goal_v3'
WORLD_NAME = 'rl_square_arena'
ARENA_SIZE_M = 9.0
ARENA_HALF_M = ARENA_SIZE_M / 2.0
WALL_THICKNESS_M = 0.25
WALL_HEIGHT_M = 0.90
BOX_SIZE_M = 1.0
BOX_HEIGHT_M = 0.80
BOX_HALF_M = BOX_SIZE_M / 2.0
BOX_COUNT = 15
BOX_NAMES = tuple(f'arena_box_{i:02d}' for i in range(BOX_COUNT))
MIN_BOX_GAP_M = 0.65
MIN_BOX_WALL_GAP_M = 0.30
GOAL_RADIUS_M = 0.25
GOAL_SIZE_M = 2.0 * GOAL_RADIUS_M
ROBOT_FOOTPRINT_RADIUS_M = 0.22
GOAL_SUCCESS_RADIUS_M = GOAL_RADIUS_M
START_CLEARANCE_M = 0.50
GOAL_CLEARANCE_M = 0.50
GOAL_EDGE_CLEARANCE_M = 0.10
PATH_GRID_RESOLUTION_M = 0.10
PATH_CLEARANCE_M = ROBOT_FOOTPRINT_RADIUS_M + 0.03
Pose = tuple[float, float, float, float]


@dataclass(frozen=True)
class StageDefinition:
    size_m: float
    distance_limits_m: tuple[float, float]
    box_count_limits: tuple[int, int]
    max_steps: int
    blocked_path_probability: float


STAGES = {
    0: StageDefinition(3.0, (0.8, 2.2), (0, 2), 450, 0.0),
    1: StageDefinition(6.0, (2.0, 5.0), (5, 7), 900, 0.5),
    2: StageDefinition(9.0, (3.0, 10.0), (12, 15), 1600, 0.8),
}
CURRICULUM_DISTANCE_LIMITS_M = {level: stage.distance_limits_m for level, stage in STAGES.items()}
REHEARSAL_PROBABILITY = 0.20
DETOUR_PHASE_NAMES = ("one_box", "two_boxes", "three_boxes", "full_stage2")
DETOUR_PHASE_COUNT = 4


def detour_stage_definition(phase: int) -> StageDefinition:
    """Training-only practice in the same 6 m patch; benchmarks stay unchanged."""
    if isinstance(phase, bool) or phase not in (0, 1, 2):
        raise ValueError("Detour practice phase must be 0, 1 or 2.")
    return StageDefinition(6.0, ((2.0, 3.2), (2.0, 4.0), (2.0, 5.0))[phase],
                           (phase + 1, phase + 1), 900, 1.0)


def parked_box_pose(index: int) -> Pose:
    """Unused static boxes stay outside the walls and away from robot staging."""
    return (-7.0 + index, -8.0, BOX_HEIGHT_M / 2.0, 0.0)


@dataclass(frozen=True)
class ArenaEpisode:
    layout: dict[str, Pose]
    start: tuple[float, float, float]
    goal: tuple[float, float]
    straight_path_blocked: bool
    layout_sampling_attempt: int
    straight_path_blocker_count: int = 0
    curriculum_level: int = 2
    region_center: tuple[float, float] = (0.0, 0.0)
    region_size_m: float = ARENA_SIZE_M
    practice_phase: int = -1
    task_kind: str = "random"
    mirrored: bool = False


def _check_layout(layout: Mapping[str, Pose]) -> None:
    for name, pose in layout.items():
        if name not in BOX_NAMES:
            raise ValueError(f'Unknown arena box: {name}')
        if len(pose) != 4 or not np.isfinite(pose).all():
            raise ValueError('Box poses must contain finite x, y, z, yaw.')
        if abs(pose[3]) > 1e-9:
            raise ValueError('This geometry contract uses axis-aligned boxes.')


def point_is_clear(point: tuple[float, float], layout: Mapping[str, Pose],
                   clearance_m: float = START_CLEARANCE_M) -> bool:
    """Conservative square clearance around a point, including arena walls."""
    if not np.isfinite(point).all() or not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError('Point and clearance must be finite; clearance nonnegative.')
    x, y = point
    if max(abs(x), abs(y)) > ARENA_HALF_M - clearance_m:
        return False
    return all(max(abs(x - p[0]), abs(y - p[1])) > BOX_HALF_M + clearance_m
               for p in layout.values())


def goal_region_reached(robot_xy: tuple[float, float], goal_xy: tuple[float, float]) -> bool:
    """Success means the robot CENTRE reaches the small circular marker."""
    if not np.isfinite((*robot_xy, *goal_xy)).all():
        raise ValueError('Goal and robot coordinates must be finite.')
    return math.dist(robot_xy, goal_xy) <= GOAL_SUCCESS_RADIUS_M


def straight_path_blocker_count(start_xy: tuple[float, float], goal_xy: tuple[float, float],
                                layout: Mapping[str, Pose],
                                clearance_m: float = PATH_CLEARANCE_M) -> int:
    """Count boxes intersecting the swept straight-line footprint; reset metadata only."""
    _check_layout(layout)
    if not np.isfinite((*start_xy, *goal_xy, clearance_m)).all() or clearance_m < 0:
        raise ValueError('Segment coordinates and nonnegative clearance must be finite.')
    start = np.asarray(start_xy, dtype=float)
    delta = np.asarray(goal_xy, dtype=float) - start
    half = BOX_HALF_M + clearance_m
    count = 0
    for pose in layout.values():
        centre = np.asarray(pose[:2])
        low, high = centre - half, centre + half
        enter, leave = 0.0, 1.0
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-12:
                if not low[axis] <= start[axis] <= high[axis]:
                    enter, leave = 1.0, 0.0
                    break
            else:
                a = (low[axis] - start[axis]) / delta[axis]
                b = (high[axis] - start[axis]) / delta[axis]
                enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
        if enter <= leave:
            count += 1
    return count


def straight_path_is_clear(start_xy: tuple[float, float], goal_xy: tuple[float, float],
                           layout: Mapping[str, Pose],
                           clearance_m: float = PATH_CLEARANCE_M) -> bool:
    if not point_is_clear(start_xy, layout, clearance_m) or not point_is_clear(goal_xy, layout, clearance_m):
        return False
    return straight_path_blocker_count(start_xy, goal_xy, layout, clearance_m) == 0


def navigation_path_exists(start_xy: tuple[float, float], goal_xy: tuple[float, float],
                           layout: Mapping[str, Pose],
                           resolution_m: float = PATH_GRID_RESOLUTION_M,
                           region_center: tuple[float, float] = (0.0, 0.0),
                           region_size_m: float = ARENA_SIZE_M) -> bool:
    """Boolean reachability only: no route, waypoint or action is returned.

    Inflate by the robot margin AND half a grid-cell diagonal. This makes the
    four-connected cell-centre test conservative between neighbouring cells.
    """
    _check_layout(layout)
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError('resolution_m must be positive and finite.')
    if not np.isfinite((*region_center, region_size_m)).all() or not 0 < region_size_m <= ARENA_SIZE_M:
        raise ValueError('Region must be finite and fit inside the arena.')
    if max(abs(v) for v in region_center) + region_size_m / 2 > ARENA_HALF_M + 1e-9:
        raise ValueError('Region extends beyond the arena.')
    if not point_is_clear(start_xy, layout, PATH_CLEARANCE_M) or not point_is_clear(goal_xy, layout, PATH_CLEARANCE_M):
        return False
    clearance = PATH_CLEARANCE_M + resolution_m * math.sqrt(2.0) / 2.0
    limit = region_size_m / 2 - clearance
    if limit <= 0:
        return False
    coordinates = np.arange(-limit, limit + 1e-9, resolution_m)
    xx, yy = np.meshgrid(coordinates, coordinates)
    xx += region_center[0]
    yy += region_center[1]
    blocked = np.zeros(xx.shape, dtype=bool)
    for pose in layout.values():
        blocked |= (np.abs(xx - pose[0]) <= BOX_HALF_M + clearance) & (np.abs(yy - pose[1]) <= BOX_HALF_M + clearance)

    def cell(point):
        column, row = (int(round((v - origin + limit) / resolution_m))
                       for v, origin in zip(point, region_center))
        if 0 <= row < len(coordinates) and 0 <= column < len(coordinates):
            return row, column
        return None

    source, target = cell(start_xy), cell(goal_xy)
    if source is None or target is None or blocked[source] or blocked[target]:
        return False
    if source == target:
        return True
    pending = deque([source])
    visited = np.zeros(blocked.shape, dtype=bool)
    visited[source] = True
    while pending:
        row, col = pending.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < len(coordinates) and 0 <= nc < len(coordinates)):
                continue
            if blocked[nr, nc] or visited[nr, nc]:
                continue
            if (nr, nc) == target:
                return True
            visited[nr, nc] = True
            pending.append((nr, nc))
    return False


def sample_box_layout(rng: np.random.Generator, count: int = BOX_COUNT,
                      region_center: tuple[float, float] = (0.0, 0.0),
                      region_size_m: float = ARENA_SIZE_M) -> dict[str, Pose]:
    """Continuously sample boxes; no fixed obstacle grid or fixed corridor."""
    if not 0 <= count <= BOX_COUNT:
        raise ValueError('Invalid active box count.')
    # The patch edge is a sampling boundary, not an additional physical wall.
    limit = region_size_m / 2 - BOX_HALF_M - 0.15
    for _ in range(40):
        layout = {}
        for name in BOX_NAMES[:count]:
            for _ in range(600):
                x, y = (float(v) for v in rng.uniform(-limit, limit, 2) + region_center)
                if max(abs(x), abs(y)) > ARENA_HALF_M - BOX_HALF_M - MIN_BOX_WALL_GAP_M:
                    continue
                if all(max(abs(x - p[0]), abs(y - p[1])) >= BOX_SIZE_M + MIN_BOX_GAP_M
                       for p in layout.values()):
                    layout[name] = (x, y, BOX_HEIGHT_M / 2.0, 0.0)
                    break
            else:
                break
        if len(layout) == count:
            return layout
    raise RuntimeError('Could not pack the requested boxes with safe gaps.')


def _sample_random_episode(seed: int, curriculum_level: int = 0,
                   practice_phase: int | None = None) -> ArenaEpisode:
    """Random start, goal and obstacles; the same seed reproduces the task.

    Reachability is checked within the spawning patch. The policy receives
    no planned path, waypoint, or action from this reset-time check.
    """
    if curriculum_level not in STAGES:
        raise ValueError('Curriculum level must be 0, 1 or 2.')
    rng = np.random.default_rng(int(seed))
    if practice_phase is not None and curriculum_level != 1:
        raise ValueError("Detour practice is defined only for Stage 2 (level 1).")
    stage = (STAGES[curriculum_level] if practice_phase is None
             else detour_stage_definition(practice_phase))
    for attempt in range(1, 101):
        centre_limit = ARENA_HALF_M - stage.size_m / 2
        centre = tuple(float(v) for v in rng.uniform(-centre_limit, centre_limit, 2))
        if curriculum_level == 0:
            count = int(rng.choice((0, 1, 2), p=(0.25, 0.50, 0.25)))
        else:
            count = int(rng.integers(stage.box_count_limits[0], stage.box_count_limits[1] + 1))
        try:
            layout = sample_box_layout(rng, count, centre, stage.size_m)
        except RuntimeError:
            continue
        require_blocker = rng.random() < stage.blocked_path_probability
        limit = stage.size_m / 2 - 0.35
        for _ in range(500):
            start_xy = tuple(float(v) for v in rng.uniform(-limit, limit, 2) + centre)
            goal_xy = tuple(float(v) for v in rng.uniform(-limit, limit, 2) + centre)
            distance = math.dist(start_xy, goal_xy)
            if not stage.distance_limits_m[0] <= distance <= stage.distance_limits_m[1]:
                continue
            if not point_is_clear(start_xy, layout, START_CLEARANCE_M) or not point_is_clear(goal_xy, layout, GOAL_CLEARANCE_M):
                continue
            blockers = straight_path_blocker_count(start_xy, goal_xy, layout)
            if require_blocker and blockers == 0:
                continue
            if not navigation_path_exists(start_xy, goal_xy, layout,
                                          region_center=centre, region_size_m=stage.size_m):
                continue
            # Initially most headings are within 60 degrees of the goal;
            # include fully random headings too. Later stages use all headings.
            bearing = math.atan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
            yaw = (bearing + rng.uniform(-math.pi / 3, math.pi / 3)
                   if ((curriculum_level == 0 and rng.random() < 0.75)
                       or practice_phase == 0)
                   else rng.uniform(-math.pi, math.pi))
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            return ArenaEpisode(layout, (*start_xy, float(yaw)), goal_xy,
                                blockers > 0, attempt, blockers,
                                curriculum_level, centre, stage.size_m,
                                -1 if practice_phase is None else practice_phase)
    raise RuntimeError(f'No connected stage {curriculum_level + 1} task for seed {seed}.')


def _sample_corner_episode(seed: int, practice_phase: int | None) -> ArenaEpisode:
    """Approach a box face, then go around its nearer corner to the goal.

    No steering command is supplied. Rotate the whole arrangement by a random
    quarter-turn; reflection is handled separately in sample_episode().
    """
    rng = np.random.default_rng(seed)
    stage = STAGES[1] if practice_phase is None else detour_stage_definition(practice_phase)
    for attempt in range(1, 201):
        centre = tuple(float(v) for v in rng.uniform(-1.5, 1.5, 2))
        count = int(rng.integers(stage.box_count_limits[0], stage.box_count_limits[1] + 1))
        try:
            layout = sample_box_layout(rng, count, centre, stage.size_m)
        except RuntimeError:
            continue
        box = layout[BOX_NAMES[0]]
        angle = int(rng.integers(4)) * math.pi / 2
        cosine, sine = math.cos(angle), math.sin(angle)

        def around_box(x, y):
            return (box[0] + cosine * x - sine * y,
                    box[1] + sine * x + cosine * y)

        start = around_box(-float(rng.uniform(1.05, 1.30)), float(rng.uniform(-0.2, 0.2)))
        goal = around_box(float(rng.uniform(0.8, 1.3)), float(rng.uniform(1.05, 1.40)))
        if not stage.distance_limits_m[0] <= math.dist(start, goal) <= stage.distance_limits_m[1]:
            continue
        if any(max(abs(p[i] - centre[i]) for i in (0, 1)) > stage.size_m / 2 - 0.35
               for p in (start, goal)):
            continue
        if not point_is_clear(start, layout) or not point_is_clear(goal, layout):
            continue
        blockers = straight_path_blocker_count(start, goal, layout)
        if not blockers or not navigation_path_exists(start, goal, layout,
                region_center=centre, region_size_m=stage.size_m):
            continue
        yaw = wrap_to_pi(angle + float(rng.uniform(-0.35, 0.35)))
        return ArenaEpisode(layout, (*start, yaw), goal, True, attempt, blockers,
                            1, centre, stage.size_m,
                            -1 if practice_phase is None else practice_phase, "corner")
    raise RuntimeError(f"Could not create a reachable corner task for seed {seed}.")


def sample_episode(seed: int, curriculum_level: int = 0,
                   practice_phase: int | None = None, *,
                   corner_probability: float = 0.0,
                   mirror_pair: bool = False) -> ArenaEpisode:
    """Defaults keep the original benchmark seeds exactly unchanged.

    Training can use pairs: seeds 10000/10001 describe the same task reflected
    across the world's x-axis. Both episodes are actually run in Gazebo (PPO
    stays on-policy). Corner tasks are mixed only into Stage 2.
    """
    if not math.isfinite(corner_probability) or not 0 <= corner_probability <= 1:
        raise ValueError("corner_probability must be in [0, 1].")
    task_seed = int(seed) - int(seed) % 2 if mirror_pair else int(seed)
    choose_corner = (curriculum_level == 1 and
                     np.random.default_rng(task_seed + 314159).random() < corner_probability)
    episode = (_sample_corner_episode(task_seed, practice_phase) if choose_corner
               else _sample_random_episode(task_seed, curriculum_level, practice_phase))
    if not mirror_pair or int(seed) % 2 == 0:
        return episode
    return replace(episode,
        layout={name: (x, -y, z, -yaw) for name, (x, y, z, yaw) in episode.layout.items()},
        start=(episode.start[0], -episode.start[1], -episode.start[2]),
        goal=(episode.goal[0], -episode.goal[1]),
        region_center=(episode.region_center[0], -episode.region_center[1]), mirrored=True)


def _disc_segment_is_clear(start, end, layout, radius):
    """Check a circular footprint along a straight segment (including corners).

    Near a box corner, radial clearance differs from square inflation. Check
    endpoint-to-box and box-corner-to-segment distances, plus intersections.
    This is only a reward helper; it does not change episode termination.
    """
    if any(max(abs(v) for v in point) > ARENA_HALF_M - radius for point in (start, end)):
        return False
    if straight_path_blocker_count(start, end, layout, clearance_m=0) > 0:
        return False
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    for bx, by, _, _ in layout.values():
        for x, y in (start, end):
            distance = math.hypot(max(abs(x - bx) - BOX_HALF_M, 0),
                                  max(abs(y - by) - BOX_HALF_M, 0))
            if distance < radius:
                return False
        for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            x, y = bx + sx * BOX_HALF_M, by + sy * BOX_HALF_M
            fraction = (0.0 if length_squared == 0 else
                        float(np.clip(((x - start[0]) * dx + (y - start[1]) * dy) / length_squared, 0, 1)))
            if math.hypot(x - start[0] - fraction * dx, y - start[1] - fraction * dy) < radius:
                return False
    return True


class DetourDistance:
    """Training reward helper: distance around boxes with a safety margin.

    Nodes are the goal and corners of inflated boxes. Dijkstra computes their
    distances to the goal once per reset. distance() returns ONE NUMBER; no
    route, next waypoint, preferred turn or action is exposed to the policy.
    This is a geometric heuristic, not a shortest-time differential-drive plan.
    """

    def __init__(self, layout: Mapping[str, Pose], goal: tuple[float, float],
                 clearance_m: float = 0.30):
        _check_layout(layout)
        if not math.isfinite(clearance_m) or clearance_m < PATH_CLEARANCE_M:
            raise ValueError("Detour clearance must be at least the reachability margin.")
        self.layout, self.goal, self.clearance = dict(layout), goal, clearance_m
        self.limit = ARENA_HALF_M - clearance_m
        self.corner_offset = BOX_HALF_M + clearance_m + 0.001
        points = [goal]
        for x, y, _, _ in layout.values():
            for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                corner = (x + dx * self.corner_offset, y + dy * self.corner_offset)
                if self._clear(corner):
                    points.append(corner)
        self.points = points
        edges = [[] for _ in points]
        for i, a in enumerate(points):
            for j in range(i):
                b = points[j]
                if self._visible(a, b):
                    length = math.dist(a, b)
                    edges[i].append((j, length))
                    edges[j].append((i, length))
        self.costs = [float("inf")] * len(points)
        self.costs[0] = 0.0
        pending = [(0.0, 0)]
        while pending:
            cost, index = heapq.heappop(pending)
            if cost != self.costs[index]:
                continue
            for neighbour, length in edges[index]:
                candidate = cost + length
                if candidate < self.costs[neighbour]:
                    self.costs[neighbour] = candidate
                    heapq.heappush(pending, (candidate, neighbour))

    def _clear(self, point):
        return point_is_clear(point, self.layout, self.clearance)

    def _visible(self, a, b):
        return straight_path_is_clear(a, b, self.layout, self.clearance)

    def _safe_distance(self, point):
        return min((math.dist(point, target) + cost
                    for target, cost in zip(self.points, self.costs)
                    if math.isfinite(cost) and self._visible(point, target)),
                   default=float("inf"))

    def distance(self, point: tuple[float, float]) -> float:
        """Remain defined when the robot enters the preferred safety margin.

        Project to safe space, checking the escape segment against the original
        collision margin. A larger escape cost discourages hugging corners.
        A bounded fallback handles disconnected/numerically degenerate states.
        """
        if self._clear(point):
            return min(20.0, self._safe_distance(point))
        x, y = point
        exits = [(float(np.clip(x, -self.limit + 0.001, self.limit - 0.001)),
                  float(np.clip(y, -self.limit + 0.001, self.limit - 0.001)))]
        for bx, by, _, _ in self.layout.values():
            if max(abs(x - bx), abs(y - by)) <= self.corner_offset:
                exits.extend(((bx - self.corner_offset, y), (bx + self.corner_offset, y),
                              (x, by - self.corner_offset), (x, by + self.corner_offset)))
        candidates = [3.0 * math.dist(point, exit_point) + self._safe_distance(exit_point)
                      for exit_point in exits if self._clear(exit_point)
                      and _disc_segment_is_clear(point, exit_point, self.layout, COLLISION_DISTANCE_M)]
        return min(20.0, min(candidates, default=20.0))


def ensure_checkpoint_environment(metadata: Mapping) -> None:
    if metadata.get('environment_id') != ENVIRONMENT_ID:
        raise ValueError('This checkpoint belongs to another environment (or has no arena ID). '
                         'Start a fresh curriculum run; resume only v3 curriculum checkpoints.')


# ---------------- Observation, action and reward math ----------------

LIDAR_BINS = 72
OBSERVATION_SIZE = 77

LIDAR_MIN_M = 0.15
LIDAR_CAP_M = 8.0
MAX_INVALID_SCAN_FRACTION = 0.10
MAX_INVALID_SCAN_RETRIES = 3
# The 9 m square arena diagonal is approximately 12.73 m.
GOAL_DISTANCE_CAP_M = 13.0

MAX_LINEAR_SPEED_MPS = 0.22

MAX_ANGULAR_SPEED_RADPS = 1.20
CONTROL_INTERVAL_SEC = 0.10

# ------------------------- Reward/termination contract ---------------------

GOAL_REACHED_DISTANCE_M = GOAL_SUCCESS_RADIUS_M

# LiDAR is mounted at the chassis centre. Keep the existing 0.22 m termination
# threshold for comparable tests; the 0.30 m detour margin only shapes rewards.
COLLISION_DISTANCE_M = 0.22

# Start discouraging the robot before reaching collision distance.
SAFETY_DISTANCE_M = 0.35

MAX_EPISODE_STEPS = {level: stage.max_steps for level, stage in STAGES.items()}

PROGRESS_REWARD_SCALE = 5.0
HEADING_REWARD_SCALE = 0.0  # Do not reward facing the goal while a detour is needed.
STEP_PENALTY = -0.01
NEAR_OBSTACLE_PENALTY_SCALE = 0.08
STUCK_PENALTY = -0.05

SUCCESS_REWARD = 25.0
COLLISION_PENALTY = -25.0
TIMEOUT_PENALTY = -5.0


@dataclass(frozen=True)
class RewardSettings:
    """Reward profile; defaults reproduce existing V3/V5 checkpoints."""

    safety_distance_m: float = SAFETY_DISTANCE_M
    near_obstacle_penalty_scale: float = NEAR_OBSTACLE_PENALTY_SCALE
    progress_mode: str = "euclidean"  # Old checkpoints keep their original reward.
    discount_factor: float = 0.99
    detour_clearance_m: float = 0.30
    braking_penalty_scale: float = 0.0  # Old checkpoints keep their original reward.

    @classmethod
    def from_mapping(cls, values: Mapping) -> "RewardSettings":
        if not isinstance(values, Mapping):
            raise TypeError("reward must be a mapping.")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("Unknown reward settings: " + ", ".join(sorted(unknown)))
        settings = cls(**dict(values))
        if settings.progress_mode not in ("euclidean", "detour_potential"):
            raise ValueError("reward.progress_mode must be euclidean or detour_potential.")
        if not 0 < settings.discount_factor < 1:
            raise ValueError("reward.discount_factor must be in (0, 1).")
        if not math.isfinite(settings.detour_clearance_m) or not PATH_CLEARANCE_M <= settings.detour_clearance_m < 0.325:
            raise ValueError("Detour clearance must be in [0.25, 0.325), to retain the 0.65 m gaps.")
        if (isinstance(settings.safety_distance_m, bool)
                or not math.isfinite(settings.safety_distance_m)
                or settings.safety_distance_m <= COLLISION_DISTANCE_M):
            raise ValueError("reward.safety_distance_m must exceed the collision distance.")
        if (isinstance(settings.near_obstacle_penalty_scale, bool)
                or not math.isfinite(settings.near_obstacle_penalty_scale)
                or settings.near_obstacle_penalty_scale < 0):
            raise ValueError("reward.near_obstacle_penalty_scale must be finite and nonnegative.")
        if (isinstance(settings.braking_penalty_scale, bool)
                or not math.isfinite(settings.braking_penalty_scale)
                or settings.braking_penalty_scale < 0):
            raise ValueError("reward.braking_penalty_scale must be finite and nonnegative.")
        return settings


@dataclass(frozen=True)
class ActionSettings:
    """Reserve part of the speed action for zero-speed, freely chosen turns."""

    stop_fraction: float = 0.0  # Legacy mapping. New YAML selects 0.15.

    @classmethod
    def from_mapping(cls, values: Mapping) -> "ActionSettings":
        if not isinstance(values, Mapping):
            raise TypeError("action must be a mapping.")
        if set(values) - {"stop_fraction"}:
            raise ValueError("Unknown action settings.")
        settings = cls(**dict(values))
        if isinstance(settings.stop_fraction, bool) or not 0 <= settings.stop_fraction < 0.5:
            raise ValueError("action.stop_fraction must be in [0, 0.5).")
        return settings


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
    action_settings: ActionSettings = ActionSettings(),
) -> tuple[float, float]:

    action_array = np.asarray(action, dtype=np.float32)

    if action_array.shape != (2,):
        raise ValueError("Action must have shape (2,)")

    if not np.isfinite(action_array).all():
        raise ValueError("Action must contain finite values")

    clipped_action = np.clip(action_array, -1, 1)

    speed_fraction = (clipped_action[0] + 1) / 2
    # A plateau at zero is reachable without saturating the policy's tanh.
    # Above it, smoothly rescale the remaining range to the same maximum speed.
    moving_fraction = max(0.0, float(speed_fraction) - action_settings.stop_fraction)
    if action_settings.stop_fraction > 0 and moving_fraction < 1e-7:
        moving_fraction = 0.0  # Float32 round-off at the stop boundary.
    linear = moving_fraction / (1.0 - action_settings.stop_fraction) * MAX_LINEAR_SPEED_MPS
    angular = clipped_action[1] * MAX_ANGULAR_SPEED_RADPS

    return float(linear), float(angular)

def velocity_to_policy_action(
    linear_velocity_mps: float,
    angular_velocity_radps: float,
    action_settings: ActionSettings = ActionSettings(),
) -> np.ndarray:

    if not np.isfinite(
        [linear_velocity_mps, angular_velocity_radps]
    ).all():
        raise ValueError("Velocities must contain finite values")
    clipped_linear = np.clip(linear_velocity_mps, 0, MAX_LINEAR_SPEED_MPS)
    clipped_angular = np.clip(angular_velocity_radps, -MAX_ANGULAR_SPEED_RADPS, MAX_ANGULAR_SPEED_RADPS)

    stop = action_settings.stop_fraction
    speed_fraction = stop + (1 - stop) * clipped_linear / MAX_LINEAR_SPEED_MPS
    linear_action = 2 * speed_fraction - 1 if clipped_linear > 0 else -1 + stop
    angular_action = clipped_angular / MAX_ANGULAR_SPEED_RADPS

    return np.array([linear_action, angular_action], dtype=np.float32)

def determine_episode_outcome(
    goal_distance_m: float,
    nearest_obstacle_m: float,
    step_count: int,
    max_episode_steps: int,
    goal_reached: bool | None = None,
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

    # Runtime and distance-only callers use the same circular-goal radius.
    if goal_reached is None:
        goal_reached = goal_distance_m <= GOAL_REACHED_DISTANCE_M
    if goal_reached:
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
    stuck: bool = False,
    reward_settings: RewardSettings = RewardSettings(),
    previous_detour_distance_m: float | None = None,
    current_detour_distance_m: float | None = None,
    previous_nearest_obstacle_m: float | None = None,
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
    if reward_settings.progress_mode == "detour_potential":
        if previous_detour_distance_m is None or current_detour_distance_m is None:
            raise ValueError("Detour rewards require both distance measurements.")
        if not np.isfinite([previous_detour_distance_m, current_detour_distance_m]).all():
            raise ValueError("Detour distances must be finite.")
        # Ng et al. (1999): F = gamma * Phi(next_state) - Phi(state).
        # Phi is negative safe-route distance, bounded to [-20, 0].
        # True terminal states have Phi=0. Time limits are truncations: keep
        # Phi(next_state), because PPO bootstraps their value rather than ending
        # the underlying task. This is not the old raw distance difference.
        previous_potential = -float(np.clip(previous_detour_distance_m, 0, 20))
        next_potential = (0.0 if outcome in ("success", "collision") else
                          -float(np.clip(current_detour_distance_m, 0, 20)))
        progress_reward = reward_settings.discount_factor * next_potential - previous_potential

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

    if nearest_obstacle_m < reward_settings.safety_distance_m:
        unsafe_fraction = float(
            np.clip(
                (
                    reward_settings.safety_distance_m
                    - nearest_obstacle_m
                )
                / (
                    reward_settings.safety_distance_m
                    - COLLISION_DISTANCE_M
                ),
                0.0,
                1.0,
            )
        )

        obstacle_penalty = (
            -reward_settings.near_obstacle_penalty_scale
            * unsafe_fraction
        )

    terminal_reward = terminal_rewards[outcome]
    # Teach braking when moving toward a nearby obstacle. Turning in place,
    # opening the clearance, and driving in clear space get no added penalty.
    # A 2 mm deadband reduces reactions to the configured 1 mm LiDAR noise.
    braking_penalty = 0.0
    if reward_settings.braking_penalty_scale > 0:
        if (previous_nearest_obstacle_m is None
                or not math.isfinite(previous_nearest_obstacle_m)
                or previous_nearest_obstacle_m < 0):
            raise ValueError("Braking rewards need a finite previous obstacle distance.")
        danger = float(np.clip(
            (reward_settings.safety_distance_m - nearest_obstacle_m)
            / (reward_settings.safety_distance_m - COLLISION_DISTANCE_M), 0, 1))
        closing = float(np.clip(
            (previous_nearest_obstacle_m - nearest_obstacle_m - 0.002)
            / (MAX_LINEAR_SPEED_MPS * CONTROL_INTERVAL_SEC), 0, 1))
        braking_penalty = (-reward_settings.braking_penalty_scale
                           * danger * closing * forward_fraction ** 2)
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
        + braking_penalty
        + stuck_penalty
        + terminal_reward
    )

    components = {
        "progress_m": float(progress_m),
        "progress_reward": float(progress_reward),
        "heading_reward": float(heading_reward),
        "step_penalty": float(STEP_PENALTY),
        "obstacle_penalty": float(obstacle_penalty),
        "braking_penalty": float(braking_penalty),
        "stuck_penalty": float(stuck_penalty),
        "terminal_reward": float(terminal_reward),
        "total_reward": total_reward,
    }

    return total_reward, components

