"""Stress-test RosNavEnv with deterministic random actions before PPO."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import rclpy

from nav_learning.ros_nav_env import (
    MAX_EPISODE_STEPS,
    OBSERVATION_SIZE,
    RosNavEnv,
)


CURRICULUM_LEVELS = (0, 1, 2)
EPISODES_PER_LEVEL = 2
MAX_VALIDATION_STEPS = 120
ACTION_HOLD_STEPS = 4
ACTION_SEED = 20260828
RESET_SEED_BASE = 1000

OUTCOME_FLAGS = {
    "running": (False, False),
    "success": (True, False),
    "collision": (True, False),
    "timeout": (False, True),
}

REWARD_TERMS = (
    "progress_reward",
    "heading_reward",
    "step_penalty",
    "obstacle_penalty",
    "waypoint_reward",
    "stuck_penalty",
    "terminal_reward",
)


def check_observation(observation: np.ndarray) -> None:
    """Check shape, dtype, finiteness, and normalised bounds."""

    if observation.shape != (OBSERVATION_SIZE,):
        raise RuntimeError(
            f"Wrong observation shape: {observation.shape}."
        )

    if observation.dtype != np.float32:
        raise RuntimeError(
            f"Wrong observation dtype: {observation.dtype}."
        )

    if not np.isfinite(observation).all():
        raise RuntimeError(
            "Observation contains NaN or infinity."
        )

    if np.any(np.abs(observation) > 1.00001):
        raise RuntimeError(
            "Observation escaped normalised bounds."
        )


def check_transition(
    observation: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    expected_step: int,
) -> None:
    """Check one complete step() result."""

    check_observation(observation)

    if not math.isfinite(reward):
        raise RuntimeError("Reward is not finite.")

    outcome = info["outcome"]

    if outcome not in OUTCOME_FLAGS:
        raise RuntimeError(
            f"Unknown outcome: {outcome}"
        )

    if (terminated, truncated) != OUTCOME_FLAGS[outcome]:
        raise RuntimeError(
            f"Outcome/flag mismatch for {outcome}."
        )

    if info["step_count"] != expected_step:
        raise RuntimeError(
            f"Step count {info['step_count']} "
            f"!= {expected_step}."
        )

    action = np.asarray(
        info["executed_action"],
        dtype=np.float32,
    )

    if action.shape != (2,):
        raise RuntimeError(
            "Executed action shape is not (2,)."
        )

    if not np.isfinite(action).all():
        raise RuntimeError(
            "Executed action is not finite."
        )

    if np.any(np.abs(action) > 1.00001):
        raise RuntimeError(
            "Executed action escaped [-1, 1]."
        )

    components = info["reward_components"]

    rebuilt_reward = sum(
        float(components[name])
        for name in REWARD_TERMS
    )

    if not math.isclose(
        reward,
        rebuilt_reward,
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise RuntimeError(
            "Reward components do not sum correctly."
        )


def check_real_timeout(
    environment: RosNavEnv,
) -> None:
    """Reach the real timeout branch without waiting for 300 steps."""

    environment.reset(
        seed=900,
        curriculum_level=0,
    )

    # Move directly to one step before the level-0 limit.
    # This is only a validation shortcut.
    environment._step_count = (
        MAX_EPISODE_STEPS[0] - 1
    )

    (
        _,
        _,
        terminated,
        truncated,
        info,
    ) = environment.step(
        action=(-1.0, 0.0)
    )

    if (
        terminated
        or not truncated
        or info["outcome"] != "timeout"
    ):
        raise RuntimeError(
            "Real timeout transition failed."
        )

    # After timeout, another action must be rejected
    # until reset() is called again.
    try:
        environment.step(
            action=(-1.0, 0.0)
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "Action was accepted after timeout."
        )


def run_episode(
    environment: RosNavEnv,
    rng: np.random.Generator,
    level: int,
    episode_index: int,
) -> dict:
    """Run one bounded random-policy episode."""

    reset_seed = (
        RESET_SEED_BASE
        + level * 100
        + episode_index
    )

    observation, reset_info = environment.reset(
        seed=reset_seed,
        curriculum_level=level,
    )

    check_observation(observation)

    action = np.zeros(
        2,
        dtype=np.float32,
    )

    total_reward = 0.0
    minimum_clearance = float("inf")
    final_info = reset_info
    outcome = "validation_cap"

    for step_index in range(
        MAX_VALIDATION_STEPS
    ):
        # Hold each random action for four steps.
        # This produces smoother motion than changing
        # direction every 0.10 seconds.
        if step_index % ACTION_HOLD_STEPS == 0:
            action = rng.uniform(
                -1.0,
                1.0,
                2,
            ).astype(np.float32)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = environment.step(action)

        steps = step_index + 1

        check_transition(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            expected_step=steps,
        )

        total_reward += reward

        minimum_clearance = min(
            minimum_clearance,
            float(
                info["nearest_obstacle_m"]
            ),
        )

        final_info = info

        if terminated or truncated:
            outcome = info["outcome"]
            break

    else:
        # The validator intentionally stops after
        # 120 steps, even if the episode is still running.
        steps = MAX_VALIDATION_STEPS
        environment.publish_stop()

    return {
        "level": level,
        "episode": episode_index,
        "seed": reset_seed,
        "steps": steps,
        "reward": float(total_reward),
        "outcome": outcome,
        "clearance": float(
            minimum_clearance
        ),
        "distance": float(
            final_info["distance_to_goal_m"]
        ),
    }


def main(args=None) -> None:
    """Validate timeout handling and six random rollouts."""

    rclpy.init(args=args)
    environment = None

    try:
        environment = RosNavEnv()

        rng = np.random.default_rng(
            ACTION_SEED
        )

        check_real_timeout(environment)

        environment.get_logger().info(
            "Real timeout transition passed."
        )

        results = []

        for level in CURRICULUM_LEVELS:
            for episode_index in range(
                EPISODES_PER_LEVEL
            ):
                result = run_episode(
                    environment=environment,
                    rng=rng,
                    level=level,
                    episode_index=episode_index,
                )

                results.append(result)

                environment.get_logger().info(
                    "Random episode passed: "
                    f"level={result['level']}, "
                    f"episode={result['episode']}, "
                    f"seed={result['seed']}, "
                    f"steps={result['steps']}, "
                    f"reward={result['reward']:.3f}, "
                    f"outcome={result['outcome']}, "
                    f"min_clearance="
                    f"{result['clearance']:.3f} m, "
                    f"final_distance="
                    f"{result['distance']:.3f} m"
                )

        counts = Counter(
            result["outcome"]
            for result in results
        )

        total_steps = sum(
            result["steps"]
            for result in results
        )

        mean_reward = float(
            np.mean(
                [
                    result["reward"]
                    for result in results
                ]
            )
        )

        minimum_clearance = min(
            result["clearance"]
            for result in results
        )

        environment.get_logger().info(
            "Random-policy validation passed: "
            f"episodes={len(results)}, "
            f"steps={total_steps}, "
            f"mean_reward={mean_reward:.3f}, "
            f"outcomes={dict(counts)}, "
            f"minimum_clearance="
            f"{minimum_clearance:.3f} m"
        )

    finally:
        if environment is not None:
            environment.close()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
