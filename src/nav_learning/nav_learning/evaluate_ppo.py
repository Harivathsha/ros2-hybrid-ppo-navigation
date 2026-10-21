"""Evaluate a trained PPO navigation policy on fixed unseen seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from nav_learning.nav_math import ENVIRONMENT_ID, RewardSettings, ActionSettings, ensure_checkpoint_environment

from nav_learning.ppo import (
    choose_device,
    load_checkpoint,
    observation_to_tensor,
)


def goal_seeking_action(observation: np.ndarray) -> np.ndarray:
    if observation.shape != (77,) or not np.isfinite(observation).all():
        raise ValueError('Expected a finite 77-value observation.')
    bearing = math.atan2(float(observation[73]), float(observation[74]))
    forward_fraction = max(0.0, math.cos(bearing))
    turn_fraction = float(np.clip(2.0 * bearing / 1.2, -1, 1))
    return np.asarray([2.0 * forward_fraction - 1.0, turn_fraction], dtype=np.float32)


EPISODE_FIELDS = (
    "task_kind", "mirrored",
    "curriculum_level",
    "practice_phase",
    "region_size_m",
    "episode",
    "seed",
    "return",
    "length",
    "outcome",
    "success",
    "collision",
    "timeout",
    "initial_distance_m",
    "final_distance_m",
    "minimum_clearance_m",
    "path_length_m",
    "path_efficiency",
    "mean_linear_velocity_mps",
    "mean_absolute_angular_velocity_radps",
    "straight_path_blocked",
    "straight_path_blocker_count",
    "randomized_object_count",
    "stuck_steps",
    "stuck_step_fraction",
)

TRAJECTORY_FIELDS = (
    "robot_yaw", "commanded_linear_velocity_mps", "commanded_angular_velocity_radps",
    "nearest_obstacle_m", "relative_goal_bearing_rad",
    "curriculum_level",
    "practice_phase",
    "episode",
    "seed",
    "step",
    "robot_x",
    "robot_y",
    "goal_x",
    "goal_y",
    "outcome",
)


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write a complete evaluation CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for values in rows:
            writer.writerow(
                {name: values.get(name, "") for name in fields}
            )


def _mean_or_none(values: Iterable[float]) -> float | None:
    """Return the mean of finite values, or null for no valid values."""

    finite_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    if not finite_values:
        return None
    return float(sum(finite_values) / len(finite_values))


def _summarise_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate outcome rates and continuous evaluation metrics."""

    count = len(rows)
    if count == 0:
        raise ValueError("Cannot summarise zero evaluation episodes.")
    successes = [row for row in rows if int(row["success"]) == 1]
    return {
        "episodes": count,
        "success_rate": sum(int(row["success"]) for row in rows)
        / count,
        "collision_rate": sum(int(row["collision"]) for row in rows)
        / count,
        "timeout_rate": sum(int(row["timeout"]) for row in rows)
        / count,
        "mean_return": _mean_or_none(row["return"] for row in rows),
        "mean_length": _mean_or_none(row["length"] for row in rows),
        "mean_final_distance_m": _mean_or_none(
            row["final_distance_m"] for row in rows
        ),
        "mean_minimum_clearance_m": _mean_or_none(
            row["minimum_clearance_m"] for row in rows
        ),
        "mean_success_path_efficiency": _mean_or_none(
            row["path_efficiency"] for row in successes
        ),
        "mean_stuck_step_fraction": _mean_or_none(
            row["stuck_step_fraction"] for row in rows
        ),
    }


def _parse_arguments(args: Sequence[str] | None) -> argparse.Namespace:
    """Parse evaluation command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a PPO checkpoint with fixed unseen seeds.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--baseline", choices=("goal-seeking",), default=None)
    parser.add_argument("--practice-phase", type=int, choices=(0, 1, 2), default=None,
                        help="Evaluate Stage 2 detour practice; requires --levels 1.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--task-suite", choices=("standard", "corners"), default="standard",
                        help="corners runs mirrored Stage 2 corner tasks; requires --levels 1.")
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        choices=(0, 1, 2),
        default=(0, 1, 2),
    )
    parser.add_argument("--episodes-per-level", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=50000)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using tanh(mean).",
    )
    parser.add_argument(
        "--trajectory-episodes-per-level",
        type=int,
        default=3,
    )
    arguments = parser.parse_args(args)
    if arguments.practice_phase is not None and tuple(dict.fromkeys(arguments.levels)) != (1,):
        parser.error("--practice-phase requires --levels 1.")
    if arguments.task_suite == "corners" and tuple(dict.fromkeys(arguments.levels)) != (1,):
        parser.error("--task-suite corners requires --levels 1.")
    if not arguments.checkpoint and not arguments.baseline:
        parser.error("Provide --checkpoint or --baseline goal-seeking.")
    if arguments.checkpoint and arguments.baseline:
        parser.error("Choose either a PPO checkpoint or the baseline.")
    if arguments.episodes_per_level <= 0:
        parser.error("--episodes-per-level must be positive.")
    if arguments.seed_base < 0:
        parser.error("--seed-base must be nonnegative.")
    if arguments.trajectory_episodes_per_level < 0:
        parser.error(
            "--trajectory-episodes-per-level must be nonnegative."
        )
    return arguments


def _default_output_directory(checkpoint_path: Path) -> Path:
    """Place evaluation output beside the checkpoint's run directory."""

    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "evaluation"
    return checkpoint_path.parent / "evaluation"


def run_evaluation(
    checkpoint_path: Path | None,
    output_directory: Path,
    levels: Sequence[int],
    episodes_per_level: int,
    seed_base: int,
    requested_device: str,
    stochastic: bool,
    trajectory_episodes_per_level: int,
    baseline: bool = False,
    practice_phase: int | None = None,
    task_suite: str = "standard",
) -> dict[str, Any]:
    """Run policy episodes and save per-episode and trajectory data."""

    import rclpy

    from nav_learning.ros_nav_env import MAX_EPISODE_STEPS, RosNavEnv

    device = choose_device('cpu' if baseline else requested_device)
    if task_suite not in ("standard", "corners") or (task_suite == "corners" and tuple(levels) != (1,)):
        raise ValueError("Corner evaluation requires only level 1.")
    if practice_phase is not None and (tuple(levels) != (1,) or practice_phase not in (0, 1, 2)):
        raise ValueError("Practice evaluation requires only level 1 and phase 0, 1 or 2.")
    if baseline:
        agent, payload = None, {"training_state": {}}
    else:
        agent, payload = load_checkpoint(checkpoint_path, device=device, load_optimizer=False)
        ensure_checkpoint_environment(payload.get("metadata", {}))
        agent.model.eval()
    reward_settings = RewardSettings.from_mapping(
        payload.get("metadata", {}).get("reward_settings", {})
    )
    action_settings = ActionSettings.from_mapping(payload.get("metadata", {}).get("action_settings", {}))
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed_base)
    np.random.seed(seed_base)

    episode_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    layout_rows = []
    environment = None
    rclpy.init(args=None)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluation device: {device}")
    print(f"Evaluation output: {output_directory}")

    try:
        environment = RosNavEnv(reward_settings=reward_settings, action_settings=action_settings)
        for level in levels:
            for episode_index in range(episodes_per_level):
                seed = seed_base + level * 10000 + episode_index
                reset_options = {} if practice_phase is None else {"practice_phase": practice_phase}
                if task_suite == "corners":
                    reset_options.update(corner_probability=1.0, mirror_pair=True)
                observation, reset_info = environment.reset(
                    seed=seed,
                    curriculum_level=level,
                    **reset_options,
                )
                layout_rows.append({
                    "curriculum_level": level, "episode": episode_index, "seed": seed,
                    "practice_phase": -1 if practice_phase is None else practice_phase,
                    "task_kind": reset_info.get("task_kind", "random"),
                    "mirrored": reset_info.get("mirrored", False),
                    "start": list(reset_info["requested_start"]), "goal": list(reset_info["goal"]),
                    "straight_path_blocked": bool(reset_info["straight_path_blocked"]),
                    "straight_path_blocker_count": int(reset_info["straight_path_blocker_count"]),
                    "object_layout": reset_info["object_layout"],
                    "region_center": reset_info.get("region_center", (0, 0)),
                    "region_size_m": reset_info.get("region_size_m", 9),
                })
                goal_x, goal_y = (
                    float(reset_info["goal"][0]),
                    float(reset_info["goal"][1]),
                )
                previous_x = float(reset_info["robot_x"])
                previous_y = float(reset_info["robot_y"])
                record_trajectory = (
                    episode_index < trajectory_episodes_per_level
                )
                if record_trajectory:
                    trajectory_rows.append(
                        {
                            "curriculum_level": level,
                            "practice_phase": -1 if practice_phase is None else practice_phase,
                            "episode": episode_index,
                            "seed": seed,
                            "step": 0,
                            "robot_x": previous_x,
                            "robot_y": previous_y,
                            "goal_x": goal_x,
                            "goal_y": goal_y,
                            "outcome": "start",
                        }
                    )

                total_reward = 0.0
                path_length = 0.0
                minimum_clearance = float("inf")
                linear_velocities: list[float] = []
                angular_velocities: list[float] = []
                stuck_steps = 0
                final_info: Mapping[str, Any] = reset_info

                for step_index in range(MAX_EPISODE_STEPS[level] + 1):
                    if baseline:
                        action_array = goal_seeking_action(observation)
                    else:
                        observation_tensor = observation_to_tensor(
                            observation,
                            device,
                        )
                        with torch.inference_mode():
                            if stochastic:
                                action, _, _ = agent.model.sample_action(
                                    observation_tensor
                                )
                            else:
                                action = agent.model.deterministic_action(
                                    observation_tensor
                                )
                        action_array = (
                            action.detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = environment.step(action_array)
                    total_reward += float(reward)
                    current_x = float(info["robot_x"])
                    current_y = float(info["robot_y"])
                    path_length += math.hypot(
                        current_x - previous_x,
                        current_y - previous_y,
                    )
                    previous_x = current_x
                    previous_y = current_y
                    minimum_clearance = min(
                        minimum_clearance,
                        float(info["nearest_obstacle_m"]),
                    )
                    linear_velocities.append(
                        float(info["commanded_linear_velocity_mps"])
                    )
                    angular_velocities.append(
                        abs(
                            float(
                                info[
                                    "commanded_angular_velocity_radps"
                                ]
                            )
                        )
                    )
                    stuck_steps += int(bool(info.get("stuck", False)))
                    final_info = info
                    if record_trajectory:
                        trajectory_rows.append(
                            {
                                "curriculum_level": level,
                                "practice_phase": -1 if practice_phase is None else practice_phase,
                                "episode": episode_index,
                                "seed": seed,
                                "step": step_index + 1,
                                "robot_x": current_x,
                                "robot_y": current_y,
                                "goal_x": goal_x,
                                "goal_y": goal_y,
                                "outcome": info["outcome"],
                                **{key: info.get(key, "") for key in (
                                    "robot_yaw", "commanded_linear_velocity_mps",
                                    "commanded_angular_velocity_radps", "nearest_obstacle_m",
                                    "relative_goal_bearing_rad")},
                            }
                        )
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError(
                        "Environment exceeded its declared episode limit."
                    )

                outcome = str(final_info["outcome"])
                success = outcome == "success"
                initial_distance = float(
                    reset_info["distance_to_goal_m"]
                )
                path_efficiency = (
                    max(0.0, initial_distance - float(final_info["distance_to_goal_m"])) / path_length
                    if success and path_length > 1e-9
                    else float("nan")
                )
                row = {
                    "task_kind": reset_info.get("task_kind", "random"),
                    "mirrored": int(reset_info.get("mirrored", False)),
                    "curriculum_level": level,
                    "practice_phase": -1 if practice_phase is None else practice_phase,
                    "region_size_m": reset_info.get("region_size_m", 9),
                    "episode": episode_index,
                    "seed": seed,
                    "return": total_reward,
                    "length": int(final_info["step_count"]),
                    "outcome": outcome,
                    "success": int(success),
                    "collision": int(outcome == "collision"),
                    "timeout": int(outcome == "timeout"),
                    "initial_distance_m": initial_distance,
                    "final_distance_m": float(
                        final_info["distance_to_goal_m"]
                    ),
                    "minimum_clearance_m": minimum_clearance,
                    "path_length_m": path_length,
                    "path_efficiency": path_efficiency,
                    "mean_linear_velocity_mps": _mean_or_none(
                        linear_velocities
                    ),
                    "mean_absolute_angular_velocity_radps": (
                        _mean_or_none(angular_velocities)
                    ),
                    "straight_path_blocked": int(reset_info["straight_path_blocked"]),
                    "straight_path_blocker_count": int(reset_info["straight_path_blocker_count"]),
                    "randomized_object_count": reset_info["randomized_object_count"],
                    "stuck_steps": stuck_steps,
                    "stuck_step_fraction": (
                        stuck_steps / int(final_info["step_count"])
                        if int(final_info["step_count"]) > 0
                        else 0.0
                    ),
                }
                episode_rows.append(row)
                print(
                    f"level={level} episode={episode_index + 1}/"
                    f"{episodes_per_level} seed={seed} "
                    f"outcome={outcome} return={total_reward:.3f} "
                    f"steps={row['length']} "
                    f"clearance={minimum_clearance:.3f} "
                    f"direct_blockers={row['straight_path_blocker_count']}"
                )
    finally:
        if environment is not None:
            environment.close()
        if rclpy.ok():
            rclpy.shutdown()

    _write_csv(
        output_directory / "evaluation_episodes.csv",
        EPISODE_FIELDS,
        episode_rows,
    )
    _write_csv(
        output_directory / "evaluation_trajectories.csv",
        TRAJECTORY_FIELDS,
        trajectory_rows,
    )

    with (output_directory / "evaluation_layouts.json").open("w", encoding="utf-8") as stream:
        json.dump({"environment_id": ENVIRONMENT_ID, "episodes": layout_rows}, stream, indent=2)
        stream.write("\n")

    per_level = {
        str(level): _summarise_rows(
            [
                row
                for row in episode_rows
                if int(row["curriculum_level"]) == level
            ]
        )
        for level in levels
    }
    summary = {
        "environment_id": ENVIRONMENT_ID,
        "reward_settings": asdict(reward_settings),
        "action_settings": asdict(action_settings),
        "task_suite": task_suite,
        "control_mode": "gazebo_run_to_sim_time_v1",
        "practice_phase": practice_phase,
        "task_distribution": ("mirrored_corners" if task_suite == "corners" else
                              "standard_benchmark" if practice_phase is None else f"detour_practice_{practice_phase}"),
        "controller": "goal-seeking baseline" if baseline else "PPO final-goal policy",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_training_state": {
            "total_timesteps": payload["training_state"].get(
                "total_timesteps"
            ),
            "update_index": payload["training_state"].get(
                "update_index"
            ),
        },
        "deterministic": not stochastic,
        "seed_base": seed_base,
        "episodes_per_level": episodes_per_level,
        "levels": list(levels),
        "per_level": per_level,
        "overall": _summarise_rows(episode_rows),
        "blocked_direct_path_episodes": sum(row["straight_path_blocked"] for row in episode_rows),
        "blocked_direct_path_success_rate": _mean_or_none(
            row["success"] for row in episode_rows if row["straight_path_blocked"]),
    }
    summary_path = output_directory / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Evaluation summary: {summary_path}")
    return summary


def main(args: Sequence[str] | None = None) -> None:
    """Resolve paths and run fixed-seed PPO evaluation."""

    arguments = _parse_arguments(args)
    checkpoint_path = Path(arguments.checkpoint).expanduser().resolve() if arguments.checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_directory = (
        Path(arguments.output_dir).expanduser().resolve()
        if arguments.output_dir
        else (_default_output_directory(checkpoint_path).resolve() if checkpoint_path else Path("evaluation_baseline").resolve())
    )
    run_evaluation(
        checkpoint_path=checkpoint_path,
        output_directory=output_directory,
        levels=tuple(dict.fromkeys(arguments.levels)),
        episodes_per_level=arguments.episodes_per_level,
        seed_base=arguments.seed_base,
        requested_device=arguments.device,
        stochastic=arguments.stochastic,
        baseline=bool(arguments.baseline),
        practice_phase=arguments.practice_phase,
        task_suite=arguments.task_suite,
        trajectory_episodes_per_level=(
            arguments.trajectory_episodes_per_level
        ),
    )


if __name__ == "__main__":
    main()
