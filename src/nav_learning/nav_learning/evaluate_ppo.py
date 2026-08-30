"""Evaluate a trained PPO navigation policy on fixed unseen seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from nav_learning.ppo import (
    choose_device,
    load_checkpoint,
    observation_to_tensor,
)


EPISODE_FIELDS = (
    "curriculum_level",
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
    "route_waypoints_completed",
    "route_waypoint_count",
    "stuck_steps",
    "stuck_step_fraction",
    "start_room",
    "goal_room",
)

TRAJECTORY_FIELDS = (
    "curriculum_level",
    "episode",
    "seed",
    "step",
    "robot_x",
    "robot_y",
    "goal_x",
    "goal_y",
    "active_target_x",
    "active_target_y",
    "active_target_kind",
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
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default=None)
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
    checkpoint_path: Path,
    output_directory: Path,
    levels: Sequence[int],
    episodes_per_level: int,
    seed_base: int,
    requested_device: str,
    stochastic: bool,
    trajectory_episodes_per_level: int,
) -> dict[str, Any]:
    """Run policy episodes and save per-episode and trajectory data."""

    import rclpy

    from nav_learning.ros_nav_env import MAX_EPISODE_STEPS, RosNavEnv

    device = choose_device(requested_device)
    agent, payload = load_checkpoint(
        checkpoint_path,
        device=device,
        load_optimizer=False,
    )
    agent.model.eval()
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed_base)
    np.random.seed(seed_base)

    episode_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    environment = None
    rclpy.init(args=None)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluation device: {device}")
    print(f"Evaluation output: {output_directory}")

    try:
        environment = RosNavEnv()
        for level in levels:
            for episode_index in range(episodes_per_level):
                seed = seed_base + level * 10000 + episode_index
                observation, reset_info = environment.reset(
                    seed=seed,
                    curriculum_level=level,
                )
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
                            "episode": episode_index,
                            "seed": seed,
                            "step": 0,
                            "robot_x": previous_x,
                            "robot_y": previous_y,
                            "goal_x": goal_x,
                            "goal_y": goal_y,
                            "active_target_x": reset_info.get(
                                "active_target_x",
                                goal_x,
                            ),
                            "active_target_y": reset_info.get(
                                "active_target_y",
                                goal_y,
                            ),
                            "active_target_kind": reset_info.get(
                                "active_target_kind",
                                "goal",
                            ),
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
                                "episode": episode_index,
                                "seed": seed,
                                "step": step_index + 1,
                                "robot_x": current_x,
                                "robot_y": current_y,
                                "goal_x": goal_x,
                                "goal_y": goal_y,
                                "active_target_x": info.get(
                                    "active_target_x",
                                    goal_x,
                                ),
                                "active_target_y": info.get(
                                    "active_target_y",
                                    goal_y,
                                ),
                                "active_target_kind": info.get(
                                    "active_target_kind",
                                    "goal",
                                ),
                                "outcome": info["outcome"],
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
                    initial_distance / path_length
                    if success and path_length > 1e-9
                    else float("nan")
                )
                row = {
                    "curriculum_level": level,
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
                    "route_waypoints_completed": int(
                        final_info.get("route_waypoint_index", 0)
                    ),
                    "route_waypoint_count": int(
                        final_info.get("route_waypoint_count", 0)
                    ),
                    "stuck_steps": stuck_steps,
                    "stuck_step_fraction": (
                        stuck_steps / int(final_info["step_count"])
                        if int(final_info["step_count"]) > 0
                        else 0.0
                    ),
                    "start_room": reset_info.get("start_room", ""),
                    "goal_room": reset_info.get("goal_room", ""),
                }
                episode_rows.append(row)
                print(
                    f"level={level} episode={episode_index + 1}/"
                    f"{episodes_per_level} seed={seed} "
                    f"outcome={outcome} return={total_reward:.3f} "
                    f"steps={row['length']} "
                    f"clearance={minimum_clearance:.3f} "
                    f"route={row['route_waypoints_completed']}/"
                    f"{row['route_waypoint_count']}"
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
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
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
    checkpoint_path = Path(arguments.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_directory = (
        Path(arguments.output_dir).expanduser().resolve()
        if arguments.output_dir
        else _default_output_directory(checkpoint_path).resolve()
    )
    run_evaluation(
        checkpoint_path=checkpoint_path,
        output_directory=output_directory,
        levels=tuple(dict.fromkeys(arguments.levels)),
        episodes_per_level=arguments.episodes_per_level,
        seed_base=arguments.seed_base,
        requested_device=arguments.device,
        stochastic=arguments.stochastic,
        trajectory_episodes_per_level=(
            arguments.trajectory_episodes_per_level
        ),
    )


if __name__ == "__main__":
    main()
