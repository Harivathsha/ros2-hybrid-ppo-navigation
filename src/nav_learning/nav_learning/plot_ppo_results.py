"""Generate training, PPO-diagnostic, evaluation, and trajectory plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle
from nav_learning.nav_math import ARENA_HALF_M, BOX_SIZE_M, GOAL_RADIUS_M, WALL_THICKNESS_M, STAGES, DETOUR_PHASE_NAMES


LEVEL_COLOURS = {0: "#2b8cbe", 1: "#f28e2b", 2: "#8e5ea2"}
LEVEL_LABELS = {0: "Stage 1 (3 m)", 1: "Stage 2 (6 m)", 2: "Stage 3 (9 m)"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file or return an empty list when it is absent."""

    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, str], name: str) -> float:
    """Convert one CSV value to float, preserving missing data as NaN."""

    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _task_label(level, row):
    phase = _number(row, "practice_phase")
    label = LEVEL_LABELS[level]
    if math.isfinite(phase) and 0 <= phase < 3:
        label += " | " + DETOUR_PHASE_NAMES[int(phase)].replace('_', ' ') + " practice"
    if row.get("task_kind") == "corner":
        label += " | corner test"
    return label


def _series(rows: Sequence[dict[str, str]], name: str) -> np.ndarray:
    """Extract one numeric NumPy series from CSV rows."""

    return np.asarray([_number(row, name) for row in rows], dtype=float)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a trailing mean that ignores NaN values."""

    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(values.size):
        start = max(0, index - window + 1)
        section = values[start:index + 1]
        finite = section[np.isfinite(section)]
        if finite.size:
            result[index] = float(finite.mean())
    return result


def _finish_figure(figure: plt.Figure, path: Path) -> Path:
    """Apply final layout, save at publication resolution, and close."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _current_stage_success(rows, window=100):
    """Reset at promotions and exclude easier rehearsal episodes."""
    result, recent, previous_stage = [], [], None
    for row in rows:
        stage = (row.get("training_stage", row.get("curriculum_level")),
                 row.get("training_practice_phase", "-1"))
        if stage != previous_stage:
            recent = []
            previous_stage = stage
        if _number(row, "rehearsal") != 1:
            recent.append(_number(row, "success"))
            recent = recent[-window:]
        result.append(float(np.mean(recent)) if recent else float("nan"))
    return np.asarray(result)


def plot_training_curves(
    rows: Sequence[dict[str, str]],
    output_path: Path,
) -> Path | None:
    """Plot return, success, episode length, and minimum clearance."""

    if not rows:
        return None
    episodes = _series(rows, "episode")
    returns = _series(rows, "return")
    lengths = _series(rows, "length")
    clearances = _series(rows, "minimum_clearance_m")
    levels = np.asarray([_number(row, "training_stage") if "training_stage" in row
                         else _number(row, "curriculum_level") for row in rows])
    window = min(50, max(5, len(rows) // 10))

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    figure.suptitle("PPO Navigation Training Curves", fontsize=15)

    axes[0, 0].plot(episodes, returns, alpha=0.25, color="#4c78a8")
    axes[0, 0].plot(
        episodes,
        _rolling_mean(returns, window),
        color="#1f4e79",
        label=f"rolling mean ({window})",
    )
    axes[0, 0].set_title("Episode return")
    axes[0, 0].legend()

    axes[0, 1].plot(
        episodes,
        _current_stage_success(rows) * 100.0,
        color="#2ca02c",
    )
    axes[0, 1].set_ylim(-2, 102)
    targets = np.asarray([_number(row, "promotion_threshold")
                          if row.get("promotion_threshold", "") != "" else 0.8 for row in rows])
    axes[0, 1].plot(episodes, targets * 100, color="#777777", linestyle="--", alpha=.6,
                    label="promotion target")
    axes[0, 1].set_title("Current task success (last 100; rehearsal excluded)")
    axes[0, 1].legend()
    axes[0, 1].set_ylabel("Success (%)")

    axes[1, 0].plot(
        episodes,
        _rolling_mean(lengths, window),
        color="#f28e2b",
    )
    axes[1, 0].set_title("Rolling episode length")
    axes[1, 0].set_ylabel("Environment steps")

    axes[1, 1].plot(
        episodes,
        _rolling_mean(clearances, window),
        color="#d62728",
        label="minimum clearance",
    )
    level_axis = axes[1, 1].twinx()
    level_axis.step(
        episodes,
        levels + 1,
        where="post",
        alpha=0.35,
        color="#8e5ea2",
        label="curriculum level",
    )
    level_axis.set_yticks((1, 2, 3))
    level_axis.set_ylabel("Training stage")
    axes[1, 1].set_title("Clearance and curriculum")
    axes[1, 1].set_ylabel("Clearance (m)")

    for axis in axes.flat:
        axis.set_xlabel("Completed episode")
        axis.grid(alpha=0.25)
        for index in range(1, len(rows)):
            if (levels[index] != levels[index - 1]
                    or rows[index].get("training_practice_phase") != rows[index - 1].get("training_practice_phase")):
                axis.axvline(episodes[index], color="#8559ab", linestyle=":", alpha=.65)
    return _finish_figure(figure, output_path)


def plot_ppo_diagnostics(
    rows: Sequence[dict[str, str]],
    output_path: Path,
) -> Path | None:
    """Plot losses, exploration, update size, and critic quality."""

    if not rows:
        return None
    steps = _series(rows, "total_timesteps")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    figure.suptitle("PPO Optimisation Diagnostics", fontsize=15)

    axes[0, 0].plot(
        steps,
        _series(rows, "policy_loss"),
        label="policy loss",
    )
    axes[0, 0].plot(
        steps,
        _series(rows, "value_loss"),
        label="value loss",
    )
    axes[0, 0].set_title("Actor and critic losses")
    axes[0, 0].legend()

    axes[0, 1].plot(
        steps,
        _series(rows, "entropy"),
        label="Gaussian entropy",
    )
    if any(row.get("squashed_entropy", "") != "" for row in rows):
        axes[0, 1].plot(
            steps, _series(rows, "squashed_entropy"),
            label="bounded-action entropy (estimate)",
        )
    axes[0, 1].plot(
        steps,
        _series(rows, "actor_log_std_linear"),
        label="linear log std",
    )
    axes[0, 1].plot(
        steps,
        _series(rows, "actor_log_std_angular"),
        label="angular log std",
    )
    axes[0, 1].set_title("Exploration")
    axes[0, 1].legend()

    axes[1, 0].plot(
        steps,
        _series(rows, "approximate_kl"),
        label="approximate KL",
    )
    axes[1, 0].plot(
        steps,
        _series(rows, "clip_fraction"),
        label="clip fraction",
    )
    axes[1, 0].set_title("PPO update size")
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps,
        _series(rows, "explained_variance"),
        label="explained variance",
    )
    gradient_axis = axes[1, 1].twinx()
    gradient_axis.plot(
        steps,
        _series(rows, "gradient_norm"),
        color="#d62728",
        alpha=0.65,
        label="gradient norm",
    )
    axes[1, 1].set_title("Critic and gradient health")
    axes[1, 1].set_ylabel("Explained variance")
    gradient_axis.set_ylabel("Gradient norm")

    for axis in axes.flat:
        axis.set_xlabel("Environment timesteps")
        axis.grid(alpha=0.25)
    return _finish_figure(figure, output_path)


def _group_evaluation_rows(
    rows: Sequence[dict[str, str]],
) -> dict[int, list[dict[str, str]]]:
    """Group evaluation rows by integer curriculum level."""

    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        level_value = _number(row, "curriculum_level")
        if math.isfinite(level_value):
            groups[int(level_value)].append(row)
    return dict(groups)


def plot_evaluation_summary(
    rows: Sequence[dict[str, str]],
    output_path: Path,
) -> Path | None:
    """Compare outcomes, return, length, and safety across levels."""

    groups = _group_evaluation_rows(rows)
    if not groups:
        return None
    levels = sorted(groups)
    x_positions = np.arange(len(levels), dtype=float)

    def group_mean(level: int, field: str) -> float:
        values = _series(groups[level], field)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")

    success_rates = [group_mean(level, "success") for level in levels]
    collision_rates = [group_mean(level, "collision") for level in levels]
    timeout_rates = [group_mean(level, "timeout") for level in levels]
    mean_returns = [group_mean(level, "return") for level in levels]
    mean_lengths = [group_mean(level, "length") for level in levels]
    mean_clearances = [
        group_mean(level, "minimum_clearance_m") for level in levels
    ]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    figure.suptitle("Fixed-Seed PPO Evaluation", fontsize=15)

    width = 0.25
    axes[0, 0].bar(
        x_positions - width,
        np.asarray(success_rates) * 100.0,
        width,
        label="success",
        color="#2ca02c",
    )
    axes[0, 0].bar(
        x_positions,
        np.asarray(collision_rates) * 100.0,
        width,
        label="collision",
        color="#d62728",
    )
    axes[0, 0].bar(
        x_positions + width,
        np.asarray(timeout_rates) * 100.0,
        width,
        label="timeout",
        color="#7f7f7f",
    )
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_ylabel("Episodes (%)")
    axes[0, 0].set_title("Outcome rates")
    axes[0, 0].legend()

    axes[0, 1].bar(
        x_positions,
        mean_returns,
        color=[LEVEL_COLOURS[level] for level in levels],
    )
    axes[0, 1].set_title("Mean episode return")

    axes[1, 0].bar(
        x_positions,
        mean_lengths,
        color=[LEVEL_COLOURS[level] for level in levels],
    )
    axes[1, 0].set_title("Mean episode length")
    axes[1, 0].set_ylabel("Environment steps")

    axes[1, 1].bar(
        x_positions,
        mean_clearances,
        color=[LEVEL_COLOURS[level] for level in levels],
    )
    axes[1, 1].set_title("Mean minimum clearance")
    axes[1, 1].set_ylabel("Clearance (m)")

    for axis in axes.flat:
        axis.set_xticks(x_positions, [_task_label(level, groups[level][0]) for level in levels])
        axis.grid(axis="y", alpha=0.25)
    return _finish_figure(figure, output_path)


def plot_evaluation_trajectories(
    rows: Sequence[dict[str, str]],
    output_path: Path,
) -> Path | None:
    """Plot recorded robot paths and their goals for each level."""

    if not rows:
        return None
    grouped: dict[tuple[int, int, int], list[dict[str, str]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (
            int(_number(row, "curriculum_level")),
            int(_number(row, "episode")),
            int(_number(row, "seed")),
        )
        grouped[key].append(row)
    levels = sorted({key[0] for key in grouped})
    figure, axes_array = plt.subplots(
        1,
        len(levels),
        figsize=(6 * len(levels), 5),
        squeeze=False,
    )
    figure.suptitle("Deterministic Evaluation Trajectories", fontsize=15)

    for axis, level in zip(axes_array[0], levels):
        level_items = [
            (key, values)
            for key, values in grouped.items()
            if key[0] == level
        ]
        for key, trajectory in sorted(level_items):
            trajectory.sort(key=lambda row: _number(row, "step"))
            x_values = _series(trajectory, "robot_x")
            y_values = _series(trajectory, "robot_y")
            goal_x = _number(trajectory[0], "goal_x")
            goal_y = _number(trajectory[0], "goal_y")
            axis.plot(
                x_values,
                y_values,
                linewidth=1.8,
                label=f"seed {key[2]}",
            )
            axis.scatter(
                x_values[0],
                y_values[0],
                marker="o",
                s=35,
                color="#1f77b4",
            )
            axis.scatter(
                goal_x,
                goal_y,
                marker="*",
                s=120,
                color="#2ca02c",
                edgecolor="black",
                linewidth=0.4,
            )
        axis.set_title(_task_label(level, next(row for row in rows if int(row['curriculum_level']) == level)))
        axis.set_xlabel("World x (m)")
        axis.set_ylabel("World y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    return _finish_figure(figure, output_path)


def draw_arena(axis, layout, start, goal, title, trajectory=None,
               region_center=(0, 0), region_size_m=9):
    h, t = ARENA_HALF_M, WALL_THICKNESS_M
    axis.set_facecolor('#f5f5f0')
    for x, y, w, height in ((-h-t, -h-t, 9+2*t, t), (-h-t, h, 9+2*t, t),
                             (-h-t, -h, t, 9), (h, -h, t, 9)):
        axis.add_patch(Rectangle((x, y), w, height, facecolor='#485363'))
    for i, pose in enumerate(layout.values()):
        x, y = pose[:2]
        axis.add_patch(Rectangle((x-0.5, y-0.5), BOX_SIZE_M, BOX_SIZE_M,
                                 facecolor=('#e88835', '#c9652b', '#eaaa59')[i % 3],
                                 edgecolor='#9f4d23', linewidth=0.7))
    if region_size_m < 9:
        axis.add_patch(Rectangle((region_center[0] - region_size_m / 2,
                                 region_center[1] - region_size_m / 2),
                                region_size_m, region_size_m, fill=False,
                                edgecolor='#8559ab', linestyle='--', linewidth=1.6))
    gx, gy = goal
    axis.add_patch(Circle((gx, gy), GOAL_RADIUS_M, facecolor='#46b76a', edgecolor='#238946',
                          linewidth=1.2, zorder=5, label='0.5 m goal'))
    sx, sy, yaw = start
    axis.add_patch(Circle((sx, sy), .22, facecolor='#237bb7', edgecolor='white', linewidth=1.2, zorder=5))
    import math
    axis.arrow(sx, sy, .45*math.cos(yaw), .45*math.sin(yaw), width=.035,
               head_width=.18, color='#0c4e80', zorder=6, length_includes_head=True)
    axis.plot([sx, gx], [sy, gy], linestyle='--', color='#495566', alpha=.55, linewidth=1, zorder=1)
    if trajectory:
        axis.plot([p[0] for p in trajectory], [p[1] for p in trajectory], color='#146fc1', linewidth=2.2, zorder=4)
    axis.set_xlim(-5, 5); axis.set_ylim(-5, 5)
    axis.set_aspect('equal'); axis.set_xticks([-4, -2, 0, 2, 4]); axis.set_yticks([-4, -2, 0, 2, 4])
    axis.set_xlabel('x (m)'); axis.set_ylabel('y (m)')
    axis.set_title(title, fontsize=11, pad=12, color='#253544')
    axis.grid(alpha=.12)
    for spine in axis.spines.values():
        spine.set_visible(False)



def plot_paths_with_obstacles(evaluation_directory, trajectory_rows, evaluation_rows, output_path):
    """Draw each recorded path over its own obstacles, start and goal."""
    data = json.loads((evaluation_directory / 'evaluation_layouts.json').read_text(encoding='utf-8'))
    paths = {}
    for row in trajectory_rows:
        paths.setdefault(int(row['seed']), []).append((float(row['robot_x']), float(row['robot_y'])))
    outcomes = {int(row['seed']): row['outcome'] for row in evaluation_rows}
    episodes = [e for e in data['episodes'] if e['seed'] in paths][:9]
    if not episodes:
        return None
    columns = min(3, len(episodes))
    rows = (len(episodes) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 5.9 * rows), squeeze=False)
    figure.patch.set_facecolor('white')
    figure.suptitle('Recorded evaluation | obstacle layouts and robot paths', fontsize=18,
                   fontweight='bold', color='#253544', y=.975)
    for axis, episode in zip(axes.flat, episodes):
        seed, level = episode['seed'], episode['curriculum_level']
        count = len(episode['object_layout'])
        title = f"Stage {level + 1} · {STAGES[level].size_m:.0f} m · {count} {'box' if count == 1 else 'boxes'} · seed {seed}"
        if episode.get('practice_phase', -1) >= 0:
            title += '\nDetour practice'
        if episode.get('task_kind') == 'corner':
            title += '\nCorner task' + (' | mirrored' if episode.get('mirrored') else '')
        if seed in outcomes:
            title += f" · {outcomes[seed]}"
        draw_arena(axis, episode['object_layout'], episode['start'], episode['goal'], title,
                   paths[seed], episode.get('region_center', (0, 0)), episode.get('region_size_m', 9))
    for axis in list(axes.flat)[len(episodes):]:
        axis.set_visible(False)
    figure.text(.5, .018,
                'Orange: obstacles   Blue: start / recorded path   Green: 0.5 m goal   Purple dashed: spawning region (no wall)',
                ha='center', fontsize=11, color='#475563')
    figure.subplots_adjust(top=.87, bottom=.12, wspace=.24, hspace=.32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor='white')
    plt.close(figure)
    return output_path


def _parse_arguments(args: Sequence[str] | None) -> argparse.Namespace:
    """Parse plot command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Plot PPO training and fixed-seed evaluation results.",
    )
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--evaluation-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    """Load available CSV files and create every applicable plot."""

    arguments = _parse_arguments(args)
    run_directory = Path(arguments.run_dir).expanduser().resolve()
    if not run_directory.is_dir():
        raise NotADirectoryError(f"Run directory not found: {run_directory}")
    evaluation_directory = (
        Path(arguments.evaluation_dir).expanduser().resolve()
        if arguments.evaluation_dir
        else run_directory / "evaluation"
    )
    output_directory = (
        Path(arguments.output_dir).expanduser().resolve()
        if arguments.output_dir
        else run_directory / "plots"
    )

    episode_rows = _read_csv(
        run_directory / "metrics/train_episodes.csv"
    )
    update_rows = _read_csv(
        run_directory / "metrics/train_updates.csv"
    )
    evaluation_rows = _read_csv(
        evaluation_directory / "evaluation_episodes.csv"
    )
    trajectory_rows = _read_csv(
        evaluation_directory / "evaluation_trajectories.csv"
    )

    created = [
        plot_training_curves(
            episode_rows,
            output_directory / "training_curves.png",
        ),
        plot_ppo_diagnostics(
            update_rows,
            output_directory / "ppo_diagnostics.png",
        ),
        plot_evaluation_summary(
            evaluation_rows,
            output_directory / "evaluation_summary.png",
        ),
        plot_evaluation_trajectories(
            trajectory_rows,
            output_directory / "evaluation_trajectories.png",
        ),
    ]
    created_paths = [path for path in created if path is not None]
    if (evaluation_directory / "evaluation_layouts.json").is_file() and trajectory_rows:
        path = plot_paths_with_obstacles(evaluation_directory, trajectory_rows, evaluation_rows,
                                         output_directory / "evaluation_paths_with_obstacles.png")
        if path is not None:
            created_paths.append(path)
    if not created_paths:
        raise RuntimeError(
            "No metrics were found. Train or evaluate the policy first."
        )
    for path in created_paths:
        print(f"Created plot: {path}")


if __name__ == "__main__":
    main()
