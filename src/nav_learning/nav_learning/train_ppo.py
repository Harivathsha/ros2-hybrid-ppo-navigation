"""Collect RosNavEnv rollouts and train the from-scratch PPO agent."""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from nav_learning.ppo import (
    ActorCritic,
    PPOAgent,
    RolloutBuffer,
    choose_device,
    load_checkpoint,
    observation_to_tensor,
    save_checkpoint,
)
from nav_learning.rl_config import (
    TrainingSettings,
    find_default_config,
    load_training_settings,
)


EPISODE_FIELDS = (
    "episode",
    "total_timesteps",
    "seed",
    "curriculum_level",
    "length",
    "return",
    "outcome",
    "success",
    "collision",
    "timeout",
    "initial_distance_m",
    "final_distance_m",
    "minimum_clearance_m",
    "start_room",
    "goal_room",
    "reset_attempt",
)

UPDATE_FIELDS = (
    "update",
    "total_timesteps",
    "curriculum_level",
    "rollout_size",
    "episodes_completed",
    "steps_per_second",
    "policy_loss",
    "value_loss",
    "entropy",
    "total_loss",
    "approximate_kl",
    "clip_fraction",
    "gradient_norm",
    "explained_variance",
    "optimiser_steps",
    "epochs_completed",
    "early_stopped",
    "learning_rate",
    "actor_log_std_linear",
    "actor_log_std_angular",
    "raw_advantage_mean",
    "raw_advantage_std",
    "final_advantage_mean",
    "final_advantage_std",
    "recent_success_rate",
)


class CsvLogger:
    """Append rows while writing the header only once."""

    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        self._stream = path.open(
            "a",
            encoding="utf-8",
            newline="",
        )
        self._fields = tuple(fields)
        self._writer = csv.DictWriter(
            self._stream,
            fieldnames=self._fields,
        )
        if needs_header:
            self._writer.writeheader()
            self._stream.flush()

    def write(self, values: Mapping[str, Any]) -> None:
        """Append one filtered row and flush it immediately."""

        row = {name: values.get(name, "") for name in self._fields}
        self._writer.writerow(row)
        self._stream.flush()

    def close(self) -> None:
        """Close the underlying CSV stream."""

        self._stream.close()


def _set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and available CUDA devices."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_random_states() -> dict[str, Any]:
    """Capture random-number states for a resumable checkpoint."""

    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def _restore_random_states(states: Mapping[str, Any]) -> None:
    """Restore all random-number states present in a checkpoint."""

    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"].cpu())
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in states["torch_cuda"]]
        )


def _utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_run_directory(
    settings: TrainingSettings,
    resume_path: Path | None,
    explicit_run_directory: str | None,
) -> Path:
    """Create a new run directory or infer the resumed run directory."""

    if explicit_run_directory:
        run_directory = Path(explicit_run_directory).expanduser()
    elif resume_path is not None:
        if resume_path.parent.name == "checkpoints":
            run_directory = resume_path.parent.parent
        else:
            run_directory = resume_path.parent
    else:
        run_name = f"run_{_utc_timestamp()}_seed{settings.seed}"
        run_directory = Path(settings.output_root) / run_name
    run_directory = run_directory.resolve()
    (run_directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_directory / "metrics").mkdir(parents=True, exist_ok=True)
    return run_directory


def _new_training_state(settings: TrainingSettings) -> dict[str, Any]:
    """Create counters and curriculum state for a fresh run."""

    return {
        "update_index": 0,
        "total_timesteps": 0,
        "episode_index": 0,
        "next_reset_seed": settings.reset_seed_base,
        "curriculum_level": settings.curriculum.start_level,
        "recent_successes": [],
        "recent_episode_returns": [],
    }


def _validate_resumed_state(
    raw_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate essential counters before resuming environment work."""

    state = dict(raw_state)
    required = {
        "update_index",
        "total_timesteps",
        "episode_index",
        "next_reset_seed",
        "curriculum_level",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(
            "Checkpoint training state is missing: " + ", ".join(missing)
        )
    for name in required:
        if isinstance(state[name], bool) or not isinstance(state[name], int):
            raise TypeError(f"Checkpoint state {name} must be an integer.")
        if state[name] < 0:
            raise ValueError(f"Checkpoint state {name} is negative.")
    if state["curriculum_level"] not in (0, 1, 2):
        raise ValueError("Checkpoint curriculum level is invalid.")
    state.setdefault("recent_successes", [])
    state.setdefault("recent_episode_returns", [])
    return state


def _checkpoint_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Add current RNG states without mutating live training state."""

    checkpoint_state = dict(state)
    checkpoint_state["random_states"] = _capture_random_states()
    return checkpoint_state


def _recent_mean(values: Sequence[float]) -> float:
    """Return a finite mean or NaN for an empty sequence."""

    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _update_curriculum_after_episode(
    state: dict[str, Any],
    settings: TrainingSettings,
    success: bool,
    fixed_level: int | None,
) -> tuple[float, bool]:
    """Record success and promote only after stable performance."""

    if fixed_level is not None:
        state["curriculum_level"] = fixed_level
        return float(success), False
    curriculum = settings.curriculum
    if not curriculum.enabled:
        return float(success), False

    recent = list(state["recent_successes"])
    recent.append(int(success))
    recent = recent[-curriculum.promotion_window:]
    state["recent_successes"] = recent
    success_rate = _recent_mean(recent)
    enough_episodes = len(recent) >= curriculum.minimum_episodes
    can_promote = state["curriculum_level"] < curriculum.max_level
    promoted = (
        enough_episodes
        and can_promote
        and success_rate >= curriculum.success_threshold
    )
    if promoted:
        state["curriculum_level"] += 1
        state["recent_successes"] = []
    return success_rate, promoted


def _reset_episode(
    environment: Any,
    state: dict[str, Any],
    fixed_level: int | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Consume one deterministic reset seed and initialise accumulators."""

    level = (
        fixed_level
        if fixed_level is not None
        else int(state["curriculum_level"])
    )
    reset_seed = int(state["next_reset_seed"])
    state["next_reset_seed"] = reset_seed + 1
    observation, reset_info = environment.reset(
        seed=reset_seed,
        curriculum_level=level,
    )
    episode = {
        "seed": reset_seed,
        "level": level,
        "length": 0,
        "return": 0.0,
        "minimum_clearance_m": float("inf"),
        "initial_distance_m": float(reset_info["distance_to_goal_m"]),
        "start_room": reset_info.get("start_room", ""),
        "goal_room": reset_info.get("goal_room", ""),
        "reset_attempt": reset_info.get("reset_attempt", ""),
    }
    return observation, reset_info, episode


def _episode_row(
    state: Mapping[str, Any],
    episode: Mapping[str, Any],
    final_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert final episode state to one CSV row."""

    outcome = str(final_info["outcome"])
    minimum_clearance = float(episode["minimum_clearance_m"])
    if not math.isfinite(minimum_clearance):
        minimum_clearance = float(final_info["nearest_obstacle_m"])
    return {
        "episode": int(state["episode_index"]),
        "total_timesteps": int(state["total_timesteps"]),
        "seed": int(episode["seed"]),
        "curriculum_level": int(episode["level"]),
        "length": int(episode["length"]),
        "return": float(episode["return"]),
        "outcome": outcome,
        "success": int(outcome == "success"),
        "collision": int(outcome == "collision"),
        "timeout": int(outcome == "timeout"),
        "initial_distance_m": float(episode["initial_distance_m"]),
        "final_distance_m": float(final_info["distance_to_goal_m"]),
        "minimum_clearance_m": minimum_clearance,
        "start_room": episode["start_room"],
        "goal_room": episode["goal_room"],
        "reset_attempt": episode["reset_attempt"],
    }


def _save_training_checkpoint(
    run_directory: Path,
    filename: str,
    agent: PPOAgent,
    state: Mapping[str, Any],
    config_path: Path,
) -> Path:
    """Save one named training checkpoint."""

    metadata = {
        "run_directory": str(run_directory),
        "config_path": str(config_path),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return save_checkpoint(
        run_directory / "checkpoints" / filename,
        agent=agent,
        training_state=_checkpoint_state(state),
        metadata=metadata,
    )


def _apply_overrides(
    settings: TrainingSettings,
    arguments: argparse.Namespace,
) -> TrainingSettings:
    """Apply explicit command-line overrides to immutable YAML settings."""

    replacements: dict[str, Any] = {}
    for argument_name, setting_name in (
        ("total_timesteps", "total_timesteps"),
        ("rollout_steps", "rollout_steps"),
        ("device", "device"),
        ("seed", "seed"),
    ):
        value = getattr(arguments, argument_name)
        if value is not None:
            replacements[setting_name] = value
    if arguments.smoke:
        replacements.update(
            {
                "total_timesteps": 256,
                "rollout_steps": 128,
                "checkpoint_every_updates": 1,
                "log_every_updates": 1,
            }
        )
    if replacements:
        settings = replace(settings, **replacements)
    # Re-run full validation after dataclass replacement.
    values = {
        **settings.__dict__,
        "curriculum": settings.curriculum.__dict__,
        "ppo": settings.ppo.__dict__,
    }
    return TrainingSettings.from_mapping(values)


def _parse_arguments(args: Sequence[str] | None) -> argparse.Namespace:
    """Parse trainer command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train PPO with the existing RosNavEnv.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--fixed-level",
        type=int,
        choices=(0, 1, 2),
        default=None,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only 256 environment steps at curriculum level 0.",
    )
    return parser.parse_args(args)


def run_training(
    settings: TrainingSettings,
    config_path: Path,
    run_directory: Path,
    resume_path: Path | None,
    fixed_level: int | None,
) -> None:
    """Run collection, GAE, PPO updates, logging, and checkpointing."""

    import rclpy

    from nav_learning.ros_nav_env import RosNavEnv

    device = choose_device(settings.device)
    _set_global_seeds(settings.seed)

    if resume_path is None:
        agent = PPOAgent(
            ActorCritic().to(device),
            settings.ppo,
        )
        state = _new_training_state(settings)
    else:
        agent, payload = load_checkpoint(
            resume_path,
            device=device,
            load_optimizer=True,
        )
        state = _validate_resumed_state(payload["training_state"])
        random_states = payload["training_state"].get("random_states")
        if isinstance(random_states, Mapping):
            _restore_random_states(random_states)

    if fixed_level is not None:
        state["curriculum_level"] = fixed_level
        state["recent_successes"] = []
    if int(state["total_timesteps"]) >= settings.total_timesteps:
        raise ValueError(
            "The checkpoint already reached total_timesteps. "
            "Pass a larger --total-timesteps value."
        )

    episode_logger = CsvLogger(
        run_directory / "metrics/train_episodes.csv",
        EPISODE_FIELDS,
    )
    update_logger = CsvLogger(
        run_directory / "metrics/train_updates.csv",
        UPDATE_FIELDS,
    )
    environment = None
    rclpy.init(args=None)

    print(f"Run directory: {run_directory}")
    print(f"Training device: {device}")
    print(
        "Starting from steps="
        f"{state['total_timesteps']}, update={state['update_index']}, "
        f"level={state['curriculum_level']}"
    )

    try:
        environment = RosNavEnv()
        observation, _, episode = _reset_episode(
            environment,
            state,
            fixed_level,
        )

        while int(state["total_timesteps"]) < settings.total_timesteps:
            update_started = time.perf_counter()
            remaining = (
                settings.total_timesteps - int(state["total_timesteps"])
            )
            rollout_size = min(settings.rollout_steps, remaining)
            rollout = RolloutBuffer(rollout_size, device=device)
            agent.model.eval()

            for _ in range(rollout_size):
                observation_tensor = observation_to_tensor(
                    observation,
                    device,
                )
                action, log_probability, value = (
                    agent.model.sample_action(observation_tensor)
                )
                action_array = (
                    action.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = environment.step(action_array)
                next_observation_tensor = observation_to_tensor(
                    next_observation,
                    device,
                )
                with torch.no_grad():
                    next_value = agent.model.predict_value(
                        next_observation_tensor
                    )
                rollout.add(
                    observation=observation_tensor,
                    action=action,
                    reward=reward,
                    value=value,
                    next_value=next_value,
                    log_probability=log_probability,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )

                state["total_timesteps"] += 1
                episode["length"] += 1
                episode["return"] += float(reward)
                episode["minimum_clearance_m"] = min(
                    float(episode["minimum_clearance_m"]),
                    float(info["nearest_obstacle_m"]),
                )
                observation = next_observation

                if terminated or truncated:
                    state["episode_index"] += 1
                    row = _episode_row(state, episode, info)
                    episode_logger.write(row)
                    recent_returns = list(
                        state["recent_episode_returns"]
                    )
                    recent_returns.append(float(episode["return"]))
                    state["recent_episode_returns"] = recent_returns[-20:]
                    success_rate, promoted = (
                        _update_curriculum_after_episode(
                            state,
                            settings,
                            success=info["outcome"] == "success",
                            fixed_level=fixed_level,
                        )
                    )
                    print(
                        f"episode={state['episode_index']} "
                        f"level={episode['level']} "
                        f"outcome={info['outcome']} "
                        f"steps={episode['length']} "
                        f"return={episode['return']:.3f} "
                        f"clearance="
                        f"{float(row['minimum_clearance_m']):.3f} "
                        f"route={info.get('route_waypoint_index', 0)}/"
                        f"{info.get('route_waypoint_count', 0)} "
                        f"stuck={int(bool(info.get('stuck', False)))} "
                        f"success_rate={success_rate:.2f}"
                    )
                    if promoted:
                        print(
                            "Curriculum promoted to level "
                            f"{state['curriculum_level']}."
                        )
                    if (
                        int(state["total_timesteps"])
                        < settings.total_timesteps
                    ):
                        observation, _, episode = _reset_episode(
                            environment,
                            state,
                            fixed_level,
                        )

            gae_metrics = rollout.compute_returns_and_advantages(
                discount_factor=agent.hyperparameters.discount_factor,
                gae_lambda=agent.hyperparameters.gae_lambda,
                normalise_advantages=True,
            )
            if settings.anneal_learning_rate:
                progress_remaining = max(
                    1.0
                    - int(state["total_timesteps"])
                    / settings.total_timesteps,
                    0.01,
                )
                agent.set_learning_rate(
                    agent.hyperparameters.learning_rate
                    * progress_remaining
                )
            ppo_metrics = agent.update(rollout)
            state["update_index"] += 1

            elapsed = max(time.perf_counter() - update_started, 1e-9)
            recent_successes = list(state["recent_successes"])
            recent_success_rate = _recent_mean(recent_successes)
            update_row = {
                "update": int(state["update_index"]),
                "total_timesteps": int(state["total_timesteps"]),
                "curriculum_level": int(state["curriculum_level"]),
                "rollout_size": rollout_size,
                "episodes_completed": int(state["episode_index"]),
                "steps_per_second": rollout_size / elapsed,
                "recent_success_rate": recent_success_rate,
                **ppo_metrics,
                **gae_metrics,
            }
            update_logger.write(update_row)

            if (
                int(state["update_index"])
                % settings.log_every_updates
                == 0
            ):
                mean_return = _recent_mean(
                    state["recent_episode_returns"]
                )
                print(
                    f"update={state['update_index']} "
                    f"steps={state['total_timesteps']}/"
                    f"{settings.total_timesteps} "
                    f"level={state['curriculum_level']} "
                    f"mean_return_20={mean_return:.3f} "
                    f"policy_loss={ppo_metrics['policy_loss']:.4f} "
                    f"value_loss={ppo_metrics['value_loss']:.4f} "
                    f"kl={ppo_metrics['approximate_kl']:.6f}"
                )

            if (
                int(state["update_index"])
                % settings.checkpoint_every_updates
                == 0
            ):
                numbered_name = (
                    f"update_{int(state['update_index']):06d}.pt"
                )
                _save_training_checkpoint(
                    run_directory,
                    numbered_name,
                    agent,
                    state,
                    config_path,
                )
                latest_path = _save_training_checkpoint(
                    run_directory,
                    "latest.pt",
                    agent,
                    state,
                    config_path,
                )
                print(f"Checkpoint: {latest_path}")

        final_path = _save_training_checkpoint(
            run_directory,
            "final.pt",
            agent,
            state,
            config_path,
        )
        _save_training_checkpoint(
            run_directory,
            "latest.pt",
            agent,
            state,
            config_path,
        )
        print(f"Training complete. Final checkpoint: {final_path}")

    except KeyboardInterrupt:
        interrupted_path = _save_training_checkpoint(
            run_directory,
            "interrupted.pt",
            agent,
            state,
            config_path,
        )
        print(f"Training interrupted safely: {interrupted_path}")
    except Exception:
        crash_path = _save_training_checkpoint(
            run_directory,
            "crash_recovery.pt",
            agent,
            state,
            config_path,
        )
        print(f"Crash-recovery checkpoint: {crash_path}")
        raise
    finally:
        if environment is not None:
            environment.close()
        episode_logger.close()
        update_logger.close()
        if rclpy.ok():
            rclpy.shutdown()


def main(args: Sequence[str] | None = None) -> None:
    """Load settings and start PPO training."""

    arguments = _parse_arguments(args)
    config_path = (
        Path(arguments.config).expanduser().resolve()
        if arguments.config
        else find_default_config()
    )
    settings = _apply_overrides(
        load_training_settings(config_path),
        arguments,
    )
    resume_path = (
        Path(arguments.resume).expanduser().resolve()
        if arguments.resume
        else None
    )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    if resume_path is not None and arguments.seed is not None:
        raise ValueError("Do not override --seed when resuming a run.")
    fixed_level = 0 if arguments.smoke else arguments.fixed_level
    run_directory = _resolve_run_directory(
        settings,
        resume_path,
        arguments.run_dir,
    )
    copied_config = run_directory / "config_used.yaml"
    if not copied_config.exists():
        shutil.copy2(config_path, copied_config)
    run_training(
        settings=settings,
        config_path=config_path,
        run_directory=run_directory,
        resume_path=resume_path,
        fixed_level=fixed_level,
    )


if __name__ == "__main__":
    main()
