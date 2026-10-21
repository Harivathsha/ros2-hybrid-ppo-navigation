"""Collect RosNavEnv rollouts and train the from-scratch PPO agent."""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from nav_learning.nav_math import (
    ENVIRONMENT_ID, REHEARSAL_PROBABILITY, STAGES, RewardSettings, DETOUR_PHASE_NAMES,
    ensure_checkpoint_environment, ActionSettings, wrap_to_pi,
)

from nav_learning.ppo import (
    ActorCritic,
    PPOAgent,
    RolloutBuffer,
    PPOHyperparameters,
    choose_device,
    load_checkpoint,
    observation_to_tensor,
    save_checkpoint,
)


# ---------------- Editable YAML settings and validation ----------------

TRAINING_REVISION = "corner_detour_rewards_v1"
CONTROL_MODE = "gazebo_run_to_sim_time_v1"

def _reject_unknown_keys(
    values: Mapping[str, Any],
    data_class: type,
    section_name: str,
) -> None:
    """Fail early when a configuration key was misspelled."""

    known = {field.name for field in fields(data_class)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            f"Unknown {section_name} settings: " + ", ".join(unknown)
        )


@dataclass(frozen=True)
class CurriculumSettings:
    """Performance-based level-promotion settings."""

    enabled: bool = True
    start_level: int = 0
    max_level: int = 2
    promotion_window: int = 100
    minimum_episodes: int = 100
    success_threshold: float = 0.80
    validation_episodes: int = 10
    validation_success_threshold: float = 0.80
    validation_retry_episodes: int = 25
    detour_practice: bool = False  # Old run configs keep their original distribution.
    detour_success_threshold: float = 0.90
    corner_practice_probability: float = 0.0
    mirror_training: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "CurriculumSettings":
        """Build and validate curriculum settings."""

        _reject_unknown_keys(values, cls, "curriculum")
        settings = cls(**dict(values))
        if not isinstance(settings.enabled, bool):
            raise TypeError("curriculum.enabled must be a bool.")
        if not isinstance(settings.detour_practice, bool):
            raise TypeError("curriculum.detour_practice must be a bool.")
        if not isinstance(settings.mirror_training, bool):
            raise TypeError("curriculum.mirror_training must be a bool.")
        if not 0 <= settings.corner_practice_probability <= 1:
            raise ValueError("curriculum.corner_practice_probability must be in [0, 1].")
        if (not math.isfinite(settings.detour_success_threshold)
                or not 0 < settings.detour_success_threshold <= 1):
            raise ValueError("detour_success_threshold must be in (0, 1].")
        for name in ("start_level", "max_level"):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"curriculum.{name} must be an integer.")
            if value not in (0, 1, 2):
                raise ValueError(f"curriculum.{name} must be 0, 1, or 2.")
        if settings.start_level > settings.max_level:
            raise ValueError("start_level cannot exceed max_level.")
        for name in ("promotion_window", "minimum_episodes", "validation_episodes", "validation_retry_episodes"):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"curriculum.{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"curriculum.{name} must be positive.")
        if settings.minimum_episodes > settings.promotion_window:
            raise ValueError(
                "minimum_episodes cannot exceed promotion_window."
            )
        if (
            not math.isfinite(settings.success_threshold)
            or not 0.0 <= settings.success_threshold <= 1.0
        ):
            raise ValueError("success_threshold must be in [0, 1].")
        if not math.isfinite(settings.validation_success_threshold) or not 0 <= settings.validation_success_threshold <= 1:
            raise ValueError("validation_success_threshold must be in [0, 1].")
        return settings


@dataclass(frozen=True)
class TrainingSettings:
    """Environment collection, output, and optimiser settings."""

    total_timesteps: int
    rollout_steps: int
    seed: int
    reset_seed_base: int
    device: str
    output_root: str
    checkpoint_every_updates: int
    log_every_updates: int
    anneal_learning_rate: bool
    curriculum: CurriculumSettings
    ppo: PPOHyperparameters
    reward: RewardSettings = RewardSettings()
    action: ActionSettings = ActionSettings()

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "TrainingSettings":
        """Build the complete settings object from parsed YAML."""

        _reject_unknown_keys(values, cls, "training")
        required = {
            "total_timesteps",
            "rollout_steps",
            "seed",
            "reset_seed_base",
            "device",
            "output_root",
            "checkpoint_every_updates",
            "log_every_updates",
            "anneal_learning_rate",
            "curriculum",
            "ppo",
        }
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(
                "Missing training settings: " + ", ".join(missing)
            )

        curriculum_values = values["curriculum"]
        ppo_values = values["ppo"]
        if not isinstance(curriculum_values, Mapping):
            raise TypeError("curriculum must be a mapping.")
        if not isinstance(ppo_values, Mapping):
            raise TypeError("ppo must be a mapping.")

        plain_values = dict(values)
        plain_values["curriculum"] = CurriculumSettings.from_mapping(
            curriculum_values
        )
        plain_values["ppo"] = PPOHyperparameters.from_mapping(ppo_values)
        plain_values["reward"] = RewardSettings.from_mapping(values.get("reward", {}))
        plain_values["action"] = ActionSettings.from_mapping(values.get("action", {}))
        settings = cls(**plain_values)
        if (settings.reward.progress_mode == "detour_potential" and
                settings.reward.discount_factor != settings.ppo.discount_factor):
            raise ValueError("Reward potential and PPO must use the same discount_factor.")

        for name in (
            "total_timesteps",
            "rollout_steps",
            "checkpoint_every_updates",
            "log_every_updates",
        ):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("seed", "reset_seed_base"):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if settings.rollout_steps > settings.total_timesteps:
            raise ValueError(
                "rollout_steps cannot exceed total_timesteps."
            )
        if settings.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda.")
        if not settings.output_root.strip():
            raise ValueError("output_root cannot be empty.")
        if not isinstance(settings.anneal_learning_rate, bool):
            raise TypeError("anneal_learning_rate must be a bool.")
        return settings


def find_default_config() -> Path:
    """Locate the YAML file from source or an installed ROS package."""

    candidates = [
        Path.cwd() / "src/nav_learning/config/ppo_training.yaml",
        Path(__file__).resolve().parents[1]
        / "config/ppo_training.yaml",
    ]
    try:
        from ament_index_python.packages import (
            get_package_share_directory,
        )

        candidates.append(
            Path(get_package_share_directory("nav_learning"))
            / "config/ppo_training.yaml"
        )
    except (ImportError, LookupError):
        pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate config/ppo_training.yaml. "
        "Pass it explicitly with --config."
    )


def load_training_settings(path: str | Path) -> TrainingSettings:
    """Parse a YAML file and return validated immutable settings."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, Mapping):
        raise TypeError("The YAML root must be a mapping.")
    return TrainingSettings.from_mapping(values)


# ---------------- Training, logging and checkpointing ----------------

EPISODE_FIELDS = (
    "task_kind", "mirrored",
    "episode",
    "total_timesteps",
    "seed",
    "curriculum_level",
    "training_stage",
    "rehearsal",
    "practice_phase",
    "training_practice_phase",
    "promotion_threshold",
    "region_size_m",
    "length",
    "return",
    "outcome",
    "success",
    "collision",
    "timeout",
    "initial_distance_m",
    "final_distance_m",
    "minimum_clearance_m",
    "straight_path_blocked",
    "straight_path_blocker_count",
    "randomized_object_count",
    "reset_attempt",
    "mean_control_interval_sec",
    "maximum_sensor_age_sec",
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
    "squashed_entropy",
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
    "validation_steps",
    "detour_phase",
)

VALIDATION_FIELDS = ("total_timesteps", "curriculum_level", "practice_phase", "attempt", "seed",
                     "outcome", "success", "length", "return")


class CsvLogger:
    """Append rows while writing the header only once."""

    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        self._fields = tuple(fields)
        if not needs_header:
            # Preserve older logs' column order when a new diagnostic is added.
            with path.open(encoding="utf-8", newline="") as stream:
                existing_fields = tuple(next(csv.reader(stream)))
            if not existing_fields or not set(existing_fields).issubset(fields):
                raise ValueError(f"Incompatible CSV header: {path}. Use a new --run-dir.")
            self._fields = existing_fields
        self._stream = path.open(
            "a",
            encoding="utf-8",
            newline="",
        )
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
    if resume_path is None and run_directory.exists() and (
        (run_directory / "metrics/train_episodes.csv").exists()
        or any((run_directory / "checkpoints").glob("*.pt"))
    ):
        raise FileExistsError("This run already contains training results. Use a new --run-dir or --resume.")
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
        "promotion_wait_remaining": 0,
        "validation_attempts": 0,
        "validation_steps": 0,
        "detour_phase": 0,
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
    for name in ("promotion_wait_remaining", "validation_attempts", "validation_steps"):
        state.setdefault(name, 0)
    state.setdefault("detour_phase", 0)
    if isinstance(state["detour_phase"], bool) or state["detour_phase"] not in (0, 1, 2, 3):
        raise ValueError("Checkpoint detour_phase must be 0, 1, 2 or 3.")
    return state


def _checkpoint_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Add current RNG states without mutating live training state."""

    checkpoint_state = dict(state)
    checkpoint_state["random_states"] = _capture_random_states()
    return checkpoint_state


def _resume_training_agent(
    settings: TrainingSettings,
    resume_path: Path,
    device: torch.device,
    reset_exploration: bool = False,
) -> tuple[PPOAgent, dict[str, Any]]:
    """Restore learned weights/counters, then apply the selected tuning profile."""

    agent, payload = load_checkpoint(resume_path, device=device, load_optimizer=True)
    metadata = payload.get("metadata", {})
    ensure_checkpoint_environment(metadata)
    state = _validate_resumed_state(payload["training_state"])
    old_reward = RewardSettings.from_mapping(metadata.get("reward_settings", {}))
    old_action = ActionSettings.from_mapping(metadata.get("action_settings", {}))
    # A promotion-threshold change keeps the same task distribution.
    old_curriculum_for_window = dict(metadata.get("curriculum_settings") or {})
    new_curriculum_for_window = asdict(settings.curriculum)
    old_curriculum_for_window.pop("detour_success_threshold", None)
    new_curriculum_for_window.pop("detour_success_threshold", None)
    changed = (
        metadata.get("training_revision") != TRAINING_REVISION
        or agent.hyperparameters != settings.ppo
        or old_reward != settings.reward
        or old_action != settings.action
        or reset_exploration
        or metadata.get("detour_practice", False) != settings.curriculum.detour_practice
        or old_curriculum_for_window != new_curriculum_for_window
    )
    agent.set_hyperparameters(settings.ppo)
    if reset_exploration:
        agent.reset_exploration()
        print("Reset sampling log std to -0.5; learned mean and critic preserved.")
    if changed:
        state["recent_successes"] = []
        state["recent_episode_returns"] = []
        state["promotion_wait_remaining"] = 0
        print("Training settings changed: collecting a fresh promotion window at the current stage.")
    random_states = payload["training_state"].get("random_states")
    if isinstance(random_states, Mapping):
        _restore_random_states(random_states)
    state.setdefault("source_checkpoint", str(resume_path))
    return agent, state


def _practice_phase(state: Mapping[str, Any], settings: TrainingSettings,
                    level: int) -> int | None:
    if (level == 1 and getattr(settings.curriculum, "detour_practice", False)
            and int(state.get("detour_phase", 0)) < 3):
        return int(state.get("detour_phase", 0))
    return None


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
    count_for_promotion: bool = True,
) -> tuple[float, bool]:
    """Record current-stage success and signal readiness for validation.

    Rehearsal episodes train the policy but cannot promote the harder stage.
    """

    curriculum = settings.curriculum
    recent = list(state["recent_successes"])
    if not count_for_promotion:
        return _recent_mean(recent), False
    recent.append(int(success))
    recent = recent[-curriculum.promotion_window:]
    state["recent_successes"] = recent
    success_rate = _recent_mean(recent)
    # Keep reporting the rolling success rate even with the curriculum disabled.
    if fixed_level is not None:
        state["curriculum_level"] = fixed_level
        return success_rate, False
    if not curriculum.enabled:
        return success_rate, False

    enough_episodes = len(recent) >= curriculum.minimum_episodes
    can_promote = (state["curriculum_level"] < curriculum.max_level
                   or _practice_phase(state, settings, int(state["curriculum_level"])) is not None)
    wait = max(0, state.get("promotion_wait_remaining", 0) - 1)
    state["promotion_wait_remaining"] = wait
    ready = (
        enough_episodes
        and can_promote
        and success_rate >= (curriculum.detour_success_threshold
                             if _practice_phase(state, settings, int(state["curriculum_level"])) is not None
                             else curriculum.success_threshold)
        and wait == 0
    )
    return success_rate, ready


def _complete_promotion_validation(state: dict[str, Any], settings: TrainingSettings,
                                   success_rate: float) -> bool:
    """Keep learning at this stage unless its independent check passes."""
    practice_phase = _practice_phase(state, settings, int(state["curriculum_level"]))
    threshold = (settings.curriculum.detour_success_threshold if practice_phase is not None
                 else settings.curriculum.validation_success_threshold)
    if success_rate < threshold:
        state["promotion_wait_remaining"] = settings.curriculum.validation_retry_episodes
        return False
    if practice_phase is not None:
        state["detour_phase"] = practice_phase + 1
    else:
        state["curriculum_level"] += 1
    state["recent_successes"] = []
    state["promotion_wait_remaining"] = 0
    return True


def _run_promotion_validation(environment: Any, agent: PPOAgent, device: Any,
                              level: int, count: int, seed_base: int,
                              practice_phase: int | None = None,
                              corner_probability: float = 0.0,
                              mirror_pair: bool = False) -> list[dict[str, Any]]:
    """Run deterministic episodes without PPO updates or training-step counts."""
    rows = []
    agent.model.eval()
    try:
        for index in range(count):
            seed = seed_base + index
            reset_options = {} if practice_phase is None else {"practice_phase": practice_phase}
            if corner_probability or mirror_pair:
                reset_options.update(corner_probability=corner_probability, mirror_pair=mirror_pair)
            observation, _ = environment.reset(seed=seed, curriculum_level=level, **reset_options)
            total_return = 0.0
            for length in range(1, STAGES[level].max_steps + 1):
                with torch.inference_mode():
                    action = agent.model.deterministic_action(observation_to_tensor(observation, device))
                observation, reward, terminated, truncated, info = environment.step(
                    action.detach().cpu().numpy().astype(np.float32, copy=False))
                total_return += float(reward)
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Validation exceeded the stage episode limit.")
            rows.append(dict(seed=seed, outcome=info["outcome"],
                             success=int(info["outcome"] == "success"),
                             length=length, return_=total_return))
            print(f"  validation {index + 1}/{count}: {info['outcome']} ({length} steps)", flush=True)
    finally:
        environment.publish_stop()
    return rows


def _reset_episode(
    environment: Any,
    state: dict[str, Any],
    fixed_level: int | None,
    settings: TrainingSettings | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Consume one deterministic reset seed and initialise accumulators."""

    level = (
        fixed_level
        if fixed_level is not None
        else int(state["curriculum_level"])
    )
    reset_seed = int(state["next_reset_seed"])
    state["next_reset_seed"] = reset_seed + 1
    training_stage = level
    # Pure stage distributions remain available to the evaluator. Only training
    # mixes in previously learned tasks, deterministically from the reset seed.
    if fixed_level is None and level > 0:
        paired_seed = (reset_seed - reset_seed % 2 if settings is not None
                       and settings.curriculum.mirror_training else reset_seed)
        rng = np.random.default_rng(paired_seed)
        if rng.random() < REHEARSAL_PROBABILITY:
            level = int(rng.integers(0, level))
    practice_phase = (_practice_phase(state, settings, level)
                      if settings is not None and fixed_level is None else None)
    reset_options = {} if practice_phase is None else {"practice_phase": practice_phase}
    if settings is not None and fixed_level is None:
        if settings.curriculum.corner_practice_probability or settings.curriculum.mirror_training:
            reset_options.update(corner_probability=settings.curriculum.corner_practice_probability,
                                 mirror_pair=settings.curriculum.mirror_training)
    observation, reset_info = environment.reset(
        seed=reset_seed,
        curriculum_level=level,
        **reset_options,
    )
    episode = {
        "seed": reset_seed,
        "level": level,
        "training_stage": training_stage,
        "rehearsal": int(level < training_stage),
        "region_size_m": reset_info.get("region_size_m", STAGES[level].size_m),
        "length": 0,
        "return": 0.0,
        "minimum_clearance_m": float("inf"),
        "initial_distance_m": float(reset_info["distance_to_goal_m"]),
        "straight_path_blocked": int(reset_info["straight_path_blocked"]),
        "straight_path_blocker_count": int(reset_info["straight_path_blocker_count"]),
        "randomized_object_count": reset_info["randomized_object_count"],
        "reset_attempt": reset_info.get("reset_attempt", ""),
        "practice_phase": -1 if practice_phase is None else practice_phase,
        "training_practice_phase": int(state.get("detour_phase", 0)),
        "task_kind": reset_info.get("task_kind", "random"),
        "mirrored": int(reset_info.get("mirrored", False)),
        "promotion_threshold": (
            settings.curriculum.detour_success_threshold
            if settings is not None and _practice_phase(state, settings, training_stage) is not None
            else settings.curriculum.success_threshold if settings is not None else 0.80),
        "control_interval_sum": 0.0,
        "maximum_sensor_age_sec": 0.0,
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
        "task_kind": episode["task_kind"], "mirrored": episode["mirrored"],
        "total_timesteps": int(state["total_timesteps"]),
        "seed": int(episode["seed"]),
        "curriculum_level": int(episode["level"]),
        "training_stage": int(episode["training_stage"]),
        "rehearsal": episode["rehearsal"],
        "region_size_m": episode["region_size_m"],
        "length": int(episode["length"]),
        "return": float(episode["return"]),
        "outcome": outcome,
        "success": int(outcome == "success"),
        "collision": int(outcome == "collision"),
        "timeout": int(outcome == "timeout"),
        "initial_distance_m": float(episode["initial_distance_m"]),
        "final_distance_m": float(final_info["distance_to_goal_m"]),
        "minimum_clearance_m": minimum_clearance,
        "straight_path_blocked": episode["straight_path_blocked"],
        "straight_path_blocker_count": episode["straight_path_blocker_count"],
        "randomized_object_count": episode["randomized_object_count"],
        "reset_attempt": episode["reset_attempt"],
        "practice_phase": episode["practice_phase"],
        "training_practice_phase": episode["training_practice_phase"],
        "promotion_threshold": episode["promotion_threshold"],
        "mean_control_interval_sec": episode["control_interval_sum"] / max(1, episode["length"]),
        "maximum_sensor_age_sec": episode["maximum_sensor_age_sec"],
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
        "environment_id": ENVIRONMENT_ID,
        "run_directory": str(run_directory),
        "config_path": str(config_path),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_revision": TRAINING_REVISION,
        "action_settings": state.get("action_settings", asdict(ActionSettings())),
        "reward_settings": state.get("reward_settings", asdict(RewardSettings())),
        "control_mode": CONTROL_MODE,
        "detour_practice": state.get("detour_practice", False),
        "curriculum_settings": state.get("curriculum_settings"),
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
    # Recursively serialize every nested settings dataclass, including reward.
    return TrainingSettings.from_mapping(asdict(settings))


def _parse_arguments(args: Sequence[str] | None) -> argparse.Namespace:
    """Parse trainer command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train PPO with the existing RosNavEnv.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--timing-check", action="store_true",
                        help="Verify paused physics and three 0.1 s action steps, then exit.")
    parser.add_argument(
        "--reset-exploration", action="store_true",
        help="With --resume, reset only sampling log std and its Adam state once.",
    )
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
    reset_exploration: bool = False,
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
        agent, state = _resume_training_agent(
            settings, resume_path, device, reset_exploration,
        )
    state["reward_settings"] = asdict(settings.reward)
    state["action_settings"] = asdict(settings.action)
    state["detour_practice"] = settings.curriculum.detour_practice
    state["curriculum_settings"] = asdict(settings.curriculum)

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
    validation_logger = CsvLogger(run_directory / "metrics/promotion_checks.csv", VALIDATION_FIELDS)
    environment = None
    rclpy.init(args=None)

    print(f"Run directory: {run_directory}")
    print(f"Training device: {device}")
    print(f"PPO learning_rate={agent.hyperparameters.learning_rate:g} "
          f"entropy_coefficient={agent.hyperparameters.entropy_coefficient:g} "
          "entropy_loss=tanh_squashed")
    print(f"Reward safety_distance={settings.reward.safety_distance_m:g}m "
          f"near_obstacle_penalty_scale={settings.reward.near_obstacle_penalty_scale:g}")
    print(f"Braking penalty scale={settings.reward.braking_penalty_scale:g}")
    print(f"Progress reward: {settings.reward.progress_mode}; "
          f"zero-speed action fraction: {settings.action.stop_fraction:g}; "
          f"Stage 2 corner practice: {settings.curriculum.corner_practice_probability:.0%}; "
          f"mirrored training: {settings.curriculum.mirror_training}")
    print(
        "Starting from steps="
        f"{state['total_timesteps']}, update={state['update_index']}, "
        f"level={state['curriculum_level']}"
    )
    if settings.curriculum.detour_practice:
        print(f"Stage 2 detour practice: {DETOUR_PHASE_NAMES[int(state['detour_phase'])]} "
              f"({settings.curriculum.detour_success_threshold:.0%} training window and "
              f"{settings.curriculum.detour_success_threshold:.0%} separate validation "
              "for practice advancement).")

    try:
        environment = RosNavEnv(reward_settings=settings.reward, action_settings=settings.action)
        observation, _, episode = _reset_episode(
            environment,
            state,
            fixed_level,
            settings,
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
                episode["control_interval_sum"] += float(info.get("simulation_interval_sec", 0.0))
                episode["maximum_sensor_age_sec"] = max(
                    episode["maximum_sensor_age_sec"], float(info.get("maximum_sensor_age_sec", 0.0)))
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
                    success_rate, ready_for_validation = (
                        _update_curriculum_after_episode(
                            state,
                            settings,
                            success=info["outcome"] == "success",
                            fixed_level=fixed_level,
                            count_for_promotion=not bool(episode["rehearsal"]),
                        )
                    )
                    print(
                        f"episode={state['episode_index']} "
                        f"stage={episode['training_stage'] + 1} "
                        f"task_stage={episode['level'] + 1} "
                        f"rehearsal={episode['rehearsal']} "
                        f"task={episode['task_kind']} mirror={episode['mirrored']} "
                        f"practice={DETOUR_PHASE_NAMES[episode['practice_phase']] if episode['practice_phase'] >= 0 else 'standard'} "
                        f"area={episode['region_size_m']:.0f}m "
                        f"boxes={episode['randomized_object_count']} "
                        f"outcome={info['outcome']} "
                        f"steps={episode['length']} "
                        f"return={episode['return']:.3f} "
                        f"clearance="
                        f"{float(row['minimum_clearance_m']):.3f} "
                        f"direct_blockers={episode['straight_path_blocker_count']} "
                        f"stuck={int(bool(info.get('stuck', False)))} "
                        f"stage_success={success_rate:.2f} "
                        f"wins={sum(state['recent_successes'])}/{len(state['recent_successes'])} "
                        f"window={len(state['recent_successes'])}/{settings.curriculum.minimum_episodes}", flush=True,
                    )
                    if ready_for_validation:
                        checked_level = int(state["curriculum_level"])
                        checked_practice = _practice_phase(state, settings, checked_level)
                        state["validation_attempts"] += 1
                        attempt = state["validation_attempts"]
                        print(f"Checking Stage {checked_level + 1}, practice={checked_practice}, on separate layouts...", flush=True)
                        checks = _run_promotion_validation(
                            environment, agent, device, checked_level,
                            settings.curriculum.validation_episodes,
                            1_000_000_000 + attempt * 1000,
                            practice_phase=checked_practice,
                            corner_probability=settings.curriculum.corner_practice_probability,
                            mirror_pair=settings.curriculum.mirror_training,
                        )
                        for check in checks:
                            state["validation_steps"] += check["length"]
                            validation_logger.write({**check, "return": check["return_"],
                                "total_timesteps": state["total_timesteps"],
                                "curriculum_level": checked_level, "practice_phase": checked_practice,
                                "attempt": attempt})
                        check_rate = _recent_mean([check["success"] for check in checks])
                        promoted = _complete_promotion_validation(state, settings, check_rate)
                        print(
                            f"Validation success={check_rate:.0%}. "
                            + ((f"Advanced to practice: {DETOUR_PHASE_NAMES[state['detour_phase']]}"
                                if checked_practice is not None else f"Promoted to Stage {state['curriculum_level'] + 1}.")
                               if promoted else "Continuing at the current difficulty."), flush=True,
                        )
                        if promoted:
                            completed_name = (f"detour_phase_{checked_practice + 1}_complete.pt"
                                              if checked_practice is not None
                                              else f"stage_{checked_level + 1}_complete.pt")
                            _save_training_checkpoint(run_directory,
                                completed_name, agent, state, config_path)
                            _save_training_checkpoint(run_directory, "latest.pt", agent, state, config_path)
                    if (
                        int(state["total_timesteps"])
                        < settings.total_timesteps
                    ):
                        observation, _, episode = _reset_episode(
                            environment,
                            state,
                            fixed_level,
                            settings,
                        )

            paused_time_ns = environment.simulation_time_ns
            environment.assert_simulation_time_unchanged(paused_time_ns)
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
            environment.assert_simulation_time_unchanged(paused_time_ns)
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
                "validation_steps": state["validation_steps"],
                "detour_phase": state["detour_phase"],
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
                environment.assert_simulation_time_unchanged(paused_time_ns)

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
        _stop_and_pause(environment)
        interrupted_path = _save_training_checkpoint(
            run_directory,
            "interrupted.pt",
            agent,
            state,
            config_path,
        )
        print(f"Training interrupted safely: {interrupted_path}")
        _save_training_checkpoint(run_directory, "latest.pt", agent, state, config_path)
    except Exception:
        _stop_and_pause(environment)
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
        validation_logger.close()
        if rclpy.ok():
            rclpy.shutdown()


def _stop_and_pause(environment: Any) -> None:
    if environment is not None:
        try:
            environment.publish_stop()
            environment.pause_simulation()
        except Exception as error:
            print(f"Could not confirm pause during shutdown: {error}", flush=True)


def _run_timing_check(settings: TrainingSettings) -> None:
    """Live Gazebo gate: retain a forward command during artificial compute waits."""
    import rclpy
    from nav_learning.ros_nav_env import RosNavEnv
    environment = None
    rclpy.init(args=None)
    try:
        environment = RosNavEnv(reward_settings=settings.reward, action_settings=settings.action)
        environment.reset(seed=60000, curriculum_level=1, practice_phase=0)
        for index in range(3):
            _, _, terminated, truncated, info = environment.step(np.array([0.0, 0.0], dtype=np.float32))
            if terminated or truncated:
                raise RuntimeError("Timing-check task ended unexpectedly; inspect the reset.")
            interval = info["simulation_interval_sec"]
            if abs(interval - 0.10) > 0.002:
                raise RuntimeError(f"Unexpected action interval: {interval:.6f}s")
            paused_ns = environment.simulation_time_ns
            time.sleep(0.25)  # Artificial compute delay; physics must remain frozen.
            environment.assert_simulation_time_unchanged(paused_ns)
            drift = (environment.simulation_time_ns - paused_ns) / 1e9
            print(f"timing_check={index + 1}/3 action_dt={interval:.3f}s "
                  f"sensor_age={info['maximum_sensor_age_sec']:.3f}s idle_drift={drift:.3f}s", flush=True)
        print("TIMING CHECK PASSED: 0.1 s action steps; no simulation advance during compute waits.", flush=True)
        # Confirm the physical steering sign, rather than assuming a visual
        # left/right preference must have come from the neural network.
        for name, sign in (("left", 1.0), ("right", -1.0)):
            start_yaw, start_xy = info["robot_yaw"], (info["robot_x"], info["robot_y"])
            for _ in range(4):
                _, _, terminated, truncated, info = environment.step(
                    np.array([-1.0, sign * 0.5], dtype=np.float32))
                if terminated or truncated:
                    raise RuntimeError("Stationary-turn check ended unexpectedly.")
            turn = wrap_to_pi(info["robot_yaw"] - start_yaw)
            drift = math.dist(start_xy, (info["robot_x"], info["robot_y"]))
            if sign * turn < 0.02 or drift > 0.04:
                raise RuntimeError(f"{name} turn failed: yaw_change={turn:.3f}, translation={drift:.3f}m")
            print(f"turn_check={name} yaw_change={turn:+.3f}rad translation={drift:.3f}m", flush=True)
        print("TURN CHECK PASSED: both steering signs and stationary turns verified.", flush=True)
    finally:
        _stop_and_pause(environment)
        if environment is not None:
            environment.close()
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
    if arguments.timing_check:
        _run_timing_check(settings)
        return
    resume_path = (
        Path(arguments.resume).expanduser().resolve()
        if arguments.resume
        else None
    )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    if resume_path is not None and arguments.seed is not None:
        raise ValueError("Do not override --seed when resuming a run.")
    if arguments.reset_exploration and resume_path is None:
        raise ValueError("--reset-exploration requires --resume.")
    fixed_level = 0 if arguments.smoke else arguments.fixed_level
    run_directory = _resolve_run_directory(
        settings,
        resume_path,
        arguments.run_dir,
    )
    copied_config = run_directory / "config_used.yaml"
    if not copied_config.exists():
        with copied_config.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(asdict(settings), stream, sort_keys=False)
    else:
        saved_settings = load_training_settings(copied_config)
        if (saved_settings.ppo != settings.ppo or saved_settings.reward != settings.reward
                or saved_settings.action != settings.action
                or saved_settings.curriculum != settings.curriculum):
            raise ValueError("Changing training settings requires a new --run-dir; resume the existing checkpoint into that folder.")
    run_training(
        settings=settings,
        config_path=config_path,
        run_directory=run_directory,
        resume_path=resume_path,
        fixed_level=fixed_level,
        reset_exploration=arguments.reset_exploration,
    )


if __name__ == "__main__":
    main()
