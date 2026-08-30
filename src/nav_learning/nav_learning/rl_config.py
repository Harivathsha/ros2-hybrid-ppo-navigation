"""Load and validate the standalone PPO training configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from nav_learning.ppo import PPOHyperparameters


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
    promotion_window: int = 50
    minimum_episodes: int = 50
    success_threshold: float = 0.70

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
        for name in ("start_level", "max_level"):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"curriculum.{name} must be an integer.")
            if value not in (0, 1, 2):
                raise ValueError(f"curriculum.{name} must be 0, 1, or 2.")
        if settings.start_level > settings.max_level:
            raise ValueError("start_level cannot exceed max_level.")
        for name in ("promotion_window", "minimum_episodes"):
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
        settings = cls(**plain_values)

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
