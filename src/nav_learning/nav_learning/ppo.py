"""From-scratch Proximal Policy Optimisation for robot navigation.

The module is independent of ROS and Gazebo. It contains the actor-critic,
tanh-squashed Gaussian action mathematics, rollout storage, Generalised
Advantage Estimation (GAE), the clipped PPO update, and checkpoint helpers.
Run ``python3 -m nav_learning.ppo`` for a synthetic smoke test.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Normal


LIDAR_FEATURES = 72
GOAL_MOTION_FEATURES = 5
OBSERVATION_SIZE = LIDAR_FEATURES + GOAL_MOTION_FEATURES
ACTION_SIZE = 2

LIDAR_HIDDEN_SIZE = 256
LIDAR_OUTPUT_SIZE = 128
GOAL_MOTION_OUTPUT_SIZE = 32
COMBINED_INPUT_SIZE = LIDAR_OUTPUT_SIZE + GOAL_MOTION_OUTPUT_SIZE
COMBINED_OUTPUT_SIZE = 128

INITIAL_LOG_STANDARD_DEVIATION = -0.5
MIN_LOG_STANDARD_DEVIATION = -5.0
MAX_LOG_STANDARD_DEVIATION = 1.0
ACTION_OPEN_INTERVAL_EPSILON = 1e-6

OBSERVATION_BOUND_TOLERANCE = 1e-5
BEARING_UNIT_TOLERANCE = 1e-4
ADVANTAGE_NORMALISATION_EPSILON = 1e-8
CHECKPOINT_FORMAT_VERSION = 1


def _require_finite_float32_tensor(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """Reject a tensor that cannot safely enter PPO mathematics."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.dtype != torch.float32:
        raise TypeError(
            f"{name} must use torch.float32, got {tensor.dtype}."
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains NaN or infinity.")


def _require_range(
    tensor: torch.Tensor,
    minimum: float,
    maximum: float,
    name: str,
) -> None:
    """Check one normalised observation component range."""

    tolerance = OBSERVATION_BOUND_TOLERANCE
    outside = (tensor < minimum - tolerance) | (
        tensor > maximum + tolerance
    )
    if bool(outside.any().item()):
        raise ValueError(
            f"{name} must stay in [{minimum}, {maximum}]."
        )


def split_observation(
    observation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and split one observation or a minibatch."""

    _require_finite_float32_tensor(observation, "observation")
    if observation.ndim not in (1, 2):
        raise ValueError(
            "observation must have shape (77,) or (batch_size, 77)."
        )
    if observation.shape[-1] != OBSERVATION_SIZE:
        raise ValueError(
            "observation's final dimension must be "
            f"{OBSERVATION_SIZE}, got {observation.shape[-1]}."
        )

    lidar = observation[..., :LIDAR_FEATURES]
    goal_motion = observation[..., LIDAR_FEATURES:]
    _require_range(lidar, 0.0, 1.0, "normalised LiDAR")
    _require_range(
        goal_motion[..., 0],
        0.0,
        1.0,
        "normalised goal distance",
    )
    _require_range(
        goal_motion[..., 1:],
        -1.0,
        1.0,
        "bearing and velocity features",
    )

    bearing_length_squared = (
        goal_motion[..., 1].square()
        + goal_motion[..., 2].square()
    )
    if not torch.allclose(
        bearing_length_squared,
        torch.ones_like(bearing_length_squared),
        rtol=BEARING_UNIT_TOLERANCE,
        atol=BEARING_UNIT_TOLERANCE,
    ):
        raise ValueError(
            "Goal-bearing sine and cosine do not form a unit vector."
        )
    return lidar, goal_motion


def observation_to_tensor(
    observation: Any,
    device: torch.device | str,
) -> torch.Tensor:
    """Convert one NumPy-like environment observation to a tensor."""

    tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=torch.device(device),
    )
    split_observation(tensor)
    if tensor.shape != (OBSERVATION_SIZE,):
        raise ValueError(
            f"Expected one observation with shape ({OBSERVATION_SIZE},)."
        )
    return tensor


def _initialise_linear(layer: nn.Linear, gain: float) -> None:
    """Apply orthogonal weights and zero biases to one linear layer."""

    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class ActorCritic(nn.Module):
    """Encode LiDAR and goal/motion data, then predict actions and value."""

    def __init__(self) -> None:
        super().__init__()
        self.lidar_encoder = nn.Sequential(
            nn.Linear(LIDAR_FEATURES, LIDAR_HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(LIDAR_HIDDEN_SIZE, LIDAR_OUTPUT_SIZE),
            nn.Tanh(),
        )
        self.goal_motion_encoder = nn.Sequential(
            nn.Linear(
                GOAL_MOTION_FEATURES,
                GOAL_MOTION_OUTPUT_SIZE,
            ),
            nn.Tanh(),
        )
        self.combined_layer = nn.Sequential(
            nn.Linear(COMBINED_INPUT_SIZE, COMBINED_OUTPUT_SIZE),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(COMBINED_OUTPUT_SIZE, ACTION_SIZE)
        self.critic_value = nn.Linear(COMBINED_OUTPUT_SIZE, 1)
        self.actor_log_standard_deviation = nn.Parameter(
            torch.full(
                (ACTION_SIZE,),
                INITIAL_LOG_STANDARD_DEVIATION,
                dtype=torch.float32,
            )
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                _initialise_linear(module, gain=math.sqrt(2.0))
        _initialise_linear(self.actor_mean, gain=0.01)
        _initialise_linear(self.critic_value, gain=1.0)

    @torch.no_grad()
    def clamp_log_standard_deviation_(self) -> None:
        """Keep learned exploration parameters inside a safe range."""

        self.actor_log_standard_deviation.clamp_(
            min=MIN_LOG_STANDARD_DEVIATION,
            max=MAX_LOG_STANDARD_DEVIATION,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Gaussian means, log standard deviations, and values."""

        lidar, goal_motion = split_observation(observation)
        lidar_features = self.lidar_encoder(lidar)
        goal_motion_features = self.goal_motion_encoder(goal_motion)
        combined_features = torch.cat(
            (lidar_features, goal_motion_features),
            dim=-1,
        )
        shared_features = self.combined_layer(combined_features)
        action_mean = self.actor_mean(shared_features)
        state_value = self.critic_value(shared_features).squeeze(-1)
        log_standard_deviation = torch.clamp(
            self.actor_log_standard_deviation,
            min=MIN_LOG_STANDARD_DEVIATION,
            max=MAX_LOG_STANDARD_DEVIATION,
        ).expand_as(action_mean)

        expected_action_shape = observation.shape[:-1] + (ACTION_SIZE,)
        expected_value_shape = observation.shape[:-1]
        if action_mean.shape != expected_action_shape:
            raise RuntimeError("Actor mean has an unexpected shape.")
        if log_standard_deviation.shape != expected_action_shape:
            raise RuntimeError("Actor log standard deviation has a bad shape.")
        if state_value.shape != expected_value_shape:
            raise RuntimeError("Critic value has an unexpected shape.")
        _require_finite_float32_tensor(action_mean, "action_mean")
        _require_finite_float32_tensor(
            log_standard_deviation,
            "log_standard_deviation",
        )
        _require_finite_float32_tensor(state_value, "state_value")
        return action_mean, log_standard_deviation, state_value

    def _distribution_and_value(
        self,
        observation: torch.Tensor,
    ) -> tuple[Normal, torch.Tensor]:
        """Build the unsquashed Gaussian and predict the state value."""

        action_mean, log_standard_deviation, state_value = self(
            observation
        )
        standard_deviation = log_standard_deviation.exp()
        _require_finite_float32_tensor(
            standard_deviation,
            "standard_deviation",
        )
        if bool((standard_deviation <= 0.0).any().item()):
            raise RuntimeError("Action standard deviation must be positive.")
        return Normal(action_mean, standard_deviation), state_value

    @staticmethod
    def _squashed_log_probability(
        distribution: Normal,
        pre_tanh_action: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the exact, stable tanh Jacobian correction."""

        gaussian_log_probability = distribution.log_prob(
            pre_tanh_action
        )
        log_absolute_jacobian = 2.0 * (
            math.log(2.0)
            - pre_tanh_action
            - functional.softplus(-2.0 * pre_tanh_action)
        )
        log_probability = (
            gaussian_log_probability - log_absolute_jacobian
        ).sum(dim=-1)
        _require_finite_float32_tensor(
            log_probability,
            "log_probability",
        )
        return log_probability

    @staticmethod
    def _validate_action(
        observation: torch.Tensor,
        action: torch.Tensor,
        require_open_interval: bool,
    ) -> None:
        """Check action shape, device, type, finiteness, and support."""

        _require_finite_float32_tensor(action, "action")
        expected_shape = observation.shape[:-1] + (ACTION_SIZE,)
        if action.shape != expected_shape:
            raise ValueError(
                f"action must have shape {tuple(expected_shape)}, "
                f"got {tuple(action.shape)}."
            )
        if action.device != observation.device:
            raise ValueError(
                "action and observation must be on the same device."
            )
        if bool((action.abs() > 1.0).any().item()):
            raise ValueError("action escaped environment bounds [-1, 1].")
        if require_open_interval and bool(
            (action.abs() >= 1.0).any().item()
        ):
            raise ValueError(
                "A training action must be inside (-1, 1) for atanh."
            )

    @torch.no_grad()
    def sample_action(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a bounded action and return log probability and value."""

        distribution, state_value = self._distribution_and_value(
            observation
        )
        sampled_pre_tanh_action = distribution.sample()
        action = torch.tanh(sampled_pre_tanh_action).clamp(
            min=-1.0 + ACTION_OPEN_INTERVAL_EPSILON,
            max=1.0 - ACTION_OPEN_INTERVAL_EPSILON,
        )
        # Reconstruct after the numerical clamp so stored-action
        # reevaluation produces exactly the same old log probability.
        pre_tanh_action = torch.atanh(action)
        log_probability = self._squashed_log_probability(
            distribution,
            pre_tanh_action,
        )
        self._validate_action(
            observation,
            action,
            require_open_interval=True,
        )
        return action, log_probability, state_value

    @torch.no_grad()
    def deterministic_action(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Return tanh(mean), with no exploration noise."""

        action_mean, _, _ = self(observation)
        action = torch.tanh(action_mean)
        self._validate_action(
            observation,
            action,
            require_open_interval=False,
        )
        return action

    def evaluate_action_with_entropy(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reevaluate stored actions and return entropy and values."""

        distribution, state_value = self._distribution_and_value(
            observation
        )
        self._validate_action(
            observation,
            action,
            require_open_interval=True,
        )
        pre_tanh_action = torch.atanh(action)
        log_probability = self._squashed_log_probability(
            distribution,
            pre_tanh_action,
        )
        # The transformed distribution has no simple analytic entropy.
        # Gaussian entropy is a stable exploration proxy used by the loss.
        gaussian_entropy = distribution.entropy().sum(dim=-1)
        _require_finite_float32_tensor(
            gaussian_entropy,
            "gaussian_entropy",
        )
        return log_probability, gaussian_entropy, state_value

    def evaluate_action(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute a stored action's log probability and value."""

        log_probability, _, state_value = (
            self.evaluate_action_with_entropy(observation, action)
        )
        return log_probability, state_value

    def predict_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Predict the state value while retaining gradients."""

        _, _, state_value = self(observation)
        return state_value


@dataclass(frozen=True)
class PPOHyperparameters:
    """Validated hyperparameters used by one PPO optimiser."""

    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.20
    value_clip_coefficient: float = 0.20
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.50
    max_gradient_norm: float = 0.50
    update_epochs: int = 10
    minibatch_size: int = 64
    target_kl: float | None = 0.03
    adam_epsilon: float = 1e-5

    def validated(self) -> "PPOHyperparameters":
        """Validate all fields and return this immutable instance."""

        positive = {
            "learning_rate": self.learning_rate,
            "clip_coefficient": self.clip_coefficient,
            "value_clip_coefficient": self.value_clip_coefficient,
            "max_gradient_norm": self.max_gradient_norm,
            "adam_epsilon": self.adam_epsilon,
        }
        nonnegative = {
            "entropy_coefficient": self.entropy_coefficient,
            "value_loss_coefficient": self.value_loss_coefficient,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in nonnegative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name, value in {
            "discount_factor": self.discount_factor,
            "gae_lambda": self.gae_lambda,
        }.items():
            if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        for name, value in {
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.target_kl is not None:
            if (
                not math.isfinite(float(self.target_kl))
                or self.target_kl <= 0.0
            ):
                raise ValueError("target_kl must be positive or null.")
        return self

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "PPOHyperparameters":
        """Construct settings while rejecting misspelled YAML keys."""

        known_names = {field.name for field in fields(cls)}
        unknown_names = sorted(set(values) - known_names)
        if unknown_names:
            raise ValueError(
                "Unknown PPO settings: " + ", ".join(unknown_names)
            )
        return cls(**dict(values)).validated()


@dataclass(frozen=True)
class RolloutBatch:
    """One shuffled PPO minibatch."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Store one on-policy rollout and calculate GAE targets."""

    def __init__(
        self,
        capacity: int,
        device: torch.device | str,
    ) -> None:
        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity <= 0
        ):
            raise ValueError("capacity must be a positive integer.")

        self.capacity = capacity
        self.device = torch.device(device)
        self.observations = torch.empty(
            (capacity, OBSERVATION_SIZE),
            dtype=torch.float32,
            device=self.device,
        )
        self.device = self.observations.device

        self.actions = torch.empty(
            (capacity, ACTION_SIZE),
            dtype=torch.float32,
            device=self.device,
        )
        self.rewards = torch.empty(
            capacity,
            dtype=torch.float32,
            device=self.device,
        )
        self.values = torch.empty_like(self.rewards)
        self.next_values = torch.empty_like(self.rewards)
        self.log_probabilities = torch.empty_like(self.rewards)
        self.terminated = torch.empty(
            capacity,
            dtype=torch.bool,
            device=self.device,
        )
        self.truncated = torch.empty_like(self.terminated)
        self.advantages = torch.empty_like(self.rewards)
        self.returns = torch.empty_like(self.rewards)
        self._size = 0
        self._advantages_computed = False

    def __len__(self) -> int:
        """Return the number of stored transitions."""

        return self._size

    @property
    def is_full(self) -> bool:
        """Report whether the allocated rollout is complete."""

        return self._size == self.capacity

    @property
    def advantages_computed(self) -> bool:
        """Report whether valid advantages and returns are available."""

        return self._advantages_computed

    def reset(self) -> None:
        """Mark storage empty without reallocating tensors."""

        self._size = 0
        self._advantages_computed = False

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        next_value: torch.Tensor,
        log_probability: torch.Tensor,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store one detached transition after contract validation."""

        if self.is_full:
            raise RuntimeError("RolloutBuffer capacity has been exceeded.")
        if not isinstance(terminated, bool):
            raise TypeError("terminated must be a bool.")
        if not isinstance(truncated, bool):
            raise TypeError("truncated must be a bool.")
        if terminated and truncated:
            raise ValueError(
                "A transition cannot be both terminated and truncated."
            )

        split_observation(observation)
        if observation.shape != (OBSERVATION_SIZE,):
            raise ValueError(
                f"observation must have shape ({OBSERVATION_SIZE},)."
            )
        if observation.device != self.device:
            raise ValueError("observation is on the wrong device.")
        _require_finite_float32_tensor(action, "action")
        if action.shape != (ACTION_SIZE,):
            raise ValueError(f"action must have shape ({ACTION_SIZE},).")
        if action.device != self.device:
            raise ValueError("action is on the wrong device.")
        if bool((action.abs() >= 1.0).any().item()):
            raise ValueError("Stored actions must be inside (-1, 1).")

        try:
            reward_value = float(reward)
        except (TypeError, ValueError) as error:
            raise TypeError("reward must be convertible to float.") from error
        if isinstance(reward, bool) or not math.isfinite(reward_value):
            raise ValueError("reward must be a finite real number.")

        for tensor, name in (
            (value, "value"),
            (next_value, "next_value"),
            (log_probability, "log_probability"),
        ):
            _require_finite_float32_tensor(tensor, name)
            if tensor.shape != torch.Size([]):
                raise ValueError(f"{name} must be a scalar tensor.")
            if tensor.device != self.device:
                raise ValueError(f"{name} is on the wrong device.")

        index = self._size
        self.observations[index].copy_(observation.detach())
        self.actions[index].copy_(action.detach())
        self.rewards[index] = reward_value
        self.values[index].copy_(value.detach())
        self.next_values[index].copy_(next_value.detach())
        self.log_probabilities[index].copy_(
            log_probability.detach()
        )
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self._size += 1
        self._advantages_computed = False

    @torch.no_grad()
    def compute_returns_and_advantages(
        self,
        discount_factor: float,
        gae_lambda: float,
        normalise_advantages: bool = True,
    ) -> dict[str, float]:
        """Calculate GAE using distinct bootstrap and boundary masks."""

        if self._size == 0:
            raise RuntimeError("Cannot compute GAE for an empty rollout.")
        gamma = float(discount_factor)
        lambda_value = float(gae_lambda)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("discount_factor must be in [0, 1].")
        if not 0.0 <= lambda_value <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")

        gae = torch.zeros(
            (),
            dtype=torch.float32,
            device=self.device,
        )
        for index in range(self._size - 1, -1, -1):
            terminated = self.terminated[index]
            truncated = self.truncated[index]
            # Genuine terminal states have no future value. A time-limit
            # truncation still bootstraps from its final observation.
            bootstrap_mask = (~terminated).to(torch.float32)
            # Recursion stops at every episode boundary because the next
            # stored state may belong to a freshly reset episode.
            continuation_mask = (~(terminated | truncated)).to(
                torch.float32
            )
            temporal_difference = (
                self.rewards[index]
                + gamma
                * bootstrap_mask
                * self.next_values[index]
                - self.values[index]
            )
            gae = (
                temporal_difference
                + gamma
                * lambda_value
                * continuation_mask
                * gae
            )
            self.advantages[index] = gae

        active = slice(0, self._size)
        active_advantages = self.advantages[active]
        self.returns[active].copy_(
            active_advantages + self.values[active]
        )
        raw_mean = active_advantages.mean()
        raw_standard_deviation = active_advantages.std(correction=0)
        if normalise_advantages:
            active_advantages.sub_(raw_mean)
            if bool(
                (
                    raw_standard_deviation
                    <= ADVANTAGE_NORMALISATION_EPSILON
                ).item()
            ):
                active_advantages.zero_()
            else:
                active_advantages.div_(
                    raw_standard_deviation
                    + ADVANTAGE_NORMALISATION_EPSILON
                )

        if not bool(torch.isfinite(active_advantages).all().item()):
            raise RuntimeError("Computed advantages are not finite.")
        if not bool(torch.isfinite(self.returns[active]).all().item()):
            raise RuntimeError("Computed returns are not finite.")
        self._advantages_computed = True
        return {
            "raw_advantage_mean": float(raw_mean.item()),
            "raw_advantage_std": float(
                raw_standard_deviation.item()
            ),
            "final_advantage_mean": float(
                active_advantages.mean().item()
            ),
            "final_advantage_std": float(
                active_advantages.std(correction=0).item()
            ),
        }

    def minibatches(
        self,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> Iterator[RolloutBatch]:
        """Yield active rollout tensors in shuffled minibatches."""

        if not self._advantages_computed:
            raise RuntimeError("Compute GAE before requesting minibatches.")
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive.")
        if shuffle:
            indices = torch.randperm(
                self._size,
                device=self.device,
            )
        else:
            indices = torch.arange(
                self._size,
                device=self.device,
            )
        for start in range(0, self._size, minibatch_size):
            batch_indices = indices[start:start + minibatch_size]
            yield RolloutBatch(
                observations=self.observations[batch_indices],
                actions=self.actions[batch_indices],
                old_log_probabilities=(
                    self.log_probabilities[batch_indices]
                ),
                old_values=self.values[batch_indices],
                advantages=self.advantages[batch_indices],
                returns=self.returns[batch_indices],
            )


def calculate_clipped_policy_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    clip_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return clipped actor loss, approximate KL, and clip fraction."""

    log_ratio = new_log_probabilities - old_log_probabilities
    probability_ratio = log_ratio.exp()
    if not bool(torch.isfinite(probability_ratio).all().item()):
        raise RuntimeError("PPO probability ratio is not finite.")
    unclipped_objective = probability_ratio * advantages
    clipped_objective = probability_ratio.clamp(
        1.0 - clip_coefficient,
        1.0 + clip_coefficient,
    ) * advantages
    policy_loss = -torch.minimum(
        unclipped_objective,
        clipped_objective,
    ).mean()
    with torch.no_grad():
        approximate_kl = (
            (probability_ratio - 1.0) - log_ratio
        ).mean()
        clip_fraction = (
            (probability_ratio - 1.0).abs() > clip_coefficient
        ).to(torch.float32).mean()
    return policy_loss, approximate_kl, clip_fraction


def calculate_clipped_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    value_clip_coefficient: float,
) -> torch.Tensor:
    """Return PPO's clipped critic loss."""

    clipped_values = old_values + (new_values - old_values).clamp(
        -value_clip_coefficient,
        value_clip_coefficient,
    )
    original_squared_error = (new_values - returns).square()
    clipped_squared_error = (clipped_values - returns).square()
    return 0.5 * torch.maximum(
        original_squared_error,
        clipped_squared_error,
    ).mean()


def explained_variance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Measure how much return variance the critic explains."""

    target_variance = targets.var(correction=0)
    if bool((target_variance <= 1e-12).item()):
        return float("nan")
    residual_variance = (targets - predictions).var(correction=0)
    return float((1.0 - residual_variance / target_variance).item())


class PPOAgent:
    """Own an actor-critic and apply multiple clipped PPO epochs."""

    def __init__(
        self,
        model: ActorCritic,
        hyperparameters: PPOHyperparameters,
    ) -> None:
        self.model = model
        self.hyperparameters = hyperparameters.validated()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparameters.learning_rate,
            eps=self.hyperparameters.adam_epsilon,
        )

    @property
    def device(self) -> torch.device:
        """Return the model's device."""

        return next(self.model.parameters()).device

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the optimiser learning rate for linear annealing."""

        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def update(self, rollout: RolloutBuffer) -> dict[str, float]:
        """Optimise one complete on-policy rollout."""

        if rollout.device != self.device:
            raise ValueError("Rollout and PPO model devices differ.")
        if not rollout.advantages_computed:
            raise RuntimeError("GAE must be computed before PPO update.")
        if len(rollout) == 0:
            raise RuntimeError("Cannot update from an empty rollout.")

        settings = self.hyperparameters
        metric_totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
        }
        samples_seen = 0
        optimiser_steps = 0
        epochs_completed = 0
        early_stopped = False

        self.model.train()
        minibatch_size = min(settings.minibatch_size, len(rollout))
        for _ in range(settings.update_epochs):
            epoch_was_completed = True
            for batch in rollout.minibatches(minibatch_size):
                (
                    new_log_probabilities,
                    gaussian_entropy,
                    new_values,
                ) = self.model.evaluate_action_with_entropy(
                    batch.observations,
                    batch.actions,
                )
                (
                    policy_loss,
                    approximate_kl,
                    clip_fraction,
                ) = calculate_clipped_policy_loss(
                    new_log_probabilities,
                    batch.old_log_probabilities,
                    batch.advantages,
                    settings.clip_coefficient,
                )
                value_loss = calculate_clipped_value_loss(
                    new_values,
                    batch.old_values,
                    batch.returns,
                    settings.value_clip_coefficient,
                )
                entropy = gaussian_entropy.mean()
                total_loss = (
                    policy_loss
                    + settings.value_loss_coefficient * value_loss
                    - settings.entropy_coefficient * entropy
                )
                if not bool(torch.isfinite(total_loss).item()):
                    raise RuntimeError("The PPO loss is not finite.")

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=settings.max_gradient_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.model.clamp_log_standard_deviation_()

                batch_size = int(batch.observations.shape[0])
                values = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "total_loss": total_loss,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "gradient_norm": gradient_norm,
                }
                for name, value in values.items():
                    metric_totals[name] += (
                        float(value.detach().item()) * batch_size
                    )
                samples_seen += batch_size
                optimiser_steps += 1

                if (
                    settings.target_kl is not None
                    and float(approximate_kl.item())
                    > 1.5 * settings.target_kl
                ):
                    early_stopped = True
                    epoch_was_completed = False
                    break
            if epoch_was_completed:
                epochs_completed += 1
            if early_stopped:
                break

        if samples_seen == 0:
            raise RuntimeError("PPO performed no optimiser steps.")
        active = slice(0, len(rollout))
        with torch.no_grad():
            post_update_values = self.model.predict_value(
                rollout.observations[active]
            )
            critic_explained_variance = explained_variance(
                post_update_values,
                rollout.returns[active],
            )

        metrics = {
            name: total / samples_seen
            for name, total in metric_totals.items()
        }
        metrics.update(
            {
                "explained_variance": critic_explained_variance,
                "optimiser_steps": float(optimiser_steps),
                "epochs_completed": float(epochs_completed),
                "early_stopped": float(early_stopped),
                "learning_rate": float(
                    self.optimizer.param_groups[0]["lr"]
                ),
                "actor_log_std_linear": float(
                    self.model.actor_log_standard_deviation[0].item()
                ),
                "actor_log_std_angular": float(
                    self.model.actor_log_standard_deviation[1].item()
                ),
            }
        )
        return metrics


def save_checkpoint(
    path: str | Path,
    agent: PPOAgent,
    training_state: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model, optimiser, settings, and training state."""

    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "hyperparameters": asdict(agent.hyperparameters),
        "training_state": dict(training_state),
        "metadata": dict(metadata or {}),
    }
    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )
    torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def _load_payload(
    path: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load and validate a checkpoint created by this module."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=torch.device(device),
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            checkpoint_path,
            map_location=torch.device(device),
        )
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported PPO checkpoint format version.")
    if payload.get("observation_size") != OBSERVATION_SIZE:
        raise ValueError("Checkpoint observation size is incompatible.")
    if payload.get("action_size") != ACTION_SIZE:
        raise ValueError("Checkpoint action size is incompatible.")
    return payload


def load_checkpoint(
    path: str | Path,
    device: torch.device | str,
    load_optimizer: bool = True,
) -> tuple[PPOAgent, dict[str, Any]]:
    """Rebuild a PPO agent from a trusted project checkpoint."""

    resolved_device = torch.device(device)
    payload = _load_payload(path, resolved_device)
    hyperparameters = PPOHyperparameters.from_mapping(
        payload["hyperparameters"]
    )
    model = ActorCritic().to(resolved_device)
    agent = PPOAgent(model, hyperparameters)
    agent.model.load_state_dict(payload["model_state_dict"], strict=True)
    if load_optimizer:
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        for state in agent.optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(resolved_device)
    return agent, payload


def choose_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` with a clear failure."""

    normalised = requested.strip().lower()
    if normalised == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if normalised == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")
    if normalised == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cpu, cuda.")


def _make_smoke_observations() -> torch.Tensor:
    """Create four valid synthetic observations without ROS or Gazebo."""

    lidar = torch.linspace(
        0.0,
        1.0,
        steps=LIDAR_FEATURES,
        dtype=torch.float32,
    ).repeat(4, 1)
    goal_distance = torch.tensor(
        [0.20, 0.45, 0.70, 0.90],
        dtype=torch.float32,
    )
    bearing = torch.tensor(
        [0.0, 0.5, -1.0, 2.0],
        dtype=torch.float32,
    )
    linear_velocity = torch.tensor(
        [0.0, 0.25, 0.50, 1.0],
        dtype=torch.float32,
    )
    angular_velocity = torch.tensor(
        [0.0, -0.50, 0.25, 1.0],
        dtype=torch.float32,
    )
    goal_motion = torch.stack(
        (
            goal_distance,
            bearing.sin(),
            bearing.cos(),
            linear_velocity,
            angular_velocity,
        ),
        dim=-1,
    )
    return torch.cat((lidar, goal_motion), dim=-1)


def _run_smoke_test() -> dict[str, float]:
    """Exercise sampling, GAE, and a real PPO gradient update."""

    torch.manual_seed(7)
    observations = _make_smoke_observations()
    model = ActorCritic()
    actions, log_probabilities, values = model.sample_action(
        observations
    )
    reevaluated_log_probabilities, reevaluated_values = (
        model.evaluate_action(observations, actions)
    )
    if not torch.allclose(
        log_probabilities,
        reevaluated_log_probabilities,
        rtol=2e-5,
        atol=2e-5,
    ):
        raise RuntimeError("Action log-probability round trip failed.")
    if not torch.allclose(values, reevaluated_values):
        raise RuntimeError("Value reevaluation changed unexpectedly.")

    rollout = RolloutBuffer(capacity=4, device="cpu")
    rewards = (1.0, 2.0, 3.0, 0.5)
    next_values = (0.6, 123.0, 1.5, 0.4)
    terminated = (False, True, False, False)
    truncated = (False, False, True, False)
    fixed_values = (0.5, 0.6, 1.0, 0.2)
    for index in range(4):
        rollout.add(
            observation=observations[index],
            action=actions[index],
            reward=rewards[index],
            value=observations.new_tensor(fixed_values[index]),
            next_value=observations.new_tensor(next_values[index]),
            log_probability=log_probabilities[index],
            terminated=terminated[index],
            truncated=truncated[index],
        )
    rollout.compute_returns_and_advantages(
        discount_factor=0.99,
        gae_lambda=0.95,
        normalise_advantages=False,
    )
    expected_advantages = observations.new_tensor(
        [2.4107, 1.4, 3.485, 0.696]
    )
    if not torch.allclose(
        rollout.advantages[:4],
        expected_advantages,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError("Hand-calculated GAE test failed.")
    rollout.compute_returns_and_advantages(
        discount_factor=0.99,
        gae_lambda=0.95,
        normalise_advantages=True,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    agent = PPOAgent(
        model,
        PPOHyperparameters(
            update_epochs=2,
            minibatch_size=2,
            target_kl=None,
        ),
    )
    metrics = agent.update(rollout)
    changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    if not changed:
        raise RuntimeError("PPO update did not change any parameter.")
    return metrics


def main() -> None:
    """Run a ROS-free end-to-end PPO smoke test."""

    metrics = _run_smoke_test()
    parameter_count = sum(
        parameter.numel()
        for parameter in ActorCritic().parameters()
        if parameter.requires_grad
    )
    print("Complete PPO core smoke test passed.")
    print(f"trainable_parameters={parameter_count}")
    print("sample_log_probability_reevaluation=True")
    print("terminated_bootstrap_masked=True")
    print("truncated_bootstrap_preserved=True")
    print("gae_hand_calculation_match=True")
    print("ppo_gradient_update=True")
    print(f"policy_loss={metrics['policy_loss']:.6f}")
    print(f"value_loss={metrics['value_loss']:.6f}")


if __name__ == "__main__":
    main()
