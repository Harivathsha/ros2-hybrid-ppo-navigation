"""Unit tests for PPO mathematics that do not require ROS or Gazebo."""

import math

import pytest
import torch

from nav_learning.ppo import (
    ACTION_SIZE,
    LIDAR_FEATURES,
    ActorCritic,
    PPOAgent,
    PPOHyperparameters,
    RolloutBuffer,
    calculate_clipped_policy_loss,
    choose_device,
    load_checkpoint,
    save_checkpoint,
)


def make_observations(batch_size=8):
    """Build a valid deterministic observation batch."""

    lidar = torch.linspace(
        0.0,
        1.0,
        LIDAR_FEATURES,
        dtype=torch.float32,
    ).repeat(batch_size, 1)
    bearing = torch.linspace(
        -1.0,
        1.0,
        batch_size,
        dtype=torch.float32,
    )
    goal_motion = torch.stack(
        (
            torch.linspace(0.1, 0.9, batch_size),
            bearing.sin(),
            bearing.cos(),
            torch.zeros(batch_size),
            torch.zeros(batch_size),
        ),
        dim=-1,
    ).to(torch.float32)
    return torch.cat((lidar, goal_motion), dim=-1)


def make_rollout(model, size=8):
    """Create one internally consistent synthetic rollout."""

    observations = make_observations(size)
    actions, log_probabilities, values = model.sample_action(
        observations
    )
    buffer = RolloutBuffer(size, device="cpu")
    for index in range(size):
        terminated = index == size - 1
        next_value = (
            torch.zeros((), dtype=torch.float32)
            if terminated
            else values[(index + 1) % size]
        )
        buffer.add(
            observation=observations[index],
            action=actions[index],
            reward=float(index % 3) - 0.25,
            value=values[index],
            next_value=next_value,
            log_probability=log_probabilities[index],
            terminated=terminated,
            truncated=False,
        )
    buffer.compute_returns_and_advantages(
        discount_factor=0.99,
        gae_lambda=0.95,
        normalise_advantages=True,
    )
    return buffer


def test_actor_critic_shapes_and_parameter_count():
    """The locked network architecture must not change accidentally."""

    model = ActorCritic()
    observations = make_observations(4)
    means, log_stds, values = model(observations)
    assert means.shape == (4, ACTION_SIZE)
    assert log_stds.shape == (4, ACTION_SIZE)
    assert values.shape == (4,)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert parameter_count == 72773


def test_squashed_action_log_probability_round_trip():
    """Stored actions must reproduce their collection log probabilities."""

    torch.manual_seed(7)
    model = ActorCritic()
    observations = make_observations(16)
    actions, old_log_probabilities, old_values = model.sample_action(
        observations
    )
    new_log_probabilities, new_values = model.evaluate_action(
        observations,
        actions,
    )
    assert torch.all(actions.abs() < 1.0)
    assert torch.allclose(
        old_log_probabilities,
        new_log_probabilities,
        rtol=2e-5,
        atol=2e-5,
    )
    assert torch.allclose(old_values, new_values)


def test_gae_termination_and_truncation_masks():
    """Termination masks bootstrap while timeout preserves it."""

    torch.manual_seed(11)
    model = ActorCritic()
    observations = make_observations(4)
    actions, log_probabilities, _ = model.sample_action(observations)
    buffer = RolloutBuffer(4, device="cpu")
    rewards = (1.0, 2.0, 3.0, 0.5)
    values = (0.5, 0.6, 1.0, 0.2)
    next_values = (0.6, 123.0, 1.5, 0.4)
    terminated = (False, True, False, False)
    truncated = (False, False, True, False)
    for index in range(4):
        buffer.add(
            observation=observations[index],
            action=actions[index],
            reward=rewards[index],
            value=torch.tensor(values[index], dtype=torch.float32),
            next_value=torch.tensor(
                next_values[index],
                dtype=torch.float32,
            ),
            log_probability=log_probabilities[index],
            terminated=terminated[index],
            truncated=truncated[index],
        )
    buffer.compute_returns_and_advantages(
        discount_factor=0.99,
        gae_lambda=0.95,
        normalise_advantages=False,
    )
    expected_advantages = torch.tensor(
        [2.4107, 1.4, 3.485, 0.696],
        dtype=torch.float32,
    )
    expected_returns = torch.tensor(
        [2.9107, 2.0, 4.485, 0.896],
        dtype=torch.float32,
    )
    assert torch.allclose(
        buffer.advantages[:4],
        expected_advantages,
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.allclose(
        buffer.returns[:4],
        expected_returns,
        rtol=1e-5,
        atol=1e-5,
    )


def test_policy_clipping_uses_pessimistic_surrogate():
    """Both positive and negative advantages must clip correctly."""

    new_log_probabilities = torch.log(
        torch.tensor([1.3, 0.7], dtype=torch.float32)
    )
    old_log_probabilities = torch.zeros(2, dtype=torch.float32)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)
    loss, approximate_kl, clip_fraction = (
        calculate_clipped_policy_loss(
            new_log_probabilities,
            old_log_probabilities,
            advantages,
            clip_coefficient=0.2,
        )
    )
    assert float(loss.item()) == pytest.approx(-0.2, abs=1e-6)
    assert float(approximate_kl.item()) > 0.0
    assert float(clip_fraction.item()) == pytest.approx(1.0)


def test_ppo_update_changes_parameters_and_returns_finite_metrics():
    """A complete PPO update must backpropagate through the network."""

    torch.manual_seed(19)
    model = ActorCritic()
    rollout = make_rollout(model, size=8)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    agent = PPOAgent(
        model,
        PPOHyperparameters(
            update_epochs=2,
            minibatch_size=4,
            target_kl=None,
        ),
    )
    metrics = agent.update(rollout)
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    for name, value in metrics.items():
        if name != "explained_variance":
            assert math.isfinite(value), name


def test_checkpoint_round_trip(tmp_path):
    """Model, optimiser, settings, and counters must survive saving."""

    torch.manual_seed(23)
    model = ActorCritic()
    agent = PPOAgent(model, PPOHyperparameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        agent,
        training_state={"total_timesteps": 123, "update_index": 4},
        metadata={"purpose": "unit test"},
    )
    loaded_agent, payload = load_checkpoint(
        path,
        device="cpu",
        load_optimizer=True,
    )
    for original, restored in zip(
        agent.model.parameters(),
        loaded_agent.model.parameters(),
    ):
        assert torch.equal(original, restored)
    assert payload["training_state"]["total_timesteps"] == 123
    assert payload["metadata"]["purpose"] == "unit test"


def test_configuration_rejects_unknown_ppo_key():
    """Typos in YAML must fail instead of silently changing training."""

    with pytest.raises(ValueError, match="Unknown PPO settings"):
        PPOHyperparameters.from_mapping({"lerning_rate": 0.001})


def test_cpu_device_selection_is_explicit():
    """CPU mode must work even on a machine with no visible GPU."""

    assert choose_device("cpu") == torch.device("cpu")
