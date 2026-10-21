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


@pytest.mark.parametrize(("log_std", "gradient_sign"), [(-2.0, 1), (0.8, -1)])
def test_entropy_gradient_respects_bounded_actions(log_std, gradient_sign):
    """Entropy must discourage excessive saturation, unlike Gaussian entropy."""

    torch.manual_seed(91)
    model = ActorCritic()
    with torch.no_grad():
        model.actor_mean.weight.zero_()
        model.actor_mean.bias.zero_()
        model.actor_log_standard_deviation.fill_(log_std)
    observations = make_observations(1024)
    actions = torch.zeros((1024, ACTION_SIZE), dtype=torch.float32)
    _, entropy, _ = model.evaluate_action_with_entropy(observations, actions)
    entropy.mean().backward()
    gradient = model.actor_log_standard_deviation.grad
    assert torch.isfinite(entropy).all()
    assert torch.all(gradient * gradient_sign > 0.3)
    if log_std > 0:
        assert float(entropy.mean()) < 0.0


def test_tuned_resume_applies_yaml_and_preserves_learned_policy(tmp_path):
    """Old Adam settings must not silently override the new tuning profile."""

    from dataclasses import replace
    from pathlib import Path
    from nav_learning.nav_math import ENVIRONMENT_ID
    from nav_learning.train_ppo import load_training_settings, _resume_training_agent

    settings = load_training_settings(Path(__file__).parents[1] / "config/ppo_training.yaml")
    settings = replace(settings, ppo=replace(settings.ppo, adam_epsilon=3e-6))
    agent = PPOAgent(ActorCritic(), PPOHyperparameters(update_epochs=1, minibatch_size=4))
    agent.update(make_rollout(agent.model))
    with torch.no_grad():
        agent.model.actor_log_standard_deviation.fill_(0.4)
    before = {name: p.detach().clone() for name, p in agent.model.named_parameters()}
    old_mean_actions = agent.model.deterministic_action(make_observations())
    original_state = {
        "total_timesteps": 254218, "update_index": 123, "episode_index": 1851,
        "next_reset_seed": 11852, "curriculum_level": 1,
        "recent_successes": [1] * 53 + [0] * 47,
        "recent_episode_returns": [30, -20], "promotion_wait_remaining": 10,
    }
    source = save_checkpoint(tmp_path / "legacy.pt", agent, original_state,
                             metadata={"environment_id": ENVIRONMENT_ID})
    restored, state = _resume_training_agent(settings, source, torch.device("cpu"), True)
    assert restored.hyperparameters == settings.ppo
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert restored.optimizer.param_groups[0]["eps"] == pytest.approx(3e-6)
    assert torch.equal(old_mean_actions, restored.model.deterministic_action(make_observations()))
    for name, parameter in restored.model.named_parameters():
        if name == "actor_log_standard_deviation":
            assert torch.all(parameter == -0.5)
            assert parameter not in restored.optimizer.state
        else:
            assert torch.equal(parameter, before[name])
            old_parameter = dict(agent.model.named_parameters())[name]
            for key, old_value in agent.optimizer.state[old_parameter].items():
                assert torch.equal(old_value, restored.optimizer.state[parameter][key])
    for key in ("total_timesteps", "update_index", "episode_index", "next_reset_seed", "curriculum_level"):
        assert state[key] == original_state[key]
    assert state["recent_successes"] == []
    assert state["promotion_wait_remaining"] == 0
    metrics = restored.update(make_rollout(restored.model))
    assert math.isfinite(metrics["squashed_entropy"])


def test_same_tuned_checkpoint_resumes_without_resetting_progress(tmp_path):
    """A normal subsequent resume keeps its new promotion window and noise."""

    from dataclasses import asdict
    from pathlib import Path
    from nav_learning.nav_math import ENVIRONMENT_ID
    from nav_learning.train_ppo import TRAINING_REVISION, load_training_settings, _resume_training_agent

    settings = load_training_settings(Path(__file__).parents[1] / "config/ppo_training.yaml")
    agent = PPOAgent(ActorCritic(), settings.ppo)
    state = dict(total_timesteps=260000, update_index=126, episode_index=1890,
                 next_reset_seed=11891, curriculum_level=1, recent_successes=[1, 0, 1],detour_phase=2)
    path = save_checkpoint(tmp_path / "tuned.pt", agent, state, metadata={
        "environment_id": ENVIRONMENT_ID, "training_revision": TRAINING_REVISION,
        "reward_settings": asdict(settings.reward),
        "action_settings": asdict(settings.action),
        "curriculum_settings": asdict(settings.curriculum),
        "detour_practice": settings.curriculum.detour_practice,
    })
    restored, new_state = _resume_training_agent(settings, path, torch.device("cpu"))
    assert new_state["recent_successes"] == [1, 0, 1]
    assert new_state["detour_phase"] == 2
    assert torch.equal(restored.model.actor_log_standard_deviation,
                       agent.model.actor_log_standard_deviation)


def test_old_yaml_defaults_to_legacy_reward_settings(tmp_path):
    """Existing run configs without a reward section remain readable."""

    from pathlib import Path
    import yaml
    from nav_learning.nav_math import RewardSettings
    from nav_learning.train_ppo import load_training_settings

    values = yaml.safe_load((Path(__file__).parents[1] / "config/ppo_training.yaml").read_text())
    values.pop("reward")
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(values))
    assert load_training_settings(path).reward == RewardSettings()


@pytest.mark.parametrize("wrong_steering_sign", [False, True])
def test_live_preflight_checks_stationary_turn_signs(monkeypatch, wrong_steering_sign):
    """Exercise the live-check gate offline; real Gazebo still needs this check."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace, ModuleType
    from nav_learning import train_ppo
    from nav_learning.nav_math import policy_action_to_velocity

    settings = train_ppo.load_training_settings(Path(__file__).parents[1] / 'config/ppo_training.yaml')
    class Environment:
        closed = False
        def __init__(self, **kwargs):
            self.action_settings = kwargs['action_settings']
            self.simulation_time_ns = 0
            self.x = self.yaw = 0.0
        def reset(self, **kwargs):
            return None, {}
        def step(self, action):
            v, w = policy_action_to_velocity(action, self.action_settings)
            self.simulation_time_ns += 100_000_000
            self.x += v * .1
            self.yaw += w * .1 * (-1 if wrong_steering_sign else 1)
            return None, 0, False, False, dict(simulation_interval_sec=.1,
                maximum_sensor_age_sec=.01, robot_x=self.x, robot_y=0, robot_yaw=self.yaw)
        def assert_simulation_time_unchanged(self, expected):
            assert expected == self.simulation_time_ns
        def publish_stop(self): pass
        def pause_simulation(self): pass
        def close(self): Environment.closed = True
    ros = ModuleType('rclpy')
    ros.init = lambda **kwargs: None
    ros.ok = lambda: True
    ros.shutdown = lambda: None
    env_module = ModuleType('nav_learning.ros_nav_env')
    env_module.RosNavEnv = Environment
    monkeypatch.setitem(sys.modules, 'rclpy', ros)
    monkeypatch.setitem(sys.modules, 'nav_learning.ros_nav_env', env_module)
    monkeypatch.setattr(train_ppo.time, 'sleep', lambda seconds: None)
    if wrong_steering_sign:
        with pytest.raises(RuntimeError, match='left turn failed'):
            train_ppo._run_timing_check(settings)
    else:
        train_ppo._run_timing_check(settings)
    assert Environment.closed


def test_potential_discount_must_match_ppo(tmp_path):
    from pathlib import Path
    import yaml
    from nav_learning.train_ppo import load_training_settings
    values = yaml.safe_load((Path(__file__).parents[1] / 'config/ppo_training.yaml').read_text())
    values['reward']['discount_factor'] = 0.95
    path = tmp_path / 'bad_discount.yaml'
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='same discount_factor'):
        load_training_settings(path)


@pytest.mark.parametrize("mode", ["tuned_resume", "normal_resume", "smoke", "defaults"])
def test_trainer_cli_preserves_nested_reward_settings(tmp_path, monkeypatch, mode):
    """Exercise main(), including YAML -> overrides -> config snapshot -> trainer."""

    from pathlib import Path
    from nav_learning import train_ppo
    from nav_learning.nav_math import RewardSettings

    config = Path(__file__).parents[1] / "config/ppo_training.yaml"
    run = tmp_path / "run"
    arguments = ["--config", str(config), "--run-dir", str(run)]
    source = tmp_path / "source.pt"
    source.touch()
    if mode in ("tuned_resume", "normal_resume"):
        arguments += ["--resume", str(source), "--total-timesteps", "300000", "--device", "cuda"]
    if mode == "tuned_resume":
        arguments += ["--reset-exploration"]
    if mode == "smoke":
        arguments += ["--smoke", "--device", "cpu"]
    calls = []
    monkeypatch.setattr(train_ppo, "run_training", lambda **kwargs: calls.append(kwargs))
    train_ppo.main(arguments)
    assert len(calls) == 1
    actual = calls[0]
    settings = actual["settings"]
    assert settings.reward == RewardSettings(0.5, 0.2, progress_mode="detour_potential")
    assert settings.action.stop_fraction == 0.15
    assert settings.ppo.learning_rate == pytest.approx(0.0001)
    assert settings.ppo.entropy_coefficient == pytest.approx(0.001)
    assert actual["reset_exploration"] is (mode == "tuned_resume")
    assert actual["fixed_level"] == (0 if mode == "smoke" else None)
    assert train_ppo.load_training_settings(run / "config_used.yaml") == settings
    if mode in ("tuned_resume", "normal_resume"):
        assert actual["resume_path"] == source
        assert settings.total_timesteps == 300000
        assert settings.device == "cuda"
    if mode == "smoke":
        assert settings.total_timesteps == 256
        assert settings.rollout_steps == 128


def test_cli_overrides_still_reject_invalid_totals():
    """Recursive serialization must retain full validation after overrides."""

    from pathlib import Path
    from nav_learning.train_ppo import _apply_overrides, _parse_arguments, load_training_settings

    settings = load_training_settings(Path(__file__).parents[1] / "config/ppo_training.yaml")
    with pytest.raises(ValueError, match="total_timesteps must be positive"):
        _apply_overrides(settings, _parse_arguments(["--total-timesteps", "-1"]))


def test_timing_check_cli_bypasses_training_and_run_creation(tmp_path, monkeypatch):
    from pathlib import Path
    from nav_learning import train_ppo
    config=Path(__file__).parents[1]/'config/ppo_training.yaml'
    checks=[]
    monkeypatch.setattr(train_ppo,'_run_timing_check',lambda settings:checks.append(settings))
    monkeypatch.setattr(train_ppo,'run_training',lambda **kwargs:pytest.fail('Training should not start'))
    run=tmp_path/'unused'
    train_ppo.main(['--config',str(config),'--timing-check','--run-dir',str(run),'--device','cuda'])
    assert len(checks)==1 and checks[0].curriculum.detour_practice
    assert not run.exists()
