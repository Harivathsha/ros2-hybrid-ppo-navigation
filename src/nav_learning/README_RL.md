# Read the code in this order

There are still only **six functional runtime Python files**. The empty
`__init__.py` and `resource/nav_learning` are Python/ROS registration markers.
Setup files describe how ROS installs the package; they are not extra robots or
controllers. The three files under `test/` are offline checks.

| File | What to read first | What it does |
| --- | --- | --- |
| `nav_math.py` | `STAGES`, `sample_episode`, `DetourDistance`, `calculate_reward` | Task geometry, action conversion and reward calculations. No ROS/PyTorch. |
| `ros_nav_env.py` | `reset`, `_apply_action_and_observe`, `step` | Moves objects at reset, publishes the policy command, advances simulation, reads sensors. |
| `ppo.py` | `ActorCritic`, `RolloutBuffer`, `PPOAgent.update` | Neural network, stored experience, GAE and PPO optimization. Unchanged in this revision. |
| `train_ppo.py` | `TrainingSettings`, `_reset_episode`, `run_training` | Loads YAML, collects episodes, updates PPO and checks promotion. |
| `evaluate_ppo.py` | `run_evaluation` | Loads a checkpoint, runs fixed seeds without learning, saves paths and scores. |
| `plot_ppo_results.py` | `main` | Turns the CSV/JSON results into five plots. |

## One training step

1. `build_observation()` gives the network 72 laser bins and 5 goal/motion values.
2. PPO samples two normalized actions: forward-speed choice and turn choice.
3. `policy_action_to_velocity()` converts them to a Twist. With stop_fraction
   0.15, speed actions from -1 to -0.70 produce zero forward speed. The angular
   action independently selects left/right turning; it is never overridden.
4. Gazebo advances exactly 0.1 simulation seconds, then pauses.
5. New LiDAR and odometry give the next observation and episode outcome.
6. `calculate_reward()` scores that transition. The rollout retains the ORIGINAL
   sampled action and its log probability, even inside the zero-speed range.
7. After a rollout, PPO updates the network while Gazebo stays paused.

The policy has no recurrent memory. The final goal direction is supplied even
behind a box. Training must teach it that the goal direction is not always the
safe direction of immediate travel.

## Why the new detour reward helps

The old progress term was `5 * (previous_goal_distance - current_goal_distance)`.
That can give a negative immediate reward when a useful detour first moves away
from the final goal.

`DetourDistance` builds a small geometric graph from the goal and corners of
inflated boxes. Dijkstra computes costs from those corners to the goal at reset.
Each step queries a single distance number from the robot position. The preferred
margin is 0.30 m. If the robot enters that margin, a circular-footprint escape
check keeps the potential defined without requiring square corner clearance.
Distances are capped at 20 m so terminal shaping cannot outweigh a collision.

The potential is `Phi = -min(detour_distance, 20)` and the additional reward is:

```text
progress_reward = gamma * Phi(next_state) - Phi(current_state)
```

Success and collision are true terminal states, so their next potential is zero.
Timeouts are truncations: PPO bootstraps their next value, and shaping retains the
next potential. Gamma must match the YAML PPO discount factor. Obstacle, step,
stuck and terminal reward terms are still added separately.

The graph is ONLY a reward calculator. Its values and nodes are not put in the
77-input observation, and it does not publish velocity or intermediate goals.
Evaluation actions depend on the policy observation, not evaluation rewards.

## Corner tasks and reflection

Ordinary benchmark sampling is unchanged for the same seeds. Training mixes
about 30% corner tasks into Stage 2. A corner task places the robot facing a box
face and the goal beyond a nearby corner, with other boxes still randomized.
All sampled tasks must pass the existing clearance and reachability checks.

With mirror_training enabled, even/odd seed pairs run the same layout reflected
across the x-axis: `(x, y, yaw)` becomes `(x, -y, -yaw)`. These are actual episodes,
so PPO still learns from its own current policy. No synthetic mirrored actions
or demonstrations are inserted into the rollout buffer. This encourages exposure
to both directions, without requiring left/right turn counts to be equal.

## Checkpoints and testing

Old checkpoints default to the original action mapping and Euclidean reward.
New checkpoints save their new action/reward profiles. The same network shape
lets us continue learned weights. A changed training profile clears the promotion
score window once; a normal same-profile resume keeps it.

`stage_success=0.85 wins=85/100` means 85 successes in the last 100 eligible tasks,
not 85% of the weights learned. Rehearsal is excluded. Practice phases need
90/100 plus 9/10 held-out validation episodes; full stages use 80/100 plus 8/10.
Training phases are one box, two boxes, three boxes, then full 5-7-box Stage 2.

Use `--task-suite corners --levels 1 --practice-phase 2` to test paired three-box
corners. Without those practice flags, the default evaluator uses the original
stage distributions. Trajectory CSV files contain yaw, commanded velocities,
goal bearing and nearest-obstacle distance to help diagnose corner failures.

Installation, exact commands and research references are in
[README_CURRICULUM.md](../../README_CURRICULUM.md).
