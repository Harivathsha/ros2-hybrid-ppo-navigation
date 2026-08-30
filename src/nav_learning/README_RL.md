# Complete standalone PPO navigation phase

> For the checkpoint-compatible collision, connected-layout, and doorway
> routing changes, start with `NAVIGATION_UPGRADE.md`.

This package contains the standalone PPO implementation and the upgraded
`RosNavEnv`.  It does not use Stable-Baselines3, RLlib, CleanRL, TorchRL, or a
Gym wrapper.

The policy receives the existing 77-value observation:

- 72 pooled, normalised LiDAR values
- normalised goal distance
- sine and cosine of relative goal bearing
- normalised linear and angular velocity

It outputs two values in `[-1, 1]`. `RosNavEnv` keeps responsibility for mapping
them to linear and angular robot velocity, calculating reward, detecting goal or
collision termination, applying timeout truncation, and randomising each reset.

## Main files in this package

| File | Responsibility |
|---|---|
| `nav_learning/ppo.py` | Actor-critic, tanh Gaussian, rollout buffer, GAE, clipped PPO losses, optimiser, checkpoints, ROS-free smoke test |
| `nav_learning/rl_config.py` | Strict YAML parsing and validation |
| `nav_learning/train_ppo.py` | ROS rollout collection, curriculum, logs, checkpoints, safe resume |
| `nav_learning/evaluate_ppo.py` | Deterministic fixed-seed evaluation and trajectory recording |
| `nav_learning/plot_ppo_results.py` | Training, PPO-diagnostic, evaluation, and trajectory figures |
| `nav_learning/ros_nav_env.py` | ROS sensors, connected resets, doorway targets, rewards, and termination |
| `config/ppo_training.yaml` | Full-run settings and PPO hyperparameters |
| `test/test_ppo_math.py` | Mathematical and checkpoint unit tests without ROS/Gazebo |
| `setup.py`, `package.xml` | Console commands and installed YAML configuration |

## 1. Copy the package into the workspace

Make a backup of the PPO checkpoint file you already wrote if you want to keep
it for comparison:

```bash
cd ~/ai_envs/project/autonomous_bot

cp \
src/nav_learning/nav_learning/ppo.py \
src/nav_learning/nav_learning/ppo_checkpoint_2_backup.py
```

Extract the supplied ZIP at the workspace root. Its paths begin with
`src/nav_learning`, so the files land in the correct package:

```bash
cd ~/ai_envs/project/autonomous_bot

unzip -o \
~/Downloads/current_navigation_working_fixed.zip \
-d ~/ai_envs/project/autonomous_bot
```

Confirm that the environment file still exists and that the new files arrived:

```bash
cd ~/ai_envs/project/autonomous_bot

ls src/nav_learning/nav_learning/ros_nav_env.py

ls \
src/nav_learning/nav_learning/ppo.py \
src/nav_learning/nav_learning/rl_config.py \
src/nav_learning/nav_learning/train_ppo.py \
src/nav_learning/nav_learning/evaluate_ppo.py \
src/nav_learning/nav_learning/plot_ppo_results.py \
src/nav_learning/config/ppo_training.yaml \
src/nav_learning/test/test_ppo_math.py
```

## 2. Activate the same Python environment and build

Build while `robotics_ai` is active so the generated ROS console scripts use the
Python environment that already contains your CUDA-enabled PyTorch build.

```bash
cd ~/ai_envs/project/autonomous_bot

source ~/ai_envs/robotics_ai/bin/activate
source /opt/ros/jazzy/setup.bash

which python3

python3 - <<'PY'
import matplotlib
import numpy
import torch
import yaml

print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", numpy.__version__)
print("PyYAML:", yaml.__version__)
print("Matplotlib:", matplotlib.__version__)
PY

colcon build --symlink-install \
--packages-select nav_learning

source install/setup.bash
```

If only `yaml`, `matplotlib`, or `pytest` is missing, install the lightweight
dependencies without touching your existing PyTorch installation:

```bash
python3 -m pip install \
PyYAML matplotlib pytest
```

Do not reinstall PyTorch merely because `CUDA available` is false. The config
uses `device: auto`, so it will safely use CPU until the driver visibility issue
is fixed. With one Gazebo environment and a small 72,773-parameter network,
environment stepping is usually the main speed limit rather than neural-network
computation.

## 3. Test PPO without ROS or Gazebo

Compile every new Python file:

```bash
cd ~/ai_envs/project/autonomous_bot

python3 -m py_compile \
src/nav_learning/nav_learning/ppo.py \
src/nav_learning/nav_learning/rl_config.py \
src/nav_learning/nav_learning/train_ppo.py \
src/nav_learning/nav_learning/evaluate_ppo.py \
src/nav_learning/nav_learning/plot_ppo_results.py \
src/nav_learning/test/test_ppo_math.py
```

Run the self-contained PPO smoke test:

```bash
source install/setup.bash
python3 -m nav_learning.ppo
```

The important lines are:

```text
Complete PPO core smoke test passed.
trainable_parameters=72773
sample_log_probability_reevaluation=True
terminated_bootstrap_masked=True
truncated_bootstrap_preserved=True
gae_hand_calculation_match=True
ppo_gradient_update=True
```

Run the focused unit tests:

```bash
python3 -m pytest \
src/nav_learning/test/test_ppo_math.py \
-q
```

These tests do not need ROS or Gazebo. They validate the locked architecture,
bounded action probability round trip, terminal and timeout masks, clipped PPO
surrogate, real gradient update, YAML typo protection, and checkpoint reload.

## 4. Start the simulator

Use two terminals. In Terminal 1:

```bash
cd ~/ai_envs/project/autonomous_bot

source ~/ai_envs/robotics_ai/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch description_new launch_sim.launch.py
```

Wait until the robot, world, bridges, LiDAR, odometry, and set-pose service are
ready. Do not launch a second simulator for training.

## 5. Run the real 256-step integration smoke test

In Terminal 2:

```bash
cd ~/ai_envs/project/autonomous_bot

source ~/ai_envs/robotics_ai/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run nav_learning train_ppo --smoke
```

This creates a new timestamped directory under `runs/ppo_navigation`, collects
256 real Gazebo transitions at curriculum level 0, performs two PPO updates,
writes CSV logs, and saves checkpoints. It is an integration test, not enough
training for the policy to learn navigation.

Success means the command reaches a message like:

```text
Training complete. Final checkpoint: .../checkpoints/final.pt
```

Check the latest created run:

```bash
ls -dt runs/ppo_navigation/run_* | head -n 1
```

## 6. Start the full curriculum run

Keep Terminal 1 running. In Terminal 2:

```bash
cd ~/ai_envs/project/autonomous_bot

source ~/ai_envs/robotics_ai/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run nav_learning train_ppo \
--config src/nav_learning/config/ppo_training.yaml
```

The default run collects 600,000 transitions. It starts at level 0 and promotes
only when the rolling success rate reaches 70% over the configured evidence
window. Promotion history is preserved in checkpoints.

For a controlled experiment that never promotes, add one of:

```bash
--fixed-level 0
--fixed-level 1
--fixed-level 2
```

For example, a 50,000-step level-0-only run is:

```bash
ros2 run nav_learning train_ppo \
--fixed-level 0 \
--total-timesteps 50000
```

Every run receives a separate timestamped directory. The terminal prints its
exact path at startup.

## 7. Stop safely and resume

Press `Ctrl+C` once in the trainer terminal. The trainer stops the robot during
environment cleanup and writes:

```text
checkpoints/interrupted.pt
```

Resume with a target larger than the checkpoint's current step count:

```bash
ros2 run nav_learning train_ppo \
--resume runs/ppo_navigation/RUN_NAME/checkpoints/interrupted.pt \
--total-timesteps 600000
```

For a normally running job, use `latest.pt`:

```bash
ros2 run nav_learning train_ppo \
--resume runs/ppo_navigation/RUN_NAME/checkpoints/latest.pt \
--total-timesteps 800000
```

Resume restores the actor, critic, learned exploration standard deviations,
Adam optimiser, PPO settings, step/update/episode counters, curriculum level,
recent curriculum evidence, next reset seed, and random-number states. A new
environment episode is reset because a Gazebo physical state is not stored in a
PyTorch checkpoint.

## 8. Evaluate on fixed unseen seeds

Keep the simulator running and ensure no trainer is commanding the robot. Then:

```bash
cd ~/ai_envs/project/autonomous_bot

source ~/ai_envs/robotics_ai/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run nav_learning evaluate_ppo \
--checkpoint runs/ppo_navigation/RUN_NAME/checkpoints/final.pt \
--levels 0 1 2 \
--episodes-per-level 30 \
--seed-base 50000
```

Evaluation is deterministic by default: it uses `tanh(actor_mean)` rather than
exploration noise. The seeds start at 50,000 and are deliberately separate from
the training reset seed range.

It writes:

```text
runs/ppo_navigation/RUN_NAME/evaluation/evaluation_episodes.csv
runs/ppo_navigation/RUN_NAME/evaluation/evaluation_trajectories.csv
runs/ppo_navigation/RUN_NAME/evaluation/evaluation_summary.json
```

The summary includes per-level and overall success, collision, and timeout
rates, return, episode length, final goal distance, minimum clearance, and
successful-path efficiency.

## 9. Generate graphs

Gazebo is not needed for plotting:

```bash
cd ~/ai_envs/project/autonomous_bot

source ~/ai_envs/robotics_ai/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run nav_learning plot_ppo_results \
--run-dir runs/ppo_navigation/RUN_NAME
```

Available data produces up to four images:

```text
plots/training_curves.png
plots/ppo_diagnostics.png
plots/evaluation_summary.png
plots/evaluation_trajectories.png
```

## Output structure

```text
runs/ppo_navigation/RUN_NAME/
├── config_used.yaml
├── checkpoints/
│   ├── latest.pt
│   ├── update_000010.pt
│   └── final.pt
├── metrics/
│   ├── train_episodes.csv
│   └── train_updates.csv
├── evaluation/
│   ├── evaluation_episodes.csv
│   ├── evaluation_trajectories.csv
│   └── evaluation_summary.json
└── plots/
    ├── training_curves.png
    ├── ppo_diagnostics.png
    ├── evaluation_summary.png
    └── evaluation_trajectories.png
```

## What proves the RL phase is complete

Code execution alone is not evidence of learning. Consider standalone PPO
complete only after all of these hold:

1. The ROS-free smoke test and unit tests pass.
2. The 256-step Gazebo integration smoke test produces a checkpoint and logs.
3. Training curves improve rather than merely remaining finite.
4. Fixed unseen-seed evaluation reports strong success with acceptably low
   collision rate at all required curriculum levels.
5. Recorded trajectories show plausible, collision-free paths to the goal.

Behavior cloning, Nav2 demonstration collection, and PPO+BC comparison belong
to the later imitation-learning phase and are intentionally not included here.
