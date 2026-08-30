# Checkpoint-compatible multi-room navigation upgrade

## Why this change exists

The original feed-forward PPO policy received the final goal bearing even when
a partition wall blocked the direct route.  It could therefore approach the
correct room, become attracted to a wall corner, and fail to discover the
doorway.  The old `0.30 m` LiDAR collision threshold also ended many episodes
while the chassis was still visibly clear of an obstacle.  This was a virtual
termination margin, not the physical radius of each furniture model.

The supplied training log confirms this second issue: recent collision
clearances concentrate around `0.29-0.30 m`.

## What changed

- LiDAR collision termination is now `0.22 m`.  The furniture meshes and
  Gazebo collision geometry are unchanged.
- Every randomized episode is checked on a `0.10 m` grid after obstacles are
  inflated by the collision clearance.  Unreachable layouts are resampled.
- Different-room routes use doorway approach and exit waypoints.
- The observation's five-value goal/motion branch points to the active doorway
  target until it is crossed, then returns to the final green goal.
- Reaching a doorway waypoint gives a small reward.
- Remaining inside a `0.15 m` area for 80 control steps adds a stuck penalty,
  but does not forcibly end the episode.
- Training logs now show minimum clearance, completed route waypoints, and the
  final-step stuck flag.
- Evaluation trajectories record the active target as well as the final goal.

The actor-critic, observation size (`77`), action size (`2`), PPO checkpoint
format, optimizer state, and YAML hyperparameters are unchanged.  The existing
checkpoint can therefore be used as a warm start.

## Install and build

First, wait for an `update=...` line in the current training terminal and press
`Ctrl+C` once.  The trainer should print a path ending in
`checkpoints/interrupted.pt`.  Preserve that exact pre-upgrade policy:

```bash
cd ~/ai_envs/project/autonomous_bot

cp \
  runs/ppo_navigation/run_20260829_080217_seed7/checkpoints/interrupted.pt \
  runs/ppo_navigation/run_20260829_080217_seed7/checkpoints/baseline_reactive_before_navigation_fix.pt
```

If the current process is not running and `interrupted.pt` does not exist, copy
`latest.pt` to the same baseline filename instead.

Place the corrected ZIP in the workspace root.  Back up the current package,
extract the correction, and build it:

```bash
cd ~/ai_envs/project/autonomous_bot

mkdir -p working_backups/nav_route_fix_20260829
cp -a \
  src/nav_learning \
  working_backups/nav_route_fix_20260829/

unzip -o current_navigation_working_fixed.zip

source ~/ai_envs/robotics_ai/bin/activate
source /opt/ros/jazzy/setup.bash

python3 -m py_compile \
  src/nav_learning/nav_learning/ros_nav_env.py \
  src/nav_learning/nav_learning/train_ppo.py \
  src/nav_learning/nav_learning/evaluate_ppo.py

colcon build --symlink-install --packages-select nav_learning description_new
source install/setup.bash
```

## Test before training

Start one simulator in Terminal 1:

```bash
cd ~/ai_envs/project/autonomous_bot

source ~/ai_envs/robotics_ai/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch description_new launch_sim.launch.py
```

In Terminal 2, source the workspace and run the integration smoke test:

```bash
cd ~/ai_envs/project/autonomous_bot

source ~/ai_envs/robotics_ai/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash

python3 -m nav_learning.random_policy_smoke_test
```

Then visually test the preserved baseline checkpoint using five unseen level-2
episodes:

```bash
python3 -m nav_learning.evaluate_ppo \
  --checkpoint runs/ppo_navigation/run_20260829_080217_seed7/checkpoints/baseline_reactive_before_navigation_fix.pt \
  --levels 2 \
  --episodes-per-level 5 \
  --seed-base 50000 \
  --trajectory-episodes-per-level 5 \
  --device cuda \
  --output-dir runs/ppo_navigation/evaluation_route_fix_before_retraining
```

## Continue training without mixing old and new metrics

Stop evaluation before training.  Never run evaluation and training together,
because both control and reset the same robot.  Use a new run directory while
loading the complete old checkpoint:

```bash
python3 -m nav_learning.train_ppo \
  --config src/nav_learning/config/ppo_training.yaml \
  --resume runs/ppo_navigation/run_20260829_080217_seed7/checkpoints/baseline_reactive_before_navigation_fix.pt \
  --run-dir runs/ppo_navigation/run_20260829_route_fix_seed7 \
  --total-timesteps 600000
```

The first updates may fluctuate because the critic must adapt to doorway
targets and the new reward terms.  Judge progress using deterministic fixed-seed
evaluation, not only the stochastic rolling training success rate.

## What the new console fields mean

```text
clearance=0.247 route=3/4 stuck=0
```

- `clearance`: minimum LiDAR clearance during the episode.
- `route=3/4`: three of four doorway approach/exit waypoints were crossed.
- `stuck=1`: the final observation was inside the stuck window; it does not
  necessarily mean the whole episode was stuck.
