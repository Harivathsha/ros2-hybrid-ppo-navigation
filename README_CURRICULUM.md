# Square-arena curriculum

The runtime is a final-goal PPO policy. Read the [main README](README.md) for all commands.

| Stage | Spawn region | Obstacles |
|---|---|---|
| 1 (level 0) | 3 × 3 m | 0–2 boxes |
| 2 (level 1) | 6 × 6 m | 1, then 2, then 3, then 5–7 boxes |
| 3 (level 2) | 9 × 9 m | 12–15 boxes |

The physical world is always 9 × 9 m. Spawn patches are sampling boundaries, not extra walls. Current-task promotion uses a 100-episode window and separate 10-episode validation, both with an 80% threshold. Failed validation waits 25 eligible training episodes. Earlier-stage rehearsal is excluded from promotion counts.

Training includes 30% corner practice in Stage 2 and actual reflected training episode pairs. The network has 77 inputs and two outputs. Linear velocity is in [0, 0.22] m/s; angular velocity is in [-1.2, 1.2] rad/s. The bottom 15% of the speed-action range permits zero forward speed and turning in place.

The goal radius is 0.25 m. A minimum LiDAR range at or below 0.22 m terminates the episode as a collision; this is a distance-based termination rule, not a contact sensor.

## Reward and training assistance

The reward combines terminal outcomes, step/proximity/stuck costs, a braking cost and a discount-correct detour potential. The detour potential uses known box geometry to calculate a reward-only reference distance. The actor receives no obstacle poses, route, waypoint, preferred side or corrective action.

The braking term penalizes forward motion while the nearest obstacle distance decreases inside the 0.5 m proximity region. It is an additional task cost, not a potential-invariant shaping term. Both proximity and braking costs may influence passage-entry behavior; the evaluation does not establish their causal effect.

## Published experiment

The archived seed-42 experiment changed its curriculum gate and later enabled braking before this checkpoint. The bundled default config matches the final current profile: 700,000-step target, gate 0.8, braking scale 1.0. Starting fresh with this config uses that profile from initialization and is not an exact replay of the historical sequence.

The published checkpoint was interrupted at 687,935 environment steps (update 334). The included continuation training CSV begins at 257,373 steps and ends at completed episode step 687,766; its last completed update is at 687,150 steps. Thus the supplied training figures do not show the entire history from zero.

The 50 deterministic Stage 3 tests produced 32 successes, 7 collisions and 11 timeouts. The policy remains feed-forward and simulation-only. More training is not guaranteed to improve its success rate.

[Detailed code guide](src/nav_learning/README_RL.md) · [Results](docs/results/stage3_687935/README.md)

