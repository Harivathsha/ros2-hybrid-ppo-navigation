# Deterministic Level-2 evaluation

- Checkpoint source: `snapshot_route_fix_260k.pt`
- Published model name: `ppo_route_guided_263443_steps.pt`
- Checkpoint training state: 263,443 timesteps, PPO update 125
- Policy mode: deterministic (`tanh(actor_mean)`)
- Curriculum level: 2, with six randomized movable obstacles
- Seeds: `90000` through `90029`
- Episodes: 30

## Aggregate results

| Metric | Value |
|---|---:|
| Successes | 25 / 30 |
| Collisions | 5 / 30 |
| Timeouts | 0 / 30 |
| Success rate | 83.33% |
| Collision rate | 16.67% |
| Timeout rate | 0% |
| Mean return | 57.2809 |
| Mean episode length | 282.2 steps |
| Mean final distance | 0.9791 m |
| Mean minimum LiDAR clearance | 0.4353 m |
| Mean successful-path efficiency | 0.8820 |
| Mean stuck-step fraction | 0.0 |

## Interpretation

The checkpoint completed both one-doorway (`route=2/2`) and two-doorway
(`route=4/4`) routes. All five failures were collision terminations; there were
no timeouts and no measured stuck steps. The result demonstrates reliable but
not collision-free navigation on held-out randomized layouts.

This summary is derived from `evaluation_summary.json` and the 30 printed
episode outcomes. The machine-readable JSON and CSV files remain the canonical
evidence.
