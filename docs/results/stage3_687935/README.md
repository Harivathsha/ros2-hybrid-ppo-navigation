# Stage 3 evaluation

| Item | Value |
|---|---:|
| Checkpoint environment steps | 687,935 |
| Last PPO update | 334 |
| Training budget | 700,000 |
| Episodes | 50 |
| Success | 32 (64%) |
| Collision | 7 (14%) |
| Timeout | 11 (22%) |
| Arena / boxes | 9 × 9 m / 12–15 |
| Deterministic seeds | 140000–140049 |
| Mean episode length | 860.76 steps |
| Mean minimum LiDAR range | 0.3474 m |

This benchmark evaluates Stage 3 only, with `seed_base=120000` and `tanh(actor_mean)` actions. It does not establish easier-stage performance, reliable passage crossing or hardware performance. The old three-room benchmark is a different task and is not directly comparable.

## Evidence

- [Summary](evaluation_summary.json), [all episode outcomes](evaluation_episodes.csv), [layouts](evaluation_layouts.json), [all trajectories](evaluation_trajectories.csv).
- [Original run configuration](config_used.yaml), [training episodes](metrics/train_episodes.csv), [PPO updates](metrics/train_updates.csv), [promotion validation](metrics/promotion_checks.csv).
- [Training curves](../../plots/training_curves.png) and [PPO diagnostics](../../plots/ppo_diagnostics.png).
- [Evidence checksums](SHA256SUMS), [published checkpoint checksum](../../../models/SHA256SUMS) and [checkpoint metadata](../../../models/checkpoint_metadata.json).

The figures cover the later braking continuation, not all training from initialization. Curriculum transitions reset the rolling success window; early 100% values after a reset do not indicate sustained perfect performance.

Timeout cases average approximately 63.5% of steps flagged as stuck by the low-translation criterion. Hesitation, collisions and timeouts remain limitations. A controlled gap benchmark is needed to diagnose passage failures.

The two main README environment views are reproducible overhead reconstructions of recorded seeds 140000 (success) and 140002 (timeout). They are labelled as reconstructions, not Gazebo screenshots.

The uploaded checkpoint matches the evaluator's recorded step/update counters and reward/action profiles. Its SHA-256 identifies the supplied model; the original evaluation summary did not record a model hash. Checkpoint bytes and supplied evaluation data are preserved.

