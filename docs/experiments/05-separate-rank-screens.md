# Separate-rank screens with sampler V1

Historical September 15/16, 2026. Rank-16 separate factors are a capacity
increase (1,815,201 head parameters), not the parameter-matched R8 comparison
(984,417). Both use K128, a frozen backbone, seed-1 training to 7,000 updates,
and fresh architecture-compatible checkpoints.

| Variant | Training checkpoint | PPL | Generated samples | Sampler |
|---|---:|---:|---:|---|
| Separate R8 FD | 2k | 119.285 | 1 | V1 optimized |
| Separate R8 FD | 4k | 82.832 | 1 | V1 optimized |
| Separate R8 FD | 6k | 62.831 | 1 | V1 optimized |
| Separate R8 DD | 2k | 35.291 | 1 | V1 optimized |
| Separate R8 DD | 4k | 83.489 | 1 | V1 optimized |
| Separate R8 DD | 6k | 35.744 | 1 | V1 optimized |
| Separate R16 FD | 7k | 86.395 | 1 | V1 optimized |
| Separate R16 DD | 7k | 77.440 | 1 | V1 optimized |

Eight generated samples represented in this table. Each uses seed 91001,
length 1024, batch 1, 1,000 transitions plus cleanup, and the pinned GPT-2-large
float32/EOS scoring policy. These are one-sample screens, not reliable model
rankings. The favorable 35.291/35.744 R8 DD outcomes motivated independent-seed
checks; same-seed cross-hardware discrepancies remained unresolved in later
historical replay attempts. Do not pool these selected samples with fresh
multi-sample evaluations.

The generic rank-8/rank-16 runner and its restart checks are shared. No older
checkpoint is silently resized or resumed into a different rank. Archived
paths have portable aliases and require local checkpoint/cache configuration.
The result labels describe the historical code snapshot, not a promise that
running the same launcher at a later branch HEAD recreates that implementation.
