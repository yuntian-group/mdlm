# Generation experiment series

This archive organizes work above the retained recovery baseline, through the
completed 8/16/32-step checkpoint sweeps. Results are historical observations.
Commit bodies use a short title and three bullets: change, result and settings.

## Read the experiments in order

| Stage | Study | Scored sample count | Sampler |
|---|---|---|---|
| 01 | [Matched four arms](01-matched-four-arm-baseline.md) | 10 per arm/result | O |
| 02 | [Training continuation](02-continuations-and-checkpoint-screens.md) | 2 per paired cell; separate 1-sample screens | O |
| 03 | [Endpoints and conditioners](03-endpoints-and-conditioners.md) | 1 per reported result | O |
| 04 | [First sampling optimization](../ccf-sampling-speed-fix.md) | Synthetic timing, 3 repeats/state | V1 |
| 05 | [Separate-rank screens](05-separate-rank-screens.md) | 1 per result, 8 in table | V1 |
| 06 | [Learning-rate decay](../ccf-lr-decay.md) | 2 per arm, 8 total | O snapshot |
| 07 | [Second sampling optimization](../ccf-sampling-speed-fix-v2.md) | 5 timing repeats/state; 1 per follow-up PPL, 14 cells | V2 |
| 08 | [Level batching and profiling](../ccf-fast-generation.md) | 2 per implementation/comparison | L and named candidates |
| 09 | [Fresh seeds and objective diagnostic](../ccf-quality-diagnosis-20260916.md) | 9 fresh samples; toy probe generates no text | V2 |
| 10 | [Five-sample evaluations](10-five-sample-generation.md) | 5 per cell, 28 complete cells, 140 total | L |
| 11 | [Basic four arms at 16/32 steps](11-basic-low-step-evaluation.md) | 20 per cell, 10 cells, 200 total | L |
| 12 | [All checkpoints at 16/32 steps](12-full-low-step-sweep.md) | 20 per cell; 2,080 new + 200 reused | L |
| 13 | [All checkpoints at 8 steps](13-eight-step-sweep.md) | 20 per cell, 1,140 fresh | L |

O = unoptimized; V1/V2 are sequential conservative optimizations; L is the
opt-in level-batched sampler. MDLM cells use the released factorized backbone.

## Completed low-step comparison

Each entry below is **PPL at the best observed training checkpoint**, with
20 generated samples per cell. The checkpoint minima were selected on the same
evaluation seeds and are exploratory. The [complete result JSON](data/low-step-results.json)
includes all 171 cells, per-seed scores, scored token counts, timing, actual model
calls, scorer metadata and equivalence-gate status. It excludes raw text and
private paths, hosts, accounts and job identifiers.

| Model | 8 transitions | 16 transitions | 32 transitions |
|---|---:|---:|---:|
| MDLM | 929.75 | 311.78 | 136.34 |
| Shared R16 DD | 655.33 (3k) | 258.75 (7k) | 132.30 (5k) |
| Shared R16 DF | 835.09 (5k) | 351.65 (5k) | 197.69 (7k) |
| Shared R16 FD | 669.69 (6k) | 221.93 (6k) | 126.83 (2k) |
| Shared R16 SS | 892.74 (1k) | 320.86 (6k) | 178.90 (3k) |
| Separate R8 DD | 668.00 (2k) | 248.14 (6k) | 130.89 (2k) |
| Separate R8 FD | 666.14 (6k) | 248.50 (5k) | 130.76 (6k) |
| Separate R16 DD | 638.40 (6k) | 277.29 (2k) | 151.60 (3k) |
| Separate R16 FD | 665.81 (6k) | 239.69 (5k) | 131.57 (3k) |

At 16 transitions, the selected shared-FD/MDLM PPL ratio is about 0.712
with exploratory paired-bootstrap 95% interval 0.523–0.948. At 32 transitions,
the ratio is about 0.930 with interval 0.699–1.224. Neither interval accounts
for searching checkpoints; these are not confirmatory claims. Bootstrap units
are generation seeds, not independent training runs.

The eight-step setting reduces generation time but substantially worsens PPL.
MDLM remains faster: its 32-step result (PPL 136.34, 0.417 seconds/sample) is
better and faster than the best shared-FD 16-step result (221.93, 0.803 seconds/sample).
Scored lengths differ with EOS, and evaluator PPL is not a complete text-quality measure.

## Reproduction and provenance

See [setup and sampler versions](reproducing.md), [validation](validation.md),
and [coverage of the captured work](coverage.md). Standard algorithm citations
and adapted-code attribution are in [the fast-generation notes](../ccf-fast-generation.md).
The PyTorch license is retained under `third-party/`. This curation adds no new
model-improvement claim or external algorithm implementation.
