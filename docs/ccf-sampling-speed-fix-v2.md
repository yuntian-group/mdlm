# Sampling optimization V2 and follow-up checkpoint screens

Historical September 16, 2026. V2 builds on V1: joint sampling computes only
needed upward messages/root beliefs and removes duplicate probability checks
already performed by native multinomial. Draw order, categorical calls,
precision, constraints and default full-marginal inference remain unchanged.
Invalid categorical inputs now raise the native PyTorch error.

## Historical isolated GPU benchmark

| Synthetic state | V1 seconds | V2 seconds | Speedup | Timed repetitions/implementation |
|---|---:|---:|---:|---:|
| All active | 1.601726 | 0.897576 | 1.785x | 5 |
| Mixed | 1.161066 | 0.682430 | 1.701x | 5 |
| None, diagnostic | 0.627442 | 0.420247 | 1.493x | 5 |

B2/L1024/K128/R16, synthetic vocabulary 256, warmup excluded, alternating
implementation order. These are isolated sampler timings, not generated-text
latency or PPL. Historical GPU evidence: 14 tests, 54 pinned pre-V1 cases,
six L1024 V1 comparisons, and three real-model update comparisons passed.
That evidence did not establish full-trajectory identity. The native-draw
profiler test was corrected to allow PyTorch's own two scalar checks while
requiring removal of the two duplicate checks in this code.

The V1 verifier now points at the equivalent curated V1 commit available in
this branch. Its required source SHA-256 remains unchanged:
`61892523cb59ad6da842326df3fcf319989d5b9efb50da8d0b039aa99e0a15bf`.
The older audit script is retained as an experimental prototype, not a new
production entry point or a benchmark of the final fast path.

## One-sample follow-up screens, V2 optimized

| Variant | Checkpoint | PPL | Generated samples |
|---|---:|---:|---:|
| Separate R8 FD | 1500 | 145.887 | 1 |
| Separate R8 FD | 2500 | 105.745 | 1 |
| Separate R8 FD | 5500 | 84.784 | 1 |
| Separate R8 FD | 6500 | 141.666 | 1 |
| Separate R8 DD | 1500 | 141.601 | 1 |
| Separate R8 DD | 2500 | 112.212 | 1 |
| Separate R8 DD | 5500 | 104.686 | 1 |
| Separate R8 DD | 6500 | 67.242 | 1 |
| Separate R16 FD | 2000 | 139.666 | 1 |
| Separate R16 FD | 4000 | 55.139 | 1 |
| Separate R16 FD | 6000 | 105.256 | 1 |
| Separate R16 DD | 2000 | 43.867 | 1 |
| Separate R16 DD | 4000 | 63.171 | 1 |
| Separate R16 DD | 6000 | 160.968 | 1 |

Fourteen generated samples total, one per cell: seed 91001, length 1024,
1,000 transitions plus cleanup, batch 1, pinned GPT-2-large float32/EOS scoring.
These are exploratory checkpoint screens; minima are selected on one sample.
Earlier V1 and O screens retain their own labels and are not silently pooled.
Historical node identifiers, login paths and notification addresses are omitted.
No GPU experiments were rerun during commit curation.
