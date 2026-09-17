# Twenty-sample basic four-arm evaluation at 16/32 transitions

Historical evaluation completed September 16–17, 2026. Saved results, not a new run. All 10 cells completed on the same H200 NVL host. All200 scored samples completed,20 per cell. All8 CCF same-GPU
reference/optimized first-sample comparisons passed final-token, NFE and
CPU/CUDA RNG checks. No unresolved mask tokens were reported.

CCF sampler: **L, level-batched optimized**. CCF: shared rank16, constant-LR training-update7000 checkpoints. MDLM:
released frozen backbone in factorized mode. Length1024, batch1, matched
seeds91001–91020. All10 cells have the same pairing digest.

| Model | 16-step PPL | 32-step PPL | 16-step seconds/sample | 32-step seconds/sample |
|---|---:|---:|---:|---:|
| MDLM | 311.783 | 136.340 | 0.316 | 0.417 |
| CCF static/static | 330.729 | 191.064 | 0.798 | 1.588 |
| CCF fixed/dynamic | 250.329 | 149.135 | 0.798 | 1.609 |
| CCF dynamic/fixed | 396.017 | 197.694 | 1.028 | 2.113 |
| CCF dynamic/dynamic | 258.746 | 165.599 | 1.052 | 2.078 |

PPL is exp(token-weighted GPT2-large mean NLL), using the unchanged
first-nonleading-EOS policy. Times are generation only, excluding loading,
verification duplicates and GPT2 scoring. Actual model calls were17/33 for
MDLM and16/32 for CCF, because of the existing cleanup/early-stop behavior.
Thus these are matched reverse-transition settings, not identical measured
model-call counts or wall-clock budgets.

| Model | 16-step scored tokens | 32-step scored tokens | 16-step mean repeated 4-gram rate | 32-step mean repeated 4-gram rate |
|---|---:|---:|---:|---:|
| MDLM | 11040 | 8717 | 0.255% | 0.637% |
| CCF static/static | 10124 | 11025 | 0.328% | 0.500% |
| CCF fixed/dynamic | 13176 | 8535 | 0.632% | 0.975% |
| CCF dynamic/fixed | 12318 | 11862 | 0.269% | 0.583% |
| CCF dynamic/dynamic | 10113 | 12994 | 0.573% | 0.544% |

At16 steps, both contextual-factor CCF arms have lower aggregate PPL than
MDLM; at32 steps, MDLM has lower aggregate PPL than every CCF arm. This initial report makes no confidence-interval or significance claim; later
checkpoint-sweep analysis is reported separately.
Scored lengths vary because of EOS, and lower evaluator PPL alone does not
establish better overall text quality.


These 10 cells (including both MDLM baselines) were reused in the later
16/32-step checkpoint sweep. Reuse means the same generated samples and scores,
not a new baseline run or extra samples. The 8-step MDLM baseline was fresh.
