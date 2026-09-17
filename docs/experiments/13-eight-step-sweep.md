# Full checkpoint sweep at eight sampling transitions

Historical evaluation completed September 17, 2026. The same eight CCF configurations
were evaluated at training updates 1,000–7,000, plus a **fresh 8-step MDLM baseline**.
All 57 cells completed with **20 generated samples per cell: 1,140 new samples**.
Seeds 91001–91020, length 1,024, batch 1, GPT2-large scoring, same H200 NVL host.
CCF used **L, level-batched optimized** sampling. All 56 first-sample comparison gates passed;
the verified optimized sample counts toward each cell’s 20 samples.

| Family / arm | Training update | 8-step PPL | Generation seconds/sample | Actual model calls |
|---|---:|---:|---:|---|
| MDLM | — | 929.75 | 0.134 | 9 |
| Shared R16 DD | 1000 | 771.32 | 0.497 | 8 |
| Shared R16 DD | 2000 | 780.60 | 0.515 | 8 |
| Shared R16 DD | 3000 | 655.33 | 0.518 | 8 |
| Shared R16 DD | 4000 | 785.89 | 0.514 | 8 |
| Shared R16 DD | 5000 | 670.45 | 0.523 | 8 |
| Shared R16 DD | 6000 | 710.03 | 0.522 | 8 |
| Shared R16 DD | 7000 | 700.25 | 0.519 | 8 |
| Shared R16 DF | 1000 | 943.78 | 0.494 | 8 |
| Shared R16 DF | 2000 | 933.41 | 0.506 | 8 |
| Shared R16 DF | 3000 | 922.24 | 0.526 | 8 |
| Shared R16 DF | 4000 | 900.26 | 0.515 | 8 |
| Shared R16 DF | 5000 | 835.09 | 0.516 | 8 |
| Shared R16 DF | 6000 | 885.88 | 0.524 | 8 |
| Shared R16 DF | 7000 | 868.89 | 0.523 | 8 |
| Shared R16 FD | 1000 | 788.78 | 0.401 | 8 |
| Shared R16 FD | 2000 | 796.01 | 0.401 | 8 |
| Shared R16 FD | 3000 | 673.43 | 0.403 | 8 |
| Shared R16 FD | 4000 | 679.26 | 0.402 | 8 |
| Shared R16 FD | 5000 | 690.70 | 0.405 | 8 |
| Shared R16 FD | 6000 | 669.69 | 0.403 | 8 |
| Shared R16 FD | 7000 | 682.21 | 0.402 | 8 |
| Shared R16 SS | 1000 | 892.74 | 0.404 | 8 |
| Shared R16 SS | 2000 | 958.62 | 0.403 | 8 |
| Shared R16 SS | 3000 | 974.22 | 0.404 | 8 |
| Shared R16 SS | 4000 | 1010.27 | 0.405 | 8 |
| Shared R16 SS | 5000 | 928.43 | 0.404 | 8 |
| Shared R16 SS | 6000 | 958.17 | 0.404 | 8 |
| Shared R16 SS | 7000 | 944.68 | 0.401 | 8 |
| Separate R8 DD | 1000 | 800.45 | 0.508 | 8 |
| Separate R8 DD | 2000 | 668.00 | 0.526 | 8 |
| Separate R8 DD | 3000 | 743.94 | 0.521 | 8 |
| Separate R8 DD | 4000 | 692.29 | 0.518 | 8 |
| Separate R8 DD | 5000 | 694.21 | 0.519 | 8 |
| Separate R8 DD | 6000 | 704.53 | 0.518 | 8 |
| Separate R8 DD | 7000 | 796.68 | 0.515 | 8 |
| Separate R8 FD | 1000 | 697.31 | 0.402 | 8 |
| Separate R8 FD | 2000 | 702.10 | 0.400 | 8 |
| Separate R8 FD | 3000 | 689.30 | 0.403 | 8 |
| Separate R8 FD | 4000 | 772.52 | 0.403 | 8 |
| Separate R8 FD | 5000 | 703.07 | 0.401 | 8 |
| Separate R8 FD | 6000 | 666.14 | 0.403 | 8 |
| Separate R8 FD | 7000 | 839.93 | 0.408 | 8 |
| Separate R16 DD | 1000 | 864.71 | 0.513 | 8 |
| Separate R16 DD | 2000 | 699.87 | 0.528 | 8 |
| Separate R16 DD | 3000 | 676.61 | 0.520 | 8 |
| Separate R16 DD | 4000 | 774.45 | 0.520 | 8 |
| Separate R16 DD | 5000 | 742.22 | 0.526 | 8 |
| Separate R16 DD | 6000 | 638.40 | 0.523 | 8 |
| Separate R16 DD | 7000 | 863.20 | 0.516 | 8 |
| Separate R16 FD | 1000 | 722.71 | 0.399 | 8 |
| Separate R16 FD | 2000 | 692.46 | 0.402 | 8 |
| Separate R16 FD | 3000 | 711.23 | 0.402 | 8 |
| Separate R16 FD | 4000 | 754.74 | 0.407 | 8 |
| Separate R16 FD | 5000 | 691.58 | 0.403 | 8 |
| Separate R16 FD | 6000 | 665.81 | 0.403 | 8 |
| Separate R16 FD | 7000 | 828.23 | 0.405 | 8 |

The lowest observed PPL is **638.40 for separate R16 DD at 6,000 training updates**;
the best shared model is DD at 3,000 updates (655.33), versus MDLM 929.75.
Each PPL uses 20 generated samples. Checkpoint minima use the evaluation seeds for selection
and require independent confirmation.

Across matched CCF checkpoints, eight transitions take about half the generation time of sixteen,
while PPL is roughly 2.1–3.1 times higher. These timings exclude loading, verification and scoring.
The combined 8/16/32 archive has **171 unique cells and 3,420 scored samples**,
including the original ten reused cells exactly once. All 168 CCF gates passed.
