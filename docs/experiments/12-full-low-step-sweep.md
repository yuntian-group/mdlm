# Full checkpoint sweep at 16 and 32 sampling transitions

Historical evaluation completed September 17, 2026; these are saved results.
Each CCF family was evaluated at training updates 1,000 through 7,000 in increments of 1,000.
The matrix contains the four shared-endpoint R16 arms and the FD/DD arms for separate R8 and R16.

All cells used **20 generated samples**, length 1,024, batch 1, seeds 91001–91020,
and GPT2-large scoring through the first nonleading EOS. CCF used **L, level-batched optimized** sampling.
There are 114 unique completed cells and 2,280 scored samples across the two budgets.
Of these, the initial ten cells (including both MDLM baselines) and their 200 samples were reused;
this extension generated **2,080 new samples in 104 new cells**. No samples were added by baseline reuse.

All 112 CCF cells passed their same-GPU first-sample token, NFE and RNG gate.
The verified first optimized sample counts toward the 20 scored samples; reference duplicates do not.
Actual model-call counts are recorded separately from requested transitions in the final result archive.

| Family / arm | Training update | 16-step PPL | 32-step PPL | 16-step seconds/sample | 32-step seconds/sample |
|---|---:|---:|---:|---:|---:|
| MDLM | — | 311.78 | 136.34 | 0.316 | 0.417 |
| Shared R16 DD | 1000 | 295.82 | 181.05 | 1.005 | 2.013 |
| Shared R16 DD | 2000 | 312.02 | 181.73 | 1.030 | 2.057 |
| Shared R16 DD | 3000 | 259.74 | 147.47 | 1.040 | 2.080 |
| Shared R16 DD | 4000 | 272.38 | 157.82 | 1.050 | 2.090 |
| Shared R16 DD | 5000 | 285.17 | 132.30 | 1.053 | 2.090 |
| Shared R16 DD | 6000 | 264.41 | 150.84 | 1.038 | 2.100 |
| Shared R16 DD | 7000 | 258.75 | 165.60 | 1.052 | 2.078 |
| Shared R16 DF | 1000 | 389.03 | 210.93 | 1.047 | 1.972 |
| Shared R16 DF | 2000 | 357.98 | 207.46 | 1.023 | 2.028 |
| Shared R16 DF | 3000 | 352.85 | 225.10 | 1.041 | 2.074 |
| Shared R16 DF | 4000 | 378.71 | 201.73 | 1.025 | 2.042 |
| Shared R16 DF | 5000 | 351.65 | 201.58 | 1.027 | 2.067 |
| Shared R16 DF | 6000 | 358.08 | 230.09 | 1.034 | 2.076 |
| Shared R16 DF | 7000 | 396.02 | 197.69 | 1.028 | 2.113 |
| Shared R16 FD | 1000 | 315.63 | 163.15 | 0.779 | 1.635 |
| Shared R16 FD | 2000 | 274.06 | 126.83 | 0.803 | 1.621 |
| Shared R16 FD | 3000 | 268.46 | 141.55 | 0.804 | 1.649 |
| Shared R16 FD | 4000 | 247.46 | 161.66 | 0.799 | 1.615 |
| Shared R16 FD | 5000 | 234.61 | 149.84 | 0.797 | 1.608 |
| Shared R16 FD | 6000 | 221.93 | 141.84 | 0.803 | 1.613 |
| Shared R16 FD | 7000 | 250.33 | 149.13 | 0.798 | 1.609 |
| Shared R16 SS | 1000 | 331.12 | 180.98 | 0.778 | 1.618 |
| Shared R16 SS | 2000 | 352.57 | 181.36 | 0.790 | 1.601 |
| Shared R16 SS | 3000 | 322.16 | 178.90 | 0.794 | 1.605 |
| Shared R16 SS | 4000 | 325.79 | 193.75 | 0.793 | 1.596 |
| Shared R16 SS | 5000 | 329.37 | 186.79 | 0.794 | 1.590 |
| Shared R16 SS | 6000 | 320.86 | 218.15 | 0.792 | 1.608 |
| Shared R16 SS | 7000 | 330.73 | 191.06 | 0.798 | 1.588 |
| Separate R8 DD | 1000 | 327.71 | 174.23 | 1.008 | 2.048 |
| Separate R8 DD | 2000 | 285.13 | 130.89 | 1.039 | 2.105 |
| Separate R8 DD | 3000 | 291.04 | 160.61 | 1.036 | 2.089 |
| Separate R8 DD | 4000 | 275.50 | 160.52 | 1.037 | 2.074 |
| Separate R8 DD | 5000 | 257.77 | 172.90 | 1.033 | 2.066 |
| Separate R8 DD | 6000 | 248.14 | 150.73 | 1.040 | 2.063 |
| Separate R8 DD | 7000 | 309.13 | 171.63 | 1.038 | 2.058 |
| Separate R8 FD | 1000 | 266.97 | 180.96 | 0.794 | 1.570 |
| Separate R8 FD | 2000 | 282.45 | 145.96 | 0.796 | 1.592 |
| Separate R8 FD | 3000 | 304.13 | 139.30 | 0.801 | 1.604 |
| Separate R8 FD | 4000 | 301.04 | 158.79 | 0.801 | 1.613 |
| Separate R8 FD | 5000 | 248.50 | 160.21 | 0.795 | 1.606 |
| Separate R8 FD | 6000 | 267.96 | 130.76 | 0.798 | 1.600 |
| Separate R8 FD | 7000 | 289.11 | 161.76 | 0.797 | 1.611 |
| Separate R16 DD | 1000 | 312.34 | 188.93 | 1.015 | 2.037 |
| Separate R16 DD | 2000 | 277.29 | 151.97 | 1.044 | 2.097 |
| Separate R16 DD | 3000 | 311.64 | 151.60 | 1.042 | 2.071 |
| Separate R16 DD | 4000 | 303.75 | 164.35 | 1.039 | 2.081 |
| Separate R16 DD | 5000 | 287.62 | 166.74 | 1.042 | 2.071 |
| Separate R16 DD | 6000 | 299.03 | 160.34 | 1.054 | 2.074 |
| Separate R16 DD | 7000 | 321.56 | 190.44 | 1.042 | 2.058 |
| Separate R16 FD | 1000 | 268.70 | 169.63 | 0.803 | 1.603 |
| Separate R16 FD | 2000 | 291.64 | 156.96 | 0.800 | 1.589 |
| Separate R16 FD | 3000 | 300.98 | 131.57 | 0.797 | 1.601 |
| Separate R16 FD | 4000 | 306.36 | 171.91 | 0.803 | 1.608 |
| Separate R16 FD | 5000 | 239.69 | 142.81 | 0.803 | 1.599 |
| Separate R16 FD | 6000 | 252.18 | 139.77 | 0.801 | 1.600 |
| Separate R16 FD | 7000 | 332.95 | 188.00 | 0.800 | 1.611 |

Shared FD has the lowest observed PPL at each budget: **221.93 at 6,000 updates / 16 transitions**,
and **126.83 at 2,000 updates / 32 transitions**, compared with MDLM 311.78 and 136.34.
Each reported PPL uses 20 samples. These checkpoints were selected on the same evaluation seeds,
so the minima are exploratory and need fresh held-out seeds before claiming an advantage.

Generation times exclude loading, verification duplicates and scoring. All cells ran on the same H200 NVL host,
but sequential observed timings are not an isolated hardware benchmark. Different scored lengths reflect EOS.
At a similar observed runtime, MDLM at 32 transitions (PPL 136.34, 0.417 seconds/sample) outperforms
shared FD at 16 transitions (PPL 221.93, 0.803 seconds/sample).
