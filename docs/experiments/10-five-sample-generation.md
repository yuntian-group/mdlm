# Five-sample generation with level-batched optimization

Historical September 16, 2026. The evaluation matrix expands beyond selected
one-sample minima and includes the released MDLM baseline. Every completed
cell generated **5 samples**, seeds 91001–91005: **28 cells, 140 samples total**.
All 27 CCF first-sample gates matched final tokens, actual calls and CPU/CUDA
RNG against the appropriate conservative reference. The verified first sample
counts toward the five; it is not an extra scored generation.

All CCF cells below use **L: level-batched optimized sampling**, length 1024,
batch 1 and 1,000 reverse transitions plus cleanup. MDLM uses the factorized
harness. PPL is exp(token-weighted GPT-2-large NLL) under the established EOS
policy, not mean sample PPL. Times exclude loading, verification and scoring;
these records are not a controlled comparison with a different hardware study.

The original DF 1k cell failed before generation on an allocation-confirmation
timeout; it has no PPL and is excluded from the 140 samples. Checkpoint choices
remain exploratory, and five samples do not establish a stable architecture
ranking. These are saved results, not fresh runs during curation.

| Family | Arm | Embedding | Rank | Train steps | Samples | Code | PPL | Seconds/sample |
|---|---|---|---|---|---|---|---|---|
| Original arms | static_static | shared | 16 | 1000 | 5 | L level-batched | 129.147 | 47.896 |
| Original arms | fixed_dynamic | shared | 16 | 1000 | 5 | L level-batched | 81.231 | 48.040 |
| Original arms | dynamic_dynamic | shared | 16 | 1000 | 5 | L level-batched | 133.674 | 60.565 |
| Separate endpoints | fixed_dynamic | separate | 8 | 1000 | 5 | L level-batched | 101.338 | 48.171 |
| Separate endpoints | dynamic_dynamic | separate | 8 | 1000 | 5 | L level-batched | 98.728 | 61.264 |
| Separate endpoints | fixed_dynamic | separate | 16 | 1000 | 5 | L level-batched | 98.557 | 47.854 |
| Separate endpoints | dynamic_dynamic | separate | 16 | 1000 | 5 | L level-batched | 128.530 | 62.808 |
| Original arms | static_static | shared | 16 | 5000 | 5 | L level-batched | 112.685 | 47.989 |
| Original arms | fixed_dynamic | shared | 16 | 5000 | 5 | L level-batched | 75.769 | 48.046 |
| Original arms | dynamic_fixed | shared | 16 | 5000 | 5 | L level-batched | 125.680 | 62.545 |
| Original arms | dynamic_dynamic | shared | 16 | 5000 | 5 | L level-batched | 73.512 | 62.580 |
| Original arms | static_static | shared | 16 | 6000 | 5 | L level-batched | 113.403 | 47.995 |
| Original arms | fixed_dynamic | shared | 16 | 6000 | 5 | L level-batched | 78.644 | 47.866 |
| Original arms | dynamic_fixed | shared | 16 | 6000 | 5 | L level-batched | 125.819 | 64.540 |
| Original arms | dynamic_dynamic | shared | 16 | 6000 | 5 | L level-batched | 107.481 | 63.492 |
| MDLM | factorized | none | — | Released | 5 | MDLM factorized harness | 32.775 | 12.117 |
| Original arms | static_static | shared | 16 | 7000 | 5 | L level-batched | 133.952 | 47.841 |
| Original arms | fixed_dynamic | shared | 16 | 7000 | 5 | L level-batched | 73.129 | 47.924 |
| Original arms | dynamic_fixed | shared | 16 | 7000 | 5 | L level-batched | 128.680 | 62.555 |
| Original arms | dynamic_dynamic | shared | 16 | 7000 | 5 | L level-batched | 80.190 | 62.440 |
| Separate endpoints | fixed_dynamic | separate | 8 | 2000 | 5 | L level-batched | 86.185 | 55.901 |
| Separate endpoints | fixed_dynamic | separate | 8 | 6000 | 5 | L level-batched | 73.829 | 48.590 |
| Separate endpoints | dynamic_dynamic | separate | 8 | 2000 | 5 | L level-batched | 107.240 | 62.399 |
| Separate endpoints | dynamic_dynamic | separate | 8 | 6000 | 5 | L level-batched | 74.488 | 72.165 |
| Separate endpoints | fixed_dynamic | separate | 16 | 2000 | 5 | L level-batched | 88.607 | 48.337 |
| Separate endpoints | fixed_dynamic | separate | 16 | 4000 | 5 | L level-batched | 88.005 | 47.878 |
| Separate endpoints | dynamic_dynamic | separate | 16 | 2000 | 5 | L level-batched | 73.126 | 62.584 |
| Separate endpoints | dynamic_dynamic | separate | 16 | 4000 | 5 | L level-batched | 79.398 | 63.259 |
