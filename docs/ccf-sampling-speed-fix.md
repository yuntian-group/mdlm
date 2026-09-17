# Sampling optimization V1

Introduced September 15/16, 2026, after the original generation screens.
Sampler O means the inherited unoptimized recovery implementation; V1 is an
optimization relative to O, even when a later replay calls V1 its reference.

- Avoid a redundant inference call before joint sampling.
- Reuse CPU edge-orientation metadata instead of per-edge CUDA scalar reads.
- Hoist unconditional empty-evidence checks out of the generation loop.

Model parameters, arithmetic/traversal order, categorical draws, seeds,
precision, reveal schedule, cleanup and scorer policy remain unchanged.
The pinned O reference is the inherited recovery baseline; source SHA-256
checks precede execution. This reference remains reachable on this branch.

Historical isolated CPU timings recorded 1.27x/1.34x/1.29x speedups for
fully active/mixed/fully clamped synthetic states, respectively: batch 2,
length 1024, K128, R16, vocabulary 256, three timing repetitions per state.
These are sampler-only timings, not GPU generation measurements or PPL.
The fully clamped case is a diagnostic fixture.

The historical verification covers 54 exact token/RNG cases with the optional
head modes present. The harness test additionally exercises 1,001 calls and
checks conditional evidence. Individual GPU update checks do not establish
full-trajectory replay. Curation reran the focused CPU tests: 5 passed and
1 CUDA-only check was skipped on this CPU host.

```bash
python -m unittest discover -s tests -p test_ccf_sampling_optimization.py
python scripts/verify_ccf_sampling_optimization.py --device cpu --benchmark
# In an allocated GPU environment:
python scripts/verify_ccf_sampling_optimization.py --device cuda --benchmark --real-step-check
```

Activate the environment before submission and configure CCF_CACHE_ROOT.
Never modify an existing experimental snapshot. New GPU evaluations require
new output locations and their own successful verification evidence.

These changes reorganize existing repository computations. Standard tree
message passing follows [Kschischang et al.](https://doi.org/10.1109/18.910572).
No external implementation was used for the historical V1 patch; this does
not make the underlying inference algorithm a new contribution. Retain all
inherited repository and dependency credits.
