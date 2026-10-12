# Optional static count and global scores

`additive_chain.static` prepares a full-vocabulary pair table once and then
reindexes it onto the existing candidate supports at each diffusion step:

\[
S(a,b)=s_{\mathrm{count}}(a,b)+\lambda_g A_a^\top B_b.
\]

This is the same signed additive chain, not a new probability factorization.
It removes repeated count-hash lookup and global dot products. It does not
remove the quadratic candidate-pair reductions in forward/backward inference.
The frozen global term can be cached because it does not depend on the current
masked input. A contextual term cannot use this backend.

The backend is explicitly opt-in. Existing `additive_chain.cuda` functions and
package import behavior are unchanged. NVIDIA CUDA and Triton are required for
sampling. CPU table construction is permitted only for small correctness
fixtures. There is no automatic slow CPU inference fallback.

## Interface

Use the package's `SparseCountPotential` and `CompactCandidates`. Candidate
preparation, reveal scheduling, the model forward, and scoring remain the
caller's responsibility.

```python
from additive_chain.static import (
    prepare_static_count_cache, sample_static_count,
    prepare_static_global_cache, sample_static_global,
)

# Count only. Build and compile once, outside steady-state generation timing.
count_cache = prepare_static_count_cache(count_potential)
tokens = sample_static_count(candidates, count_potential, count_cache, generator)
del count_cache

# Count plus a frozen GlobalAdditiveHead, with the external coefficient.
global_head.eval().requires_grad_(False)
cache = prepare_static_global_cache(count_potential, global_head, coefficient)
tokens = sample_static_global(candidates, count_potential, cache, generator)
cache.close()
```

`static_count_marginals` and `static_global_marginals` return normalized
single-site probabilities under those same chains. Sampling their sites
independently would be a different decoding control, not joint sampling.
The public count-marginal wrapper normalizes locally; the joint sampler's
noise and recurrence are unaffected.

Global weights must be frozen FP32, vocabulary-matched, and rank at most128,
matching the on-demand backend's current factor interface. Construction uses
IEEE FP32 matrix multiplication with TF32 temporarily disabled, then restores
the caller's setting. Nonfinite scores are rejected. Count/global inputs must
not be mutated during cache use. The global cache checks tensor identity and
mutation versions and requires explicit reconstruction after a change.

Exactly zero global coefficient or an exactly zero factor side allocates no
table and routes through the existing count-only sampler. Thus zero nesting
does not change its random-number layout. That zero path intentionally does
not provide static count acceleration; use the count-only interface above
when explicitly requesting that backend.

## Memory and timing

An FP32 table requires `4*V*V` bytes: **10,103,466,256 bytes (9.410GiB)** at
`V=50,258`. Default table, temporary-tensor and reserve budgets are12GiB,
128MiB and2GiB. Budgets are checked before allocation. The construction is
row-chunked; observed-edge chunks are bounded separately. Compatibility
conversion from CSR can require additional `O(E)` temporary memory, reported
in diagnostics. Library workspace is outside the tensor budget and must fit
the reserved memory. Keep room for the backbone and inference buffers.

Preparation allocates only memory, not files or checkpoints. Global cache
diagnostics report table bytes and separately measured synchronized build
time. Count preparation must be timed explicitly by the caller. Compilation,
cache construction, warmup, model loading and steady-state sampling are
different costs; report them separately. Release table references before
loading a large external scorer. Treat returned tables as immutable.

## Law and numerical validation

The table stores exactly the established count score convention: observed
pairs use `log(left[a]*right[b]+correction[a,b])`; absent pairs use
`log(left[a])+log(right[b])`. Only the globally shared edge constant is omitted,
as in the existing CUDA backend. Learned residuals are added in log space.
Candidate IDs remain real token IDs; padding is neutral in pair scores and
forbidden by its `-inf` unary. Both visible-boundary edges remain present.

Messages use FP32 and sampling noise uses FP64. Reordered FP32 arithmetic can
change probabilities or a particular draw slightly, so algebraic equality
does not establish bitwise trajectory parity on a new GPU. Integer address
arithmetic promotes to int64 **before** multiplying by vocabulary size:
`50,258**2` exceeds signed int32.

```bash
python -m pytest -q tests/test_additive_chain_*.py
```

Tests cover score composition, normalization, enumeration, visible boundaries,
padding, exact zero dispatch, memory budgets, mutation guards and large token
addresses. CUDA tests include caps500/1000/2000, long small-cap chains and
same-noise draws. CUDA skips must be reported explicitly. This portable static
module's two static test modules passed **21 tests, including 8 CUDA tests, with
no skips on an NVIDIA A100**. These are unit-scale numerical and sampling
checks. They did not allocate the real `50,258 x 50,258` table, run a full-model
private-versus-public trajectory comparison, or measure generation quality or
speed. An actual same-token model-trajectory check and end-to-end measurement
are still required before attributing a benchmark result to this package.

## Attribution

Message and sampling kernels derive from **Nina Zhang's** Apache-2.0
[`chain_crf/fused_pairs.py`](https://github.com/yuntian-group/mdlm/blob/18175679c93d7792936e1497b7c5be72a765f2de/chain_crf/fused_pairs.py)
at commit `18175679c93d7792936e1497b7c5be72a765f2de`. The static version replaces
the pair lookup with an FP32 table indexed by int64 token-pair addresses while
retaining the segmented recurrence and sampling convention. See
[CUDA_BACKEND.md](CUDA_BACKEND.md) for the original source hash and broader
provenance. The repository's Apache-2.0 license applies; retain the attribution
and license when redistributing these portions.
