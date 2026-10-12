# Optional CUDA inference

The dense `additive_chain` implementation remains the training/correctness
reference. This separate, explicit opt-in backend computes count and learned
pair scores inside tiled CUDA kernels, avoiding the full `[B,L,K,K]` edge
tensor. It requires NVIDIA CUDA, PyTorch and Triton. It raises on CPU/ROCm or
missing Triton; there is no silent fallback. Importing `additive_chain` itself
does not import Triton.

First-use Triton compilation also needs a working host C compiler and the
development headers matching the active Python interpreter (including
`Python.h`). A missing compiler or header is an environment setup failure,
not a restriction of the count or CRF model.

The computational cost is still quadratic in candidate count. Fused code is
not evidence of a particular speedup: benchmark the packaged code on the
target GPU, including candidate construction, factor preparation, sampling,
and any rejection/retry policy. Separate first-use compilation and count-table
construction from warmed generation. The dense reference's performance must
not be substituted for this backend's performance, or vice versa.

## Contract

```python
from additive_chain.candidates import extract_compact_candidates
from additive_chain.counts import SparseCountPotential
from additive_chain.cuda import sample_additive_count, additive_count_marginals

# Training-only count statistics, already resident on the chosen device:
# left_counts[V], right_counts[V], sorted unique pair_keys[E]=a*V+b,
# pair_counts[E]. Endpoint counts must equal the observed pair marginals.
potential = SparseCountPotential.from_counts(
    left_counts, right_counts, pair_keys, pair_counts,
    mode="conditional", strength=0.005, smoothing=0.1,
)
candidates = extract_compact_candidates(
    base_log_probs, masked_tokens, mask_id=mask_id, nucleus_p=0.9, top_k=500,
)
# left/right optional FP32 node factors [B,L,K,R], already coefficient-scaled.
# For the count-only model, omit both. Returns token IDs [B,L], not slots.
proposal = sample_additive_count(candidates, potential, left, right, generator)
updated = proposal.where(reveal_mask & masked_tokens.eq(mask_id), masked_tokens)
```

The frozen denoiser runs on the **original** partially masked input. Candidate
construction uses the original base logits. Count/global/contextual terms do
not reselect candidates. Learned node factors may be obtained with the package
heads and `concatenate_scaled_factors`; edge `i` uses left `i` and right `i+1`.
There is no additional softmax, positivity transform or implicit rank scaling.

- Candidate IDs are actual vocabulary IDs. Padding is ID `V` with unary
  `-inf`; valid candidate IDs must be unique within each row. Every row has
  nonempty support. Visible tokens are singleton supports.
- Candidate width is bounded by 2,048 and the combined learned rank by 128.
  Large widths need explicit target-hardware CUDA validation; a passing small
  test does not establish that all widths/configurations work on every GPU.
- Full sequence lengths are represented by actual sites. This backend does
  **not** accept the dense reference's variable-length sequence-padding API.
- Finite unaries are converted to FP32, as are hash scores and messages.
  Learned factors must be finite FP32 tensors. Sampling perturbations are
  FP64. The law is exact on the retained support up to numerical arithmetic;
  different reduction/chunking orders can change draws from a fixed seed.
- Zero-initialized/zero-coefficient learned factors route through the count
  recurrence. This preserves count-only nesting. A uniform count potential
  is law-equivalent to independent unaries, but FFBS need not consume the same
  random stream as a baseline multinomial sampler.
- Singleton sites separate spans. Both visible-boundary edges are absorbed
  into endpoint unaries. Hidden interior sites remain latent even when they
  are not selected for this step's reveal.
- `additive_count_marginals` returns the same CRF's singleton probabilities.
  Independently sampling those is a control, not joint CRF sampling.
- This is inference-only. It has no autograd/log-partition API and must not
  replace `chain.py` in training. `dense_pair_reference` is a size-bounded score
  oracle, not an optimized production path.

## Count construction

Let `N` be the total number of training bigrams and `V` the vocabulary size.
Define add-one endpoint backoffs

```
bL(a) = (cL(a)+1)/(N+V)
bR(b) = (cR(b)+1)/(N+V)
J(a,b) = (1-eps)*c(a,b)/N + eps*bL(a)*bR(b)
```

For empty counts, set `eps=1`. With `JL` and `JR` denoting marginals of `J`,
conditional mode uses `W=(J/JL)^strength`; PMI mode uses
`W=(J/(JL*JR))^strength`. The constructor represents this **powered** potential
as an exact positive separable background plus corrections on observed pairs,
scaled by one shared edge constant. It is not a low-rank approximation to `W`.
The CUDA sampler drops only the shared scalar, which cancels in normalized
fixed-length probabilities. `potential.log_scores(a,b)` includes that scalar
for explicit dense composition; the inference score oracle omits it.

Construction allocates `O(V+E)` statistics, never a `V²` table. Very extreme
strengths can exceed FP64 dynamic range; invalid values raise rather than clip
the model. Store counts on the intended CUDA device before constructing the
potential. Construct them outside `torch.inference_mode` so mutation counters
are available; ordinary `torch.no_grad` is fine.

For an existing exact sparse-plus-background potential, call
`SparseCountPotential.from_reference(existing)` once before inference. The
compatibility input needs FP64 `left/right`, a CSR correction `sparse`, sorted
optional `pair_keys`, and `log_scale`. This is a numerical interface, not a
dependency on another workflow or checkpoint format.

The hash lookup is cached for one immutable potential. Mutating cached tensors
raises; construct a new potential when changing counts/strength. Cache ownership
prevents object-ID reuse, and switching potentials releases the old cached
table. `additive_chain._count_inference.prepare_count_cache(potential)` reports
allocated cache bytes for explicit warmup accounting; `clear_count_cache()` in
`additive_chain.cuda` releases it. Keys preserve high uint32 bits and use int64
when the vocabulary range would collide with the empty-slot sentinel.

## Optional preparation helpers

`additive_chain.fast_candidates.extract_fast_candidates` is a separate opt-in
replacement for the full-sort reference. It preserves descending-score/ID
tie ordering, normalizes over the full clean vocabulary, excludes the nucleus
crossing token while always retaining the first, intersects with top-K, then
sorts surviving token IDs. Ambiguous FP64 subtraction/cumulative boundaries
fall back to the reference. Its diagnostics report retained support/mass;
the original uncapped nucleus sizes and masses are explicitly unavailable.
No tail aggregate is added.

`additive_chain.spans.relevant_span_context` can remove whole latent spans
containing no requested reveal. Use its returned context only for candidate
preparation/inference, never for the denoiser forward. This reduction requires
nearest-neighbor edges and exact singleton visible clamps. It must not remove
an unselected hidden node inside a relevant span, and it is invalid for
additional nonlocal edges. Dummy states must be legal under the positive count
potential. The final token update still uses the original tokens/reveal mask.

## Verification

```bash
python -m pytest -q tests/test_additive_chain_*.py
```

CPU tests cover the count formula/scale, candidate ordering and draws,
enumerated selected-token laws under span pruning, additive score composition,
zero nesting and CPU backend rejection. CUDA tests are explicitly skipped when
the required hardware/backend is unavailable. Their coverage includes count
nesting, visible clamps, long chains, rank128, candidate caps500/1000/2000,
large integer count keys, marginals and a tiny empirical joint-law check.
A CPU-only result must report the skips; it is not a CUDA validation receipt.
Real-model generation, quality, end-to-end latency, compilation portability,
and memory limits require separate runs.

## Provenance and modifications

The fused count-pair kernels derive from **Nina Zhang's**
[`chain_crf/fused_pairs.py`](https://github.com/yuntian-group/mdlm/blob/18175679c93d7792936e1497b7c5be72a765f2de/chain_crf/fused_pairs.py),
commit `18175679c93d7792936e1497b7c5be72a765f2de`, in the Apache-2.0-licensed
`yuntian-group/mdlm` repository. The original file SHA-256 is
`1f5b8347737d4ef7a51a565ea5fa6d604aa0211d72c187e8626c00301ad57171`.
The repository's Apache-2.0 LICENSE applies; retain this attribution and the
license when redistributing these portions.

`_count_kernels.py` extracts the original count-hash, pair, message and sampling
functions and removes unused old head adapters. `_additive_kernels.py` extends
the pair switch so count scores **and** signed learned dot products can be
added, including visible boundaries; zero residuals delegate to the extracted
count recurrence. Imports are package-relative. The adapter uses our explicit
count-potential interface and adds a bounded cache with mutation/device guards.
It does not use the original candidate builder or its aggregate tail state.
The remaining count construction, candidate preparation, span reduction and
additive adapter are supplied here under the same repository license.

These components are independent of model downloads, datasets, experiment
scheduling, cloud infrastructure and scorer policies. No such workflow is
implicitly installed or run by this package.
