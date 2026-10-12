# Optional cached-base contextual inference

`additive_chain.static_contextual` combines a fixed count/global table with
fresh contextual factors at each denoising step. Its edge score is

\[
S_i(a,b;x_t)=
\underbrace{s_{\mathrm{count}}(a,b)+\lambda_g A_a^\top B_b}_{\text{cached once}}
+\lambda_\ell U_i(a;x_t)^\top V_{i+1}(b;x_t).
\]

The count potential already contains its selected count strength. The global
coefficient is applied when preparing the table; the contextual coefficient
is applied to the supplied left factors. Do not apply any coefficient twice.
These are signed log scores: there is no positivity transform, extra softmax,
or low-rank approximation to the exponentiated pair potential.

## Interface

```python
from additive_chain.heads import concatenate_scaled_factors
from additive_chain.static import prepare_static_global_cache
from additive_chain.static_contextual import sample_static_contextual

# Once, before the timed generation loop. Count and global weights must stay
# immutable. The existing static cache enforces source/version checks.
global_head.eval().requires_grad_(False)
local_head.eval().requires_grad_(False)
cache = prepare_static_global_cache(count_potential, global_head, global_strength)
try:
    # Repeat at each denoising step, after one frozen masked-input forward:
    # candidates were made from that pass's ORIGINAL base logits;
    # masked_hidden came from that SAME input, not gold/reveal-filled tokens.
    left, right = concatenate_scaled_factors([
        (local_strength, local_head.node_factors(candidates.candidate_ids, masked_hidden)),
    ])
    proposal = sample_static_contextual(
        candidates, count_potential, cache, left, right, generator,
    )
    tokens = proposal.where(reveal_mask & tokens.eq(mask_id), tokens)
finally:
    cache.close()
```

`static_contextual_marginals` returns singleton probabilities of the same
chain. Sampling those probabilities independently is a separate control,
not joint sampling. `infer_static_contextual(..., sampling="joint")` returns
candidate slots; `sample_static_contextual` returns vocabulary token IDs.
`dense_pair_reference` is a budget-limited score oracle for small tests, not
a production CPU fallback or a training implementation.

This backend does not call the encoder, construct/reselect candidates, inject
targets, choose reveal positions, or change unaries. All unrevealed interior
sites stay in the chain. Singleton supports separate spans, and both boundary
pair scores are absorbed into their neighboring unaries. Invalid candidate
slots have neutral pair scores and remain forbidden by `-inf` unaries.

## Zero terms, precision and memory

- An absent or identically zero contextual factor side dispatches directly
  to the existing cached-global sampler, retaining its random-number layout.
- If `prepare_static_global_cache` returns an inactive zero-global cache,
  inference uses the existing on-demand count-plus-contextual backend. It does
  **not** allocate a full-vocabulary table inside the decoding loop.
- Active inference requires NVIDIA CUDA and Triton. Factors must be finite
  FP32 `[B,L,K,R]`, with `1 <= R <= 128`; candidate width is at most 2,048.
  This rank is the **local** rank, because the global score is already cached.
  Candidate IDs and padding follow [CUDA_BACKEND.md](CUDA_BACKEND.md).
- Fixed scores, contextual arithmetic and messages use FP32; categorical
  perturbations use FP64. Int64 table addresses are formed before multiplying
  by vocabulary size. FP32 reassociation can change a draw from a fixed seed;
  algebraic equivalence is not itself a bitwise trajectory guarantee.
- The fixed table costs `4*V*V` bytes, approximately 9.410 GiB at `V=50,258`.
  In addition, the two local factor arrays cost `8*B*L*K*R` bytes and must be
  regenerated when the masked input changes. Table construction uses the
  explicit memory budgets in [STATIC_BACKEND.md](STATIC_BACKEND.md).

Measure compilation, table construction, model loading and warmup separately
from steady-state generation. Include candidate preparation, fresh local
factor computation, inference and rejected attempts in generation latency.
Release the table before loading an external scorer when memory is limited.
This is inference-only; use the dense reference for differentiable objectives.

## Validation and attribution

```bash
python -m pytest -q tests/test_additive_chain_static_contextual.py
python -m pytest -q tests/test_additive_chain_*.py
```

CPU tests compare scores and complete tiny joint distributions with the
ordinary concatenated-factor model, including visible boundaries, padding,
changing masked hidden states and zero-term dispatch. Optional CUDA tests
cover messages, same-noise draws, marginals and zero-term RNG parity. CUDA
skips must be reported explicitly.

The cached contextual path also passed an actual pretrained-MDLM A100 shadow
trajectory at K500, length 1,024 and 32 steps, with count/global/contextual
strengths 0.1/0.25/0.25. Its fixed table, fresh factors, candidate supports,
unaries, tokens and proposal RNG matched the frozen research implementation
bitwise. This is one fixed raw attempt, not a quality or speed evaluation, and
does not replace the other width/marginal/zero-term tests. See the precise
[four-method validation scope](GENERATION.md#tests-and-current-validation-scope).

The message/sampling kernels retain Nina Zhang's Apache-2.0 attribution from
[`chain_crf/fused_pairs.py`](https://github.com/yuntian-group/mdlm/blob/18175679c93d7792936e1497b7c5be72a765f2de/chain_crf/fused_pairs.py),
commit `18175679c93d7792936e1497b7c5be72a765f2de`. The contextual extension
adds a dense fixed-table lookup to the fresh signed factor dot product,
without changing the existing count-only and on-demand additive branches.
See [CUDA_BACKEND.md](CUDA_BACKEND.md) for the original kernel provenance.
The repository's Apache-2.0 license applies.
