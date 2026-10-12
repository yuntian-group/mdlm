# Finite-step generation

`generation.py` is a callable generation core for a frozen masked language
model, with count, global, and contextual pair scores. It consumes the public
[backbone interface](BACKBONE.md), candidates, count potentials, heads, and
CUDA inference APIs. It does not load checkpoints, select strengths, train, or
evaluate quality. Use [evaluation.py](EVALUATION.md) on the saved output.

This is a port of the finite-step scheduler and additive reveal adapter used
for the experiments, not the upstream first-hitting sampler. The underlying
released MDLM forward comes from the pinned Block Diffusion implementation
documented in BACKBONE.md. CUDA inference derives from the separately attributed
Nina Zhang kernels documented in CUDA_BACKEND.md; the scheduler itself performs
no dynamic programming.

## Exact schedule and randomness

For scheduled step `j=0,...,T-1`, use
`t=(T-j)/T`, `s=(T-j-1)/T`, and reveal each remaining masked site with probability
`1-s/t`. The last step reveals every remaining mask. A fresh **FP64** uniform
tensor is drawn for the whole window at every scheduled step, including
already visible positions. A nonempty reveal makes exactly one frozen model
forward, then updates only the selected positions. An empty step makes no
forward or proposal call. Returned forward counts are measured, not assumed
equal to `T`. There is no extra final denoising/cleanup call.

The one-window experiment starts with 1,024 positions: the tokenizer's BOS
ID (also GPT-2's EOS ID) at position zero and masks elsewhere. It does not force
a final EOS, truncate at internal EOS, or remove that initial context token.
Generic tiny windows are supported for tests, but the pinned backbone requires
1,024. Semi-autoregressive extension to 2,048 is **not implemented here**.

Two explicit generators separate reveal and proposal draws. `stream_seed`
computes the existing SHA256 attempt/window/stream key. Methods, step counts,
accepted-sample numbers, and rejection outcomes do not enter the key. Rejection
therefore does not shift the randomness of a later raw attempt. Supply a
namespace explicitly; matching a historical stream requires its exact string:

- Development: `bd3lm_parallel_count_sweep_v1:fixed-seed-1`.
- Frozen final evaluation: `additive-ladder-final-evaluation-v1`.

Do not reuse a development prefix as a claimed new final population. Equal
seeds do not promise identical CPU/CUDA draws, changed-backend draws, or draws
across library versions. Exact trajectory parity must be checked on the
actual runtime/checkpoint; the bounded validation below covers one such setup.

## Proposals and three stages

The score is the same additive chain used in the other package modules:

`sum unary + count_strength * log p_count(next|current)`
`+ global_strength * global_dot + contextual_strength * contextual_dot`.

`count_statistics` is a dictionary of tensor arrays named `left_counts`,
`right_counts`, `pair_keys`, and `pair_counts`, from training text only.
The vocabulary includes the final mask ID. `SparseCountPotential.from_counts`
applies the count strength **once**, with the established smoothing rule.
Learned coefficients multiply left factors once; they are not temperatures
inside a head. The contextual features come from the same masked-input
forward as the unary probabilities. No gold tokens or extra encoder calls
enter generation. Heads must already be frozen FP32 modules on the device.

Before any residual score is applied, candidates use the original normalized
clean-token distribution, top-p, then intersection with top-K. The nucleus
convention excludes the crossing token and always keeps the first token;
ties break by ascending token ID. Candidate IDs are then sorted by ID and
unaries renormalized. The optional fast candidate extractor preserves that
support rather than selecting a different nucleus. Visible tokens are singleton
constraints. Unrevealed positions remain latent, including hidden positions
between two selected sites.

- With every coefficient zero, the baseline draws FP64 categorical samples
  only for selected positions. Inactive supplied heads are never called.
- `sampling="joint"` invokes full chain FFBS. The fused/static implementation
  uses its existing FP64 exponential/Gumbel noise and segmented draw ordering;
  this core does not replace it with a CPU categorical sampler.
- `sampling="own-marginal"` computes singleton marginals of **that same chain**,
  then makes independent FP32 multinomial draws only at selected positions.
  Support, unary scores, coefficients, hidden inputs and context are unchanged.

The optional `independent_control=IndependentControl(...)` argument to
`ChainDecoder` supplies a different control: independent categorical draws
after frequency correction and/or post-support temperature, with no pair
model. Import it from `additive_chain.sampling_controls`. Supply
`frequency_strength=alpha` and an FP64 `log_marginal` from
`count_controls.smoothed_unigram_log_probs(right_counts, smoothing=...)`,
or `temperature=tau`. Put the marginal on the decoder device before constructing
the control; it is validated and copied once outside the timed loop.
The saved control identity includes the copied vector's SHA256, dtype and
shape, so programmatic inputs remain distinguishable even without a count file.

Candidate extraction receives the original diffusion scores. The independent
control then applies `softmax((u + alpha*1[i>0]*log J_R)/tau)` on those candidates,
preserving all padding and using the original absolute sequence positions for
the incoming-edge mask. Zero frequency and unit temperature delegate to the
unchanged FP64 baseline categorical path. Both individual controls and their
explicit composition have unit tests; the combined law has not been measured
in the reported experiments. Nonzero pair strengths and an active independent
control are rejected explicitly. See [CLI.md](CLI.md) for commands.

`prune_irrelevant=True` uses visible separators to omit masked spans containing
no selected site. It changes only inference preparation: the model still sees
the original masked window, and the final update preserves all unselected
positions. This optimization is specific to the nearest-neighbor chain.

Shape, dtype and device contracts are checked on every reveal. Candidate
extraction checks NaN/+Inf values and a finite clean-token normalizer for every
masked row used by inference. The core does not repeat an additional
full-window validity scan by default. For diagnosing a caller's malformed
outputs, `strict_unary_validation=True` additionally checks *all* masked rows,
including irrelevant spans and unselected baseline sites. This diagnostic
option adds full-vocabulary passes and device synchronizations; the output
records whether it was enabled. Neither setting removes an unselected latent
site that is relevant to an active chain.

## Use

```python
import json
from additive_chain.generation import ChainDecoder, generate_saved

# frozen = load_bd3lm(...) from additive_chain.backbone, using local assets.
# counts: the four training-count tensors described above.
# global_head/local_head: loaded, eval-mode, requires_grad_(False), on CUDA.
decoder = ChainDecoder(
    mask_id=frozen.mask_id, device="cuda", count_statistics=counts,
    count_strength=.1, global_head=global_head, global_strength=.25,
    contextual_head=local_head, contextual_strength=local_strength,
    top_k=500, nucleus_p=.9, sampling="joint", backend="fused",
    fast_candidates=True, prune_irrelevant=True,
)
try:
    decoder.prepare()  # Explicit preparation, outside attempt timing.
    saved = generate_saved(
        frozen.forward, decoder, frozen.model.tokenizer,
        samples=64, steps=32, length=1024,
        stream_namespace="my-declared-development-stream", warmup=True,
    )
    with open("samples.json", "x") as stream:
        json.dump(saved, stream, allow_nan=False)
finally:
    decoder.close()
```

Coefficients in this example illustrate the interface; they are not a generic
recommendation or automatic validation result. For baseline generation omit
counts/heads and leave coefficients at zero. `sample_window` also accepts an
explicit token window and reveal/proposal `torch.Generator` instances, making
the schedule testable independently of a checkpoint.

The fused backend prepares its count hash explicitly. `backend="static"`
instead prepares a fixed count/global FP32 table once; optional contextual
scores remain dynamic and use the original forward's features. Pass explicit
`max_table_bytes`, `max_temporary_bytes`, and `reserve_bytes` to `prepare`.
The full-vocabulary table is large (about9.4GiB for this vocabulary); it is not
enabled automatically. See STATIC_BACKEND.md and STATIC_CONTEXTUAL_BACKEND.md
for numerical parity and memory-budget details. Both active pair backends
require NVIDIA CUDA/Triton. There is no automatic slow CPU fallback.
An exactly zero global head uses the existing fused count predecessor when
the static cache API returns no table; `prepare()` reports this explicitly as
`zero_global_fused_fallback`. It is not a measured static-cache speed result.

Treat decoder coefficients, count tensors, and frozen heads as immutable after
construction. Close the decoder before handing memory to the external scorer.
The underlying fused count hash is process-shared; use one active decoder at
a time, and do not treat closing/rebuilding it as free inference work.

## Rejections, timing, and output

Each complete attempt computes the native FP32 empirical entropy of its last
256 token IDs on the generation device. Entropy below four nats marks that
attempt as rejected. By default (`population="accepted"`), ten consecutive
rejections for one accepted-sample slot stop the run
with `status="retry_limit"`; successful acceptance resets that counter. There
is no forced cleanup, acceptance override, or retry-counter reset to hide a
failure. Every rejected token sequence remains in the output.

With `population="raw"`, `samples` instead fixes the total number of raw
attempts. Every attempt remains, no rejected slot is replaced, and there is no
consecutive-rejection limit. The same sampler, acceptance diagnostic, attempt
IDs and explicit random streams are used. Only the stopping rule changes.
The output schema becomes `additive_chain_saved_raw_generations_v1`; its
evaluator scores every raw attempt and reports accepted/rejected subsets,
including explicit undefined scores for sequences without non-EOS targets.
This prevents an acceptance rule from hiding failures in the quality result.

Accepted text uses `skip_special_tokens=False`. As in the experiment sampler,
rejected attempts retain raw IDs but are not decoded inside the timed attempt.
The portable evaluation schema records all raw attempt IDs, acceptance, actual
new token count, duration, seed keys, measured forward counts and token
diagnostics. It can be passed directly to `evaluation.evaluate_saved`.

The per-attempt timer synchronizes before/after and includes token-window and
generator preparation, all forwards, candidate preparation, dynamic pair
preparation, DP, sampling, acceptance checking, accepted-text decoding and
diagnostics. It excludes serialization and external quality scoring.
`setup_seconds` measures explicit lookup/table preparation; constructing counts,
heads or the backbone before `prepare()` is separate model preparation/loading.
Warmup uses the fixed excluded raw attempt ID1,000,000,000; actual attempts
start at zero and their explicit streams are unaffected. Warmup may trigger
shape-specific kernel compilation, so fresh hardware/runtime timing still
requires measurement rather than copying old numbers.

These timing categories include CPU token diagnostics and accepted-text
decoding, as did the experiment runner. They are **not identical timing
implementations**: this core omits that runner's component profiling, progress
callbacks, durable runtime ledger, and extra full-window validation. It also
does not estimate unrecorded work after an interrupted process. Compare methods
within one declared implementation/runtime, and measure this entrypoint anew;
do not splice its durations into previously reported experiment timings.

This is a minimal **Python API**. The separate [CLI](CLI.md) supplies authenticated
asset loading and output serialization. Neither entrypoint includes on-disk
durable attempt commits, resume after interruption, or automatic experimental
selection. Direct API callers load existing local assets, verify their
identities, and save output themselves. Exceptions propagate; the API does not
fabricate a finished record after an interrupted attempt. No downloads or
checkpoint writes occur in this module.

## Tests and current validation scope

```bash
python -m pytest -q tests/test_additive_chain_generation.py
```

CPU tests compare the exact finite schedule, baseline draws and final RNG states
to an independent original-loop transcription; test empty scheduled steps,
zero coefficients, frozen visible context, one-forward feature composition,
selected-only own-marginal draws, static/dynamic preparation separation, raw
attempt numbering, warmup and retry accounting. Active CUDA adapters are mocked
in those CPU integration tests, while tiny reference marginals check the law.
Those CPU tests alone do not establish real-checkpoint CUDA parity.

A separate A100 diagnostic at source revision `f5b1575` compared the public
`ChainDecoder.reveal` API with the frozen research implementation using the
released MDLM checkpoint. It checked exactly one fixed raw attempt (index 0)
for each of baseline, count, global and contextual decoding: length 1,024,
32 steps, top-p 0.9 and K500, with joint sampling and the static backend for
active pair models. Count strength was 0.1; active global and contextual
strengths were 0.25. The contextual value
was a diagnostic setting, not a selected quality result.

All four trajectories passed bitwise checks for candidate IDs, FP64 unaries,
active head factors, sampled tokens and proposal-generator state at every
step (128 step records). The fixed count/global tables also matched bitwise.
Two separate initial diagnostic forwards matched the native and public
feature wrappers; each sampled step then shared one frozen masked-input
forward between the two decoders.

The **native scheduler** drove these shadows. This does not validate the
public standalone scheduler/attempt runner, checkpoint-loader startup, CLI,
external scorer or training pipeline on a full model. It adds no quality
population and measures no generation speed. The diagnostic held both
implementations' caches simultaneously, so its memory use is not a deployment
benchmark. Rejection, warmup, serialization and end-to-end timing still need
their own composed reproduction checks.
