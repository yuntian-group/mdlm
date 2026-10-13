# Run generation from local assets

`python -m additive_chain.generate_cli` connects the published backbone,
asset importer, and finite-step generation API. It performs no downloads and
does not invent locations for count statistics or learned heads. Prepare the
three pinned backbone assets described in [BACKBONE.md](BACKBONE.md) and obtain
count/head exports from a compatible training run. Count and head weights are
not bundled with this command.

Use a fresh Python process with the dependencies listed in BACKBONE.md.
Baseline generation can use CPU for a smoke test; active pair inference needs
NVIDIA CUDA and Triton. This command does not use the older root `main.py` model
or its default configuration.

## Commands

Baseline, with a declared stream and a new output file:

```bash
python -m additive_chain.generate_cli \
  --upstream /path/to/bd3lms \
  --checkpoint /path/to/mdlm-owt-noeos-snapshot \
  --tokenizer /path/to/gpt2-tokenizer-snapshot \
  --output baseline.json --samples 64 \
  --steps 32 --top-k 500 --top-p .9 \
  --stream-namespace my-development-stream --fast-candidates
```

The default `--population accepted` requests 64 accepted outputs, retaining
their rejected attempts too. Add `--population raw` to request exactly 64 raw
outputs instead: no rejection replacement and no ten-rejection stopping limit.
This does not change individual trajectories or their acceptance flags. The
saved schema tells the evaluator to score all raw attempts, with separate
accepted/rejected diagnostics and explicit undefined scores when necessary.

Add these arguments for a count chain, using the actual SHA256 of your file:

```bash
--count-file /path/to/counts.pt --count-sha256 YOUR_COUNTS_SHA256 \
--count-mode conditional --smoothing .1 --count-strength .1
```

Add a global residual:

```bash
--global-head /path/to/global-head.pt --global-sha256 YOUR_GLOBAL_SHA256 \
--global-strength .25
```

Add a contextual residual **and its exact inherited global export**:

```bash
--contextual-head /path/to/contextual-head.pt \
--contextual-sha256 YOUR_CONTEXTUAL_SHA256 --contextual-strength 1
```

These coefficients demonstrate syntax, not a recommendation or a new
validation-selected result. Every supplied file needs a matching SHA, even
when its coefficient is zero. A positive coefficient without an asset fails.
Count strength is applied once: the stored count file's `strength` is never
multiplied into the CLI strength. The stored settings and requested settings
are both recorded.

### Independent frequency and temperature controls

Use the baseline command with `--population raw` for a fixed-size comparison.
These options leave the original top-p/cap support unchanged:

```bash
# Frequency only: no count-pair inference, despite reading the count asset.
--count-file /path/to/counts.pt --count-sha256 YOUR_COUNTS_SHA256 \
--frequency-strength .1 --smoothing .1 --count-strength 0

# Temperature only: no count asset required.
--temperature .95
```

The frequency term is `alpha * log J_R(v)` at original positions `i>0`,
where `J_R` is the actual destination marginal of the smoothed count joint in
`count_controls.py`. It is not merely the add-one backoff distribution and
does not exempt the first *selected* position. Temperature divides the retained
unaries only **after** the original nucleus and cap have been selected; it is
not a temperature applied before top-p. Neither control loads a learned head,
builds a pair table, or runs dynamic programming.

If both options are requested, their explicit composition is
`q_i(v) proportional to exp((u_i(v) + alpha*1[i>0]*log J_R(v))/temperature)`.
The reported experiments used frequency or temperature separately; the combined
law is supported by the API but has not been measured in those experiments.
Active independent controls cannot be combined with nonzero count/global/
contextual pair strengths; such a request fails rather than dropping an input.
Identity settings `--frequency-strength 0 --temperature 1` follow the exact
baseline sampling path. Control settings and order are saved with the output.

These commands demonstrate the interface, not a new validation selection or
an assertion that the public port reproduces historical GPU timings. Compare
the complete saved raw population, including failed entropy checks, and verify
trajectory/runtime parity on the intended checkpoint and GPU before claiming it.

Use `--sampling own-marginal` to draw selected positions independently from
the **same chain's** singleton marginals. It does not remove pair scores or
retune the model. `--backend static` explicitly requests a fixed count/global
table; contextual factors remain dynamic. The default is `fused`. Static
memory limits are `--table-gib 12 --scratch-mib 128 --reserve-gib 2`; see
[STATIC_BACKEND.md](STATIC_BACKEND.md) and
[STATIC_CONTEXTUAL_BACKEND.md](STATIC_CONTEXTUAL_BACKEND.md) before allocating
a full-vocabulary table.

`--allow-recalibration` is required when an active head's training count
strength, K, top-p or scheduled steps differ from generation, or when changing
its inherited global coefficient. For example, a head trained on a 32-step
surrogate can be measured at 16 steps with this flag; the difference is saved.
It never permits replacing a contextual head's inherited global weights.
Changing a learned residual's own external coefficient is ordinary calibration
and does not change the imported parameters. Existing head exports were trained
with conditional counts and smoothing .1; requesting PMI or another smoothing
value also requires the flag and is recorded.

## Exact imported schemas

Assets use `torch.load(..., map_location="cpu", weights_only=True)`. Import
checks the supplied hash before and after loading. There is no unrestricted
pickle fallback. Use files from a trusted source; hash checking verifies which
bytes you received, not who trained them or their scientific provenance.
The importer refuses files larger than 2 GiB and does not accept an optimizer
checkpoint as a head.

The count dictionary has exactly these fields:

```python
{
    "format": "chain_crf_counts_v1",
    "vocab_size": 50258,
    "mode": "conditional",       # or "pmi"; stored metadata
    "strength": 1.0,             # stored metadata, not implicitly applied
    "smoothing": 0.1,            # stored metadata
    "left_counts": ...,         # FP64 [V]
    "right_counts": ...,        # FP64 [V]
    "pair_keys": ...,           # int64 [E], strictly increasing a*V+b
    "pair_counts": ...,         # FP64 [E], positive observed integer counts
}
```

The four tensors are detached and dense. Counts must be finite,
integer-valued and nonnegative; endpoint counts must exactly equal the pair
marginals. No counted edge may contain the implicit MASK/PAD token. The file
does not specify which documents or split were used, so the importer explicitly
records `training_only_data_provenance_verified=false`. Build counts from
training text only and retain the dataset/tokenization/boundary record yourself.
For these experiments the actual conditional is the independence-backoff
mixture documented in `counts.py`, not an undocumented add-one conditional.

The learned export has exactly these top-level fields:

```python
{
    "schema": "additive_ladder_head_v1",
    "kind": "global",           # or "contextual"
    "head_config": {...},
    "head_state_dict": {...},
    "model_revision": "b9613c3c54da477d198c998e767ac3647e01225b",
    "training_identity": {...},
    "training_step": 500,        # positive, actual completed update
}
```

Global config is exactly `vocab_size, rank`; contextual config adds
`hidden_size, context_size`. State keys/shapes must match the corresponding
public head exactly, with finite FP32 tensors. A contextual state includes its
frozen `token_embeddings` buffer, which is also compared to the loaded
backbone before generation. No positive-mixture, scratch-CRF or unrelated
legacy checkpoint is translated into a signed additive head.

`training_identity` must describe the same kind/config and model revision
under schema `additive_ladder_training_identity_v1`. It declares count strength,
cap, top-p, schedule steps, the joint-block candidate-set surrogate (explicitly
**not data PPL**), and selected-target-only gold injection during training.
A global head has no inherited global export. A contextual head records the
inherited global `sha256`, `step`, and training global coefficient. Additional
training metadata is read but is not copied wholesale into the saved sample
file. The SHA of the imported export binds that original metadata.

The same head interchange also accepts `supported_training_continuation_v1`.
That identity is preserved, not relabeled as gold-augmented training. Its
`pilot_protocol` must declare unchanged generation support, no gold insertion,
missing targets left latent, and `-log q_C(gold_I)/|R|`, with empty supervision
skipping the optimizer entirely. The importer checks its head configuration,
model revision, fixed selection cases/policy, and exact tensor keys/shapes.
Generation reads its nested support/count/schedule anchors; a contextual head
must retain its exact inherited global export SHA. The saved asset summary
records which training schema was imported.

This import support does not change `train_cli`'s gold-augmented recipe or
reproduce the continuation's 128-case selection procedure. Using an imported
head as a *training* predecessor additionally requires an explicit direct
count-asset SHA. An older export lacking that field may be used for generation
but is rejected as a training predecessor; data/manifest hashes are not
silently substituted for the missing count-file hash.

The importer preserves the CPU RNG state while constructing/loading heads,
and returns frozen eval-mode modules. Generation uses the two explicit stream
generators documented in [GENERATION.md](GENERATION.md), so loading a head
does not select a different sample population.

## Output, timing and limits

The command always generates one 1,024-token window. T, top-p, K, all
coefficients, sampling selector, backend, candidate/pruning settings and
stream namespace are saved. It retains internal EOS, the initial boundary
token, every raw accepted/rejected attempt and the original entropy rejection
rule. It does not truncate or rerank samples. The same saved format feeds
[evaluation.py](EVALUATION.md); no scorer runs during generation.

Model loading, asset validation, count/head construction and fixed-table setup
are outside attempt timing. Warmup is enabled by default and reported
separately; `--no-warmup` disables it explicitly. Every rejected attempt's
generation time remains in the output. An output file must not already exist.
In accepted mode, ten consecutive rejections save `status="retry_limit"` and
produce exit code2. In raw mode, completion means the requested number of raw
attempts has been saved, regardless of acceptance. Successful completion
returns0. No interrupted configuration is relabeled complete.

This first command has no durable per-attempt ledger or resume mechanism.
An exception or killed process before serialization does **not** save its
partial attempts or reconstruct interrupted timing. Use a new output file for
each deliberate run; do not label an interrupted run's surviving fragments as
a completed population. It does not select a strength or automatically label
a stream development/final; that experimental decision remains the caller's.

CPU tests cover schemas, composed command routing and error cleanup. An actual
checkpoint/CUDA smoke check ran the unmodified generation command and separate
evaluation command for baseline and count strength 0.1: four fixed raw attempts
per arm, length 1,024, T=32, top-p=0.9, K=500, static backend. All eight final
token arrays exactly matched their saved reference outputs, and each separate
scoring output bound the generated input bytes. This is software execution and
saved-output replay evidence, not a quality estimate, timing benchmark,
training check, or learned-head CLI validation. The
[earlier decoder shadow check](GENERATION.md#tests-and-current-validation-scope)
has its own, separate scope. Successful execution does not substitute for
backend law tests or fresh same-device timing.
It never writes a fabricated proof or claims existing experiment receipts
apply to its new wrapper/runtime.

```bash
python -m pytest -q tests/test_additive_chain_assets.py \
  tests/test_additive_chain_generate_cli.py
python -m additive_chain.generate_cli --help
```
