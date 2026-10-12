# Adaptation API and executable training

`adaptation_step` connects the frozen model, candidate support, additive head,
and revealed-block objective. It does not load a dataset, run an optimizer,
save checkpoints, or select strengths. The separate `train_cli` below supplies
the optimizer, checkpointing, development selection and continuation. These
are tested portable scientific implementations, not a claim to reproduce
full-model training numerics, quality, or runtime.

For each clean packed row, it draws a step `j` uniformly from `1..T`, masks each
position with probability `j/T`, then selects masked positions for reveal with
probability `1/j`. The default is `T=32`. No window BOS/EOS is added; the first
position can be masked. An empty reveal set returns `loss=None` without calling
the backbone. Skip the optimizer update in that case—do not force a target.

The backbone sees only the corrupted row and time. Its logits and hidden states
come from **one frozen forward pass**, using `FrozenMDLM.forward` or an equivalent
interface. All hidden positions retain candidate supports; unrevealed positions
are integrated out, not clamped to gold. Candidates use the existing
exclude-crossing top-p convention, intersected with top-K (`.9` and `500` by
default). Only missing gold at selected reveal targets is added, preserving all
original candidates and their relative probabilities. Width may become K+1.
Set `missing='error'` to reject missing targets instead.

The loss is

`[log Z(unary, pairs) - log Z(unary clamped only at selected targets, pairs)] / |R|`.

For multiple rows the numerator is summed and divided by the total number of
selected tokens, not averaged by row. This is a **candidate-set training/dev
surrogate, not held-out data perplexity**. Generation never adds gold tokens.

## Optional unchanged-support objective

`supported_training.prepare_supported_targets` and `supported_block_loss`
provide an alternative, explicit partial-target objective. They do **not**
change `adaptation_step` or `train_cli`; the default remains the gold-augmented
objective above. On the original candidate sets `C`, let
`I = {i in R: gold_i is already in C_i}`. The alternative loss is
`-log q_C(gold_I) / |R|`: keep the original requested-target denominator, clamp
only `I`, and marginalize every other masked position, including missing
targets in `R`. No gold candidate is inserted, removed, or reordered. The
returned `per_supported_target_nll` uses `|I|` instead for diagnostics, not as
an implicit replacement training weight.

Callers build pair scores on `targets.support` and own the optimizer:

```python
from additive_chain.supported_training import prepare_supported_targets, supported_block_loss

targets = prepare_supported_targets(candidates, gold_tokens=clean,
    reveal_mask=reveal, visible_mask=visible)
value = supported_block_loss(targets, pair_scores)
optimizer.zero_grad(set_to_none=True)
if value['should_update']:
    value['per_original_selected_token_nll'].backward()
    optimizer.step()
```

When `I` is empty, **skip `optimizer.step()` entirely**. A zero-valued loss does
not prevent Adam momentum or weight decay from changing parameters. In a batch
with some supported targets, the denominator still includes all requested
targets, and one update uses the pooled objective. This API neither supplies
supervision for unavailable targets nor estimates full-data PPL. Its existence
and correctness tests do not establish improved generation quality or speed.

## Count and learned scores

Construct `SparseCountPotential.from_counts(..., strength=count_strength)` once
from training-only counts. The API receives this already-powered potential;
there is no second count-strength multiplication. Its lookup omits the shared
`log_scale` constant, as the inference backend does. This common per-edge
constant cancels in the partition ratio. Adding it back describes the same law,
but is not claimed to give bitwise-identical floating-point objectives.

For a global adaptation step, `head` is a trainable `GlobalAdditiveHead`. For a
contextual step it is a trainable `ContextualAdditiveHead`; pass the already
frozen/eval global predecessor as `global_head` and its explicitly chosen
`global_strength`. The new residual's training coefficient is one. Head
constructors retain exact zero-residual initialization and no additional rank
scaling. At zero initialization the stage recovers its predecessor's scores and
loss on the same prepared support. The API never changes a predecessor's mode,
parameters, or gradients, and never changes optimizer state.

Use a properly authenticated checkpoint/predecessor and development-selected
coefficients in an experiment. This in-memory step does not establish that
asset or selection provenance on the caller's behalf.

## Runnable toy example

This example exercises the API without downloading data or loading model weights.
Replace `TinyFrozen` with the public `FrozenMDLM` for actual adaptation.

```python
from types import SimpleNamespace
import torch
from additive_chain.adaptation import adaptation_step
from additive_chain.counts import SparseCountPotential
from additive_chain.heads import GlobalAdditiveHead

class TinyFrozen:
    mask_id = 3
    def forward(self, tokens, t):
        logits = torch.tensor([.5, .2, -.3, -torch.inf], device=tokens.device)
        return SimpleNamespace(log_probs=logits.expand(*tokens.shape, 4),
                               hidden=torch.zeros(*tokens.shape, 2, device=tokens.device))

counts = dict(left_counts=torch.tensor([2., 1., 0., 0.], dtype=torch.float64),
              right_counts=torch.tensor([0., 2., 1., 0.], dtype=torch.float64),
              pair_keys=torch.tensor([1, 6]),
              pair_counts=torch.tensor([2., 1.], dtype=torch.float64))
potential = SparseCountPotential.from_counts(**counts, strength=.1)
head = GlobalAdditiveHead(vocab_size=4, rank=2)
result = adaptation_step(torch.tensor([[0, 1, 2]]), frozen=TinyFrozen(),
    count_potential=potential, head=head,
    generator=torch.Generator().manual_seed(17), schedule_steps=1, top_k=2)
if result.loss is not None:
    result.loss.backward()  # Caller owns optimizer/reset/step and checkpointing.
```

`loss_on_corruption` exposes the same step for fixed-corruption debugging. It
requires observed tokens to equal the clean input and reveal targets to be
masked. `draw_corruption` is also available independently for schedule checks.

## Implementation scope

The corruption draw order, fast support extraction, selected-only augmentation,
FP32 pair/unary DP inputs, coefficient-one residual, and per-selected-token
normalization follow the adaptation formulation used in the experiments.
The count lookup here uses portable chunked `searchsorted`, not a Triton hash
lookup; the DP uses the public exact dense reference, not an optimized training
backend. Low-rank pair products use the public composition function. These
operations describe the same fixed-support law, but CPU tests do not establish
full-model CUDA bitwise parity, training throughput, or final quality.

The dense `[B,L-1,K,K]` pair tensor and its gradients require quadratic memory
in K. `edge_chunk` only bounds lookup temporaries; it is not a bound on total
training memory. This API does not prune unrevealed latent components or alter
the objective to reduce that cost. Packed-data provenance remains the caller's
responsibility. The CLI below provides an explicit training order, validation
population, optimizer, stopping policy, and durable resume state.

## Local-assets training command

Run `python -m additive_chain.train_cli --help` in a fresh Python process.
Nothing is downloaded. Supply the three local directories described in
[BACKBONE.md](BACKBONE.md), a SHA-bound count artifact from
`python -m additive_chain.prepare_counts`, and separate packed train/dev
manifests. `--output` and a finite `--seconds-limit` are required; there is no
default checkpoint destination. Use an output filesystem with enough space
for optimizer state and immutable best-head snapshots. The command does not
copy the frozen pretrained checkpoint into the output directory.

```sh
python -m additive_chain.train_cli \
  --upstream /data/bd3-lm --checkpoint /data/mdlm-owt-noeos \
  --tokenizer /data/gpt2 \
  --train-manifest /data/packed/train-manifest.json \
  --train-manifest-sha256 TRAIN_MANIFEST_SHA256 \
  --dev-manifest /data/packed/dev-manifest.json \
  --dev-manifest-sha256 DEV_MANIFEST_SHA256 \
  --count-file /data/counts/counts.pt --count-sha256 COUNT_SHA256 \
  --kind global --count-strength .1 --rank 64 --top-k 500 \
  --output /data/runs/global --seconds-limit 36000
```

The paths and uppercase hashes above are placeholders, not release assets.
The defaults reproduce the declared training protocol:

- One row per update, signed FP32 rank-64 head, coefficient one during training.
- Conditional count smoothing `.1`, explicit count strength, top-p `.9`, cap
  `500`, and the 32-step corruption/reveal schedule described above.
- AdamW at learning rate `.001`, zero weight decay, gradient-norm clipping at
  `1`, no learning-rate schedule, no gradient accumulation.
- A fixed CPU permutation of training rows, reused cyclically without a new
  epoch shuffle. Fixed initialization/order/corruption/dev RNG streams are
  recorded; they are reproducibility state, not repeated-seed experiments.
- Evaluate the **first 32 declared dev rows**, using the same corruption draws
  at every evaluation. Pool block NLL and divide by the total selected tokens.
  Never read a test split or exponentiate this surrogate as data PPL.
- Record initial development loss at step zero. Evaluate every 250 optimizer
  updates and at the maximum, select a new trained best only for improvement
  greater than `1e-5`, and stop after four non-improving evaluations once at
  least 1,000 updates have run, or at 5,000 updates. Step-zero loss is a
  diagnostic, not a trained best-head candidate. Generation strength zero is
  the separate, exact predecessor option during subsequent calibration.

All these settings are recorded in the training identity. CLI overrides are
explicitly different configurations, not silent reproductions. There is no
seed-sweep option. The public loader/initialization and dense FP32 operation
order are not asserted bitwise equal to the optimized experimental trainer.

### Packed inputs

Each manifest has schema `packed_chain_adaptation_v1`, split `train` or `dev`,
and these fields (use different files/data for the two splits):

```json
{
  "schema": "packed_chain_adaptation_v1",
  "split": "train",
  "vocab_size": 50258,
  "window_length": 1024,
  "mask_id": 50257,
  "eos_id": 50256,
  "tokenizer": {
    "id": "gpt2",
    "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e"
  },
  "document_eos_retained": true,
  "window_bos_eos_added": false,
  "shards": [{"path": "train.npy", "sha256": "SHA256_OF_NPY_FILE", "rows": 9765}]
}
```

Shards are memory-mapped integer `[rows,1024]` `.npy` arrays, relative to the
manifest directory or explicitly absolute. All bytes, shapes, token ranges,
packing declarations, and tokenizer revision are checked. Train/dev cannot
share a resolved shard path or identical shard bytes. **This does not prove
document-disjointness or absence of partial overlap:** audit corpus identities
and document splits during preprocessing. Article EOS tokens and the first
window token are retained in training, and no window wrappers are inserted.
The original experiment used 9,765 training windows and evaluated the first
32 rows of a separate 2,130-row development population. Different input
manifests produce a new experiment, even when optimization settings match.

### Frozen predecessor and exports

For the contextual stage, use `--kind contextual --global-head HEAD.pt
--global-sha256 HEAD_SHA256 --global-strength .25`, with an explicitly chosen
global checkpoint and coefficient. The head must be an authenticated global
export with the same count SHA, count strength, candidate rule and schedule.
Exports from this trainer include that count binding. Older head formats that
lack it can still be consumed by the lower-level asset API, but this training
entry point rejects them rather than inventing their count provenance.
The global head is evaluation-only and gradient-disabled. The frozen token
embedding table and hidden states from the same masked-input pass supply the
new contextual head; no gold-conditioned encoder pass is made. Contextual
training also uses new-head coefficient one, with the inherited global
coefficient fixed throughout.

`last.pt` is the authoritative resume checkpoint: parameters, AdamW state,
training cursor, corruption and process RNG states, complete development
history, stopping state, and byte-bound input/source identity. The command
also writes `identity.json`, `progress.json`, and, on a normal bounded stop,
`training-history.json`. Each selected improvement creates an immutable
`head-stepNNNNNN.pt` and updates `best-head.pt`. Both trained head exports use
the existing `additive_ladder_head_v1` format, so they can be passed directly
to `generate_cli` with their SHA256. The initial optimizer checkpoint is
written before any development pass or preflight, even if that pass fails.

Repeat the exact command with `--resume` to continue the same run. Resume
rejects changed scientific configuration, data/count/head/source bytes, or
runtime identity. `--max-new-updates 4` bounds a genuine four-update slice;
removing it while retaining `--resume` continues the same optimizer/RNG state.
`--phase preflight` performs one zero-step forward/backward gradient check,
without an optimizer update or advancing the saved training RNG. Resume a
successful preflight with `--phase train --resume`.

Checkpoints are atomically replaced and output has an exclusive process lock.
Before scheduled evaluation the current optimizer checkpoint is durable;
selection history is committed only after the best snapshot is durable.
An interrupted evaluation/export is replayed at the current saved step before
another update, including at the maximum step. Older missing evaluation events
are rejected because their historical weights cannot be reconstructed. A hard
kill or unexpected training failure resumes from the last complete checkpoint
(up to 25 updates may be recomputed with the same saved state); a graceful
SIGTERM/SIGINT or budget stop saves at a row boundary. The time budget is checked
between operations, not an operating-system deadline interrupting a kernel.

**Selection scope:** this command selects a training checkpoint by candidate-set
development NLL. It does not run the separate generation-PPL coefficient grid,
choose its coefficient, produce held-out data PPL, or assert improvement.
Evaluate the chosen head and coefficient zero alongside a declared coefficient
grid using a separate development-generation population before final testing.
