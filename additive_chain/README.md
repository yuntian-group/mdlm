# Additive token chains

This package supplies a PyTorch reference implementation of a linear-chain
distribution over output token candidates. Its mathematical core is independent
of model loading, datasets, experiment scheduling, and checkpoint storage.
The optional [released-backbone adapter](BACKBONE.md) supplies the native MDLM
log probabilities and masked-input features used by the experiment pipeline.
The [saved-generation evaluator](EVALUATION.md) reports external-model PPL,
rejection-aware generation time, and raw-token diagnostics without regenerating
or filtering the saved outputs.
The [finite-step generation API](GENERATION.md) connects the frozen backbone
to baseline, joint-chain, and own-marginal proposals. It returns raw accepted
and rejected attempts in the evaluator's format; checkpoint imports and a
durable command-line runner are separate from this API.
The [local-assets generation command](CLI.md) supplies strict count/head
imports and executable baseline/count/global/contextual wiring. It saves a
finite run, but does not provide durable resume or claim CUDA reproduction.
The [count-building command](COUNT_DATA.md) constructs compatible sparse
statistics from authenticated, already packed training windows.
The [adaptation-step API](ADAPTATION.md) connects masked-input features and
additive residuals to the selected-block training loss, without a job runner.

For a fixed partially masked input `c`, the score is

\[
\log q(x\mid c)=\sum_i u_i(x_i)
 +\sum_i\{\lambda C(x_i,x_{i+1})
          +\mu A(x_i)^\top B(x_{i+1})
          +\nu U_i(x_i,c)^\top V_{i+1}(x_{i+1},c)\}
 -\log Z(c).
\]

`u` contains the frozen denoiser's unary log scores. `C(a,b)` is an explicitly
chosen smoothed count log conditional, `log p_count(b|a)`. The package accepts
already looked-up candidate-pair count scores; it does not impose a smoothing
scheme or allocate a full-vocabulary count matrix. All three strengths are
external coefficients. Setting a stage's coefficient to zero removes it.

The learned residuals are **signed log scores**. Low rank constrains their dot
products; it does not replace `exp(A @ B.T)` with a low-rank positive matrix.
There is no softmax, softplus, or additional rank scaling inside a head.

## Interface and shapes

- Candidate IDs and unary scores: `[batch, length, candidates]`.
- Pair scores: `[batch, length-1, source_candidate, destination_candidate]`.
- Node factors: `[batch, length, candidates, rank]`; edge `i` uses left `i`
  and right `i+1`.
- Candidate padding: ID `vocab_size`, unary `-inf`. A visible token is a
  singleton support, not a removed position. Optional sequence lengths ignore
  padded positions and edges.
- `sample_ffbs` returns candidate slots; `sample_tokens` returns token IDs.
  Both return `[num_samples, batch, length]` and use `-1` for sequence padding.

`compose_pair_scores` combines count, global, and contextual scores under this
single convention. `log_partition`, `marginals`, `joint_nll`, and joint
forward-filtering/backward-sampling consume the resulting distribution.
Unrevealed positions remain latent: they are not cut out of the chain.

This implementation materializes dense candidate-pair scores and costs
`O(batch * length * candidates²)`. It is a correctness/training reference, not
a claim of optimized generation speed. The explicit opt-in
[CUDA backend](CUDA_BACKEND.md) evaluates count and learned pair scores inside
fused kernels without materializing that tensor. Its documentation covers
candidate preparation, supported devices, provenance, and separate validation
requirements. No optional fused kernel or candidate-selection helper is enabled
implicitly.

For frozen count/global scores, the optional [static backend](STATIC_BACKEND.md)
caches their full-vocabulary table before generation. The
[cached contextual backend](STATIC_CONTEXTUAL_BACKEND.md) adds fresh local
factors to that same fixed table at each step. These are explicit alternatives,
not changes to the reference model's probability law.

## Initialization and training

`GlobalAdditiveHead` initializes its left embedding with standard deviation
`1/sqrt(rank)` and its right embedding to zero. The contextual head clones a
frozen token embedding table, detaches masked-input hidden states, and starts
its right token projection and shift at zero. Thus each new stage initially
preserves the preceding distribution exactly. The zero side has nonzero
first-step gradients; the nonzero side's initial gradient is zero as expected.
The contextual shifts can also change individual token preferences; the head
is not a dependency-only capacity control.

Supply hidden states from the **same masked input** used to obtain the base
logits. Do not recompute them from gold-filled inputs. Freeze the base model
throughout and freeze the global head when training a separate contextual
residual. The heads use FP32 parameters; dense DP supports FP32 and FP64.

`block_reveal_nll` computes the selected-set probability by subtracting two
log partitions: clamp only the newly revealed positions in the numerator,
while marginalizing every other masked position. The default `log_partition`
backward returns exact first-order gradients without retaining the entire
dynamic-programming graph. Use `memory_efficient=False` for higher derivatives.

`prepare_training_support` rejects missing selected targets by default. Explicit
`missing="inject"` adds missing selected gold tokens without removing existing
candidates. This is a **training/development surrogate**, not the likelihood of
the original generation support. It reports the support-miss fraction and
injection count. Construct count scores and factors only after preparing the
support; numerator and denominator must share them.

A hard-truncated distribution can assign zero probability to held-out targets.
Neither gold injection nor normalization by the number of revealed tokens turns
the resulting objective into a diffusion likelihood bound or data perplexity.
This package intentionally does not label these quantities as data PPL.

## Unary-only capacity control

`IndependentAdditiveHead` adds a learned score to each position's unary without
adding or changing any pair score. It uses detached masked-input hidden states
and a zero-initialized output projection. `choose_width` reports the nearest
width to a requested trainable-parameter budget and the exact mismatch. For
example, a full-vocabulary width-125 control with vocabulary size 50,258 and
hidden size 768 matches the parameter count of a rank-64 global head within
0.1%. An optional frozen-embedding output projection supports smaller budgets.

Use `add_independent_residual(unary, ids, hidden, head, strength)` and keep the
inherited count/global pair tensor unchanged. Only the **new residual** is
unary-only: a model retaining count or global pair scores is not factorized.
With zero pair scores, it is a factorized control. These controls distinguish
extra capacity/individual-token changes from additional pairwise interactions;
parameter matching alone does not match training compute or architecture.

## Minimal example

```python
import torch
from additive_chain import (
    GlobalAdditiveHead, block_reveal_nll, compose_pair_scores,
    prepare_training_support, sample_tokens,
)

ids = torch.tensor([[[0, 1], [1, 2], [0, 2]]])
unary = torch.tensor([[[.2, -.1], [.1, .4], [-.2, .3]]])
gold = torch.tensor([[0, 999, 2]])  # the unselected middle value is ignored
reveal = torch.tensor([[True, False, True]])
support = prepare_training_support(
    ids, unary, vocab_size=3, gold_tokens=gold, reveal_mask=reveal,
)
head = GlobalAdditiveHead(vocab_size=3, rank=8)
count_table = torch.tensor([[.6, .3, .1], [.2, .6, .2], [.1, .3, .6]]).log()
count = count_table[support.candidate_ids[:, :-1, :, None],
                    support.candidate_ids[:, 1:, None, :]]
pair = compose_pair_scores(
    count, count_strength=.1,
    global_factors=head(support.candidate_ids), global_strength=1.,
)
loss = block_reveal_nll(
    support.log_unary, pair, support.gold_slots, support.reveal_mask,
)
loss.total_block_nll.backward()
tokens = sample_tokens(ids, unary, pair, vocab_size=3, num_samples=4)
```

The tiny dense count table illustrates the lookup only; a large-vocabulary
application should provide sparse count lookups rather than build that table.
For contextual composition, additionally pass `contextual_factors=local(ids,h)`
and `contextual_strength=...` to the same composition function.

## Verification and provenance

This core uses standard linear-chain sum-product inference and backward
sampling, implemented directly in PyTorch. Its tests enumerate tiny complete
distributions and compare partitions, marginals, gradients, selected-block
probabilities, and sampling behavior. Additional tests cover hard constraints,
variable lengths, zero-initialized nesting, and frozen contextual inputs.
No external checkpoint, network service, or GPU is needed for the core tests:

```bash
python -m pytest -q tests/test_additive_chain_*.py
```

Python 3.9 or newer, PyTorch 2.x, and pytest are sufficient for these tests.
The package follows the repository's license. The exact dense reference does
not imply full-vocabulary inference when its caller provides truncated support,
nor does passing these tests establish downstream generation quality.
