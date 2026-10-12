# One adaptation step

`adaptation_step` connects the frozen model, candidate support, additive head,
and revealed-block objective. It does not load a dataset, run an optimizer,
save checkpoints, or select strengths. This is a tested scientific building
block, not a claim to reproduce full-model training or its runtime.

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
the objective to reduce that cost. The caller owns packed-data provenance,
training order, validation, optimizer settings, stopping, and resume state.
