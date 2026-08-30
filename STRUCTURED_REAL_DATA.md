# Contextual-forest real-data bridge

This path is opt-in: ordinary `dit`, `crf_dit`, AR, and DiMamba configurations
do not construct or optimize the structured head.

## What it trains

For each real text batch, the existing absorbing forward process produces
`x_t`. The DIT produces hidden states and ordinary unary logits once, then the
contextual-forest head receives only `(hidden_states, unary_logits, sigma,
masked_positions)`. Candidate sets are therefore target-independent. Clean
tokens enter only the training likelihood after candidates and topology have
already been constructed.

The primary metric and objective are explicitly named **conditional denoising
NLL per masked token**. They are not presented as the original MDLM diffusion
ELBO. Generic NLL/BPD/perplexity accumulators are disabled for structured runs
so this conditional objective cannot be mistaken for a likelihood estimate.
Tail tokens retain full support through the exact residual state.
The configured `objective_name` is validated at startup and cannot be relabeled
as an ELBO accidentally.

Dynamic topology uses `gold_reveal_influence` distillation. During training
only, one masked source token per example is revealed to a detached backbone
teacher. The change in clean-token log probability at other masked positions
supervises anchor occupancy, source-to-anchor slot routing, and sparse edge
scores. At inference the teacher and clean sequence are absent; the hard forest
depends only on the corrupted context and time.

## First executable checks

Run the Torch-only bridge tests:

```bash
python -m pytest -q \
  tests/test_structured_training.py \
  tests/test_structured_objective.py \
  tests/test_structured_forest.py
```

Run a two-step random-backbone text8 plumbing smoke after installing the full
repository environment. The command assumes exactly one visible GPU; setting
`loader.num_workers=0` is supported (persistent workers are enabled only when
the worker count is positive):

```bash
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  model=contextual-forest-tiny \
  data=text8 \
  data.cache_dir=/mnt/contextual-forest/data \
  trainer.accelerator=cuda \
  trainer.devices=1 \
  trainer.max_steps=2 \
  trainer.val_check_interval=1 \
  trainer.limit_val_batches=1 \
  trainer.num_sanity_val_steps=1 \
  loader.global_batch_size=2 \
  loader.eval_global_batch_size=2 \
  loader.batch_size=2 \
  loader.eval_batch_size=2 \
  loader.num_workers=0 \
  eval.generate_samples=false \
  training.ema=0 \
  checkpointing.resume_from_ckpt=false \
  wandb=null
```

This smoke only verifies wiring. It is not evidence for the paper.

## Frozen OpenWebText adapter screen

Use a trusted released MDLM Lightning checkpoint containing `backbone.*`
weights and its EMA state. The small configuration installs the EMA backbone
parameters by default, matching normal MDLM evaluation:

```bash
python main.py \
  model=contextual-forest-small \
  data=openwebtext \
  model.structured_decoder.training.backbone_checkpoint=/mnt/checkpoints/mdlm-owt.ckpt \
  eval.generate_samples=false \
  checkpointing.resume_from_ckpt=false
```

The small configuration refuses to start a fresh frozen-backbone run without
that checkpoint. To lightly tune the backbone rather than freeze it, override:

```bash
model.structured_decoder.training.backbone_mode=joint \
model.structured_decoder.training.backbone_lr_multiplier=0.05
```

If the backbone checkpoint has no compatible EMA shadows, startup fails with a
shape/count diagnostic. Deliberately use raw checkpoint weights with
`model.structured_decoder.training.use_ema_backbone=false`; full-checkpoint
evaluation without EMA additionally requires `eval.disable_ema=true`.
EMA shadows in the upstream Lightning format are positional rather than named,
so EMA loading is restricted to trusted checkpoints produced by the same code
and parameter ordering. Shape/count checks cannot detect a permutation among
equal-shaped tensors.

The checkpoint callback monitors
`val/structured/conditional_nll_per_masked_token`; scheduler metadata uses the
same metric name even though the default constant-warmup scheduler does not
consume a monitored value. Its epoch value, candidate
recall, and retained unary mass are accumulated by active-token numerators and
denominators rather than averaging scalar batch means.

## Joint sampling and ablations

Sampling is an explicit three-way ablation:

- `factorized`: use independent ordinary-backbone predictions;
- `structured_marginal`: run exact forest inference, then independently draw
  tokens from its exact node marginals;
- `structured_joint`: draw one exact joint forest assignment.

The latter two modes use the same trained head and checkpoint; switching modes
does not alter architecture or state-dict shape. They require the non-caching
DDPM sampler. For example:

```bash
sampling.predictor=ddpm \
model.structured_decoder.sampling.mode=structured_joint
```

Each step draws one joint clean assignment and then applies the ordinary
absorbing reveal probability, preserving correlations among tokens committed
in that step. `ddpm_cache`, analytic sampling, and semi-AR strides are rejected
for this opt-in path rather than silently falling back to independent tokens.

Controlled ablations use full Hydra override paths:

- `model.structured_decoder.independent_mode=true`: neutral pair factors with
  identical architecture parameter count. Pair parameters are inactive, so
  this is an architecture-count control, not an active-parameter-matched unary
  adapter;
- `model.structured_decoder.topology_mode=fixed`: natural-order chain unless
  fixed edges are supplied;
- `model.structured_decoder.fixed_edge_path=/path/edges.pt`: corpus-level
  static forest, induced on the currently masked positions;
- `model.structured_decoder.factor_mode=fixed`: context-independent factors;
- `model.structured_decoder.training.topology_weight=0`: no influence
  distillation.

The run logs structured loss, conditional NLL, candidate recall, retained unary
mass, active fraction, selected edges, factorized auxiliary NLL, all three
topology-loss components, valid teacher examples, and teacher influence. Report
candidate recall and topology-teacher coverage with every result. Edge, anchor,
and slot coverage each log an independently accumulated rate and eligible
example denominator; do not substitute their union coverage.

## Current boundary

The bridge supports continuous-time SUBS with a DIT backbone. It does not yet
derive a structured diffusion ELBO, optimize discrete topology end-to-end, use
cached joint DDPM updates, or support semi-AR sampling. The topology teacher is
a supervised training signal and should be labeled as such in experiments.
