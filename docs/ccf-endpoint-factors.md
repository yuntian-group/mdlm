# Shared versus separate endpoint factors

The normal CCF training and generation paths accept these Hydra overrides
(`++` also works with older configs that omit this key):

```text
++model.structured_decoder.factor_embedding_mode=shared
++model.structured_decoder.factor_embedding_mode=separate
```

`shared` is the default and retains existing checkpoint keys, initialization,
and adapter identity hashes. `separate` uses independent left/right token
factor tables and independent hidden/time FiLM projections. The hidden-state
normalization and time embedding are still shared. Left means the lower token
position on a canonical edge; right means the higher position, regardless of
the direction of a tree-message pass. Exact inference and sampling are unchanged.

The option is independent of `topology_mode` and `factor_mode`: it works in all
four arms. A focused first experiment can run only `fixed_dynamic` and
`dynamic_dynamic`, keeping their existing topology weights (0.0 and 0.1).

For a parameter-matched comparison against the current shared rank-16 heads,
add these overrides to a **fresh** training command:

```text
++model.structured_decoder.factor_embedding_mode=separate
model.structured_decoder.rank=8
checkpointing.resume_from_ckpt=false
```

Use the same pretrained frozen MDLM backbone and the other matched training
settings, with a new output directory. Do not resume a shared-head checkpoint:
the head tensor shapes/keys and optimizer state differ. Rank is not changed
automatically when selecting separate factors.

For `scripts/run_generation_pilot.py`, use the matching model overrides:

```text
--override ++model.structured_decoder.factor_embedding_mode=separate
--override model.structured_decoder.rank=8
```

Keep the trained topology mode, factor mode, candidate top-K and topology
weight matched too. The existing generic `scripts/export_structured_adapter.py`
reads the endpoint choice from the training checkpoint and binds it into the
adapter manifest; no new export flag is needed. Loading rejects an endpoint
choice that disagrees with that manifest. The submission-specific production
exporter still pins its old shared-head inventory; do not use those fixed
production expectations for this new experiment.

Alternatively, select `model=contextual-forest-endpoints-small` for training
(or `--model-config contextual-forest-endpoints-small` for generation). It
extends the original small config with an explicit shared default, so the
endpoint override no longer needs `++`. The old small config stays byte-for-byte
unchanged for hash-pinned paper replay.

`DirectionalCouplingForestHead` remains available as a backward-compatible
name for the separate option. No training or generation jobs are launched by
the model-option change itself, and existing four-arm job scripts retain their
shared defaults.

Historical results and sample counts: `docs/experiments/03-endpoints-and-conditioners.md`.
