# Separate endpoint and conditioner screens

Introduced September 15, 2026. These changes expose separate left/right factor
tables and optional FiLM MLP conditioning through the standard training,
adapter-export and generation paths. Shared defaults retain their checkpoint
keys and seeded initialization. Fresh separate-R8 training matches shared-R16
head capacity: 984,417 parameters in this production configuration. FiLM width
128 adds 84,096 parameters; K=256 changes support, not learned parameter count.

| Variant at training update 7,000 | Arm | GPT-2 PPL | Generated samples | Sampler |
|---|---|---:|---:|---|
| Shared R16 reference | DD | 42.073 | 1 | O: unoptimized |
| Separate R8 | FD | 83.727 | 1 | O: unoptimized |
| Separate R8 | DD | 88.858 | 1 | O: unoptimized |
| Shared R16 + FiLM MLP128 | DD | 82.263 | 1 | O: unoptimized |
| Shared R16 + top-K256 | DD | 1165.201 | 1 | O: unoptimized |

Every PPL is from one generated sequence, length 1024, seed 91001, 1,000
transitions plus cleanup, batch 1, GPT-2-large float32 with the established EOS
scoring policy. There are four variant samples above and one reused reference
sample. None of these one-sample outcomes establishes a reliable ranking.

FiLM's earlier 2k/4k/6k screens recorded PPL 55.670/82.537/58.019, respectively,
with **one generated sample per checkpoint**, also sampler O. The later
optimized, five- and twenty-sample studies must be read separately.

Training uses seed 1, frozen MDLM, batch 4, 50 warmup updates then LR 3e-4,
and checkpoints every 500 updates. Restart handling preserves completed
outputs, validates full checkpoint state, and starts fresh if no checkpoint
survived an interruption. Private machine/account details are omitted.

The script/code snapshots are historical; this curation changes deployment
paths and documentation only and does not rerun the experiments. Checkpoints
must agree with endpoint mode, conditioner width and adapter-manifest identity.

Contextual affine modulation has prior work in
[FiLM](https://arxiv.org/abs/1709.07871); low-rank contextual structured decoding
has prior work in [Sun et al.](https://arxiv.org/abs/1910.11555). Separate endpoint
factors and their restricted expressivity motivation were already present in
the inherited recovery research. This commit integrates that option into the
production path; it does not claim to invent role-specific embeddings.

Curation validation: 65 CPU checks passed across factor, adapter, export and
restart tests. Two integration checks could not execute because this local
environment lacks Hydra/OmegaConf; these are dependency limitations, not new
GPU validation. Shell syntax and privacy checks passed.
