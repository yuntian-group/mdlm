# Using the curated experiment series

This is a code and results archive. Model weights, training data, generated text,
private runtime logs and machine access details are not included. Existing
historical snapshots remain immutable. No jobs were submitted during curation.

## Local setup

Install the repository's training/evaluation dependencies in your own Python
environment. Before calling `sbatch`, activate that environment and explicitly
export both roots (a Slurm spool copy cannot infer your checkout directory):

```bash
export CCF_CODE_ROOT=/path/to/this/checkout
export CCF_CACHE_ROOT=/path/to/your/checkpoints-cache-and-runs
```

Supply your own partition/account, node requirements and notification settings
to `sbatch`. The example resource requests are historical starting points.
The cache root contains `checkpoints/mdlm-owt-backbone.pt`, `huggingface/`, and
`runs/`. The scripts verify checkpoint/manifest hashes where recorded.

Historical run directory names use descriptive aliases, for example
`run-baseline-1k`, `run-continue-3k`, `run-continue-6k`,
`run-continue-10k`, `run-separate-r8-fd`, and `run-separate-r16-dd`.
Arrange your own checkpoints under the relative layout shown by each launcher
or update those input paths locally. The first three launchers additionally use
`runs/four-arm-1k`; `CCF_SOURCE_ROOT` and `ADAPTER_DIR` can override relevant inputs.
Old attempt suffixes identify expected artifact layouts, not access credentials.
At 1,000 sampling transitions, the `selected` evaluator also requires the older
R8 DD 2k/6k replay artifacts named in `HISTORICAL_R8`; those comparisons record
cross-hardware replay differences. Low-step sweeps require only current inputs.

## Keep implementation versions distinct

- **O (unoptimized)**: the sampler at the retained recovery baseline.
- **V1**: redundant pre-inference and host/device bookkeeping removed.
- **V2**: unused marginals and duplicate checks removed; this is the ordinary
  generation pilot at the final branch tip.
- **L (level-batched optimized)**: opt-in `scripts/run_generation_fast.py`, used
  by the five-sample and 8/16/32-step evaluator through its verification gate.

A launcher checked out today uses today's code unless an isolated snapshot is
chosen explicitly. Historical result labels describe the implementation that
actually generated those results. Select the corresponding commit or snapshot
and verify it before rerunning older experiments; do not label a current V2 run
as an unoptimized reproduction merely because its launcher originated earlier.

V1 reference source is preserved in this series as Git blob `d8d05932ede6125cea511f654884d1ee64812131` and SHA-256
`61892523cb59ad6da842326df3fcf319989d5b9efb50da8d0b039aa99e0a15bf`. The V2 verifier and selected evaluator load
that blob, so their reference survives `git am` changing commit IDs. The V2
verifier also accepts `--v1-utils` for a source export with the same checked hash.
The older baseline verifier needs the retained baseline history. Apply all
patches on that baseline; a shallow tip-only source copy lacks these references.

These packaging edits use the repository's existing Git/source-hash mechanisms.
No new sampling algorithm was introduced. Algorithm and adapted-code citations
are retained in [the fast-generation notes](../ccf-fast-generation.md).

## Measurement conventions

Training update, requested sampling transition, and actual model-call count are
separate fields. PPL is exp(total evaluator NLL / scored tokens), not the mean
of per-sample PPL. Twenty generated samples per cell can have different scored
token totals because of EOS. Verification reference duplicates are excluded
from the scored sample count. Runtime is generation only unless stated otherwise.
