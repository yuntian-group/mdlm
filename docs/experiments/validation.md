# Curation validation

Local CPU checks on September 17, 2026 used PyTorch 2.2.2 and offline model-cache
settings. The final combined run covered 13 existing suites: **107 passed,
4 dependency errors and 4 CUDA skips, out of 115 tests**. There were no assertion
failures. This is a partial environment check, not a fully passing GPU test run.

The four dependency-blocked checks require OmegaConf, Hydra or Lightning:
Diffusion endpoint initialization, Hydra embedding configuration, Hydra scheduler
instantiation and full Lightning scheduler checkpoint resume. The four skipped
checks require CUDA for draw/RNG equivalence or multinomial validation. Historical
GPU results remain clearly labeled as historical; no GPU experiment was rerun.

The suites cover sampler optimization and traversal, phase accounting, matrix
construction and first-sample gates, resume selection, endpoint/conditioner
configuration, adapter export, learning-rate scheduling and text scoring. After
portability edits, all nine matrix/gate tests were rerun successfully. All changed
shell scripts passed `bash -n`; changed Python files compiled. The V1 reference
Git blob has the original pinned SHA-256.

Every archived low-step PPL was recomputed from its 20 per-seed NLL/token records;
all 171 aggregate scores and token totals agree within floating-point tolerance.
The archive contains 3,420 distinct cell/seed records, with reused cells counted
once. These checks validate transcription and aggregation, not scientific quality.

The exact CPU objective diagnostic was checked during curation. No new training,
generation or hardware timing was performed. The plotting utility was preserved
but its plot was not rendered in the local environment without Matplotlib.
