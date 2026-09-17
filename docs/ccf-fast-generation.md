# Verified fast CCF generation: level batching

Historical September 16, 2026. Use `scripts/run_generation_fast.py` as an opt-in
wrapper around the original generation pilot. It enables `level_draws` only
for the current process and records implementation source hashes. The ordinary
pilot retains the conservative V2 implementation at this branch stage.

## Full-generation timing record

Separate R16 DD at training update 7,000, K128, length 1024, 1,000 reverse
transitions plus cleanup. **Two generated samples per implementation per
comparison**, seeds 91001/91002. Timing excludes loading and GPT-2 scoring.
Every row compares implementations on the same GPU; rows must not be pooled
as if they were one hardware-controlled experiment.

| Candidate | Reference | Reference s/sample | Candidate s/sample | GPU type | Samples per implementation |
|---|---|---:|---:|---|---:|
| unchecked | V2 | 363.875 | 262.403 | H200 NVL | 2 |
| unchecked_batched | V2 | 337.811 | 195.306 | H200 NVL | 2 |
| unchecked_batched_roots_kruskal | V2 | 340.947 | 137.654 | H200 NVL | 2 |
| root_draws | unchecked_batched_roots_kruskal | 137.255 | 125.003 | H200 NVL | 2 |
| level_draws | unchecked_batched_roots_kruskal | 136.233 | 69.827 | L40S | 2 |
| tensor_traversal | unchecked_batched_roots_kruskal | 135.911 | 69.234 | L40S | 2 |
| buffered_noise | unchecked_batched_roots_kruskal | 149.535 | 70.280 | H200 NVL | 2 |
| aligned_noise | unchecked_batched_roots_kruskal | 135.338 | 67.911 | L40S | 2 |
+
+The selected simple level path has final-token, actual-call-count, and CPU/CUDA
+RNG equality for its two paired full samples. These checks are broader than
+the earlier three individual-update comparisons but do not prove equivalence
+for every input, architecture, runtime or GPU. The slightly faster candidate
+measurements do not establish a robust advantage over the chosen level path.
+
+At the same historical comparison, factorized MDLM took about 15.58 seconds
+per sample versus about 69.83 for level-batched CCF. The optimization narrows
+the runtime gap; it does not establish parity with MDLM. These are generation
+timings, not perplexities. Later PPL studies explicitly label this path **L**.
+
+The earlier level benchmark failed due to instrumentation source inspection;
+the corrected benchmark passed. Timing with phase instrumentation is kept
+separate from uninstrumented replay. CUDA intervals can overlap, and host
+timings include GPU waits; do not sum nested intervals as kernel-busy time.
+
+## Algorithm and implementation attribution
+
+- The categorical exponential-race helper adapts [PyTorch 2.2.2's single-sample
+  multinomial implementation](https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/native/Distributions.cpp#L599-L642).
+  Rewriting it in Python does not make the algorithm new. Its license is
+  retained at `third-party/pytorch-v2.2.2-LICENSE.txt`.
+- Tree message passing uses the standard sum-product principle:
+  [Kschischang, Frey and Loeliger (2001)](https://doi.org/10.1109/18.910572).
+- Forest selection uses a constrained greedy variant of
+  [Kruskal's algorithm](https://doi.org/10.1090/S0002-9939-1956-0078686-7).
+  Component caps do not inherit ordinary MST optimality guarantees.
+- Level batching, gathers and buffers were derived from local code and
+  profiling; this provenance statement does not claim these techniques are new.
+- [DA-DLM](https://arxiv.org/abs/2609.15070) was consulted for a timing comparison;
+  its implementation was not incorporated into this optimization.
+
+Retain the original arguments, hyperparameters, seeds, precision, reveal
+schedule and EOS scoring. New GPU runs require an isolated snapshot, fresh
+output location, and a successful equivalence check in that runtime. Machine
+logins, host identifiers and private storage paths are deliberately omitted.
+Historical measurements above were not rerun during commit curation.
