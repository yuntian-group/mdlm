# Paired generation and infilling pilot

`scripts/run_generation_pilot.py` runs the same ordered prompt/seed commitment
for `factorized`, `structured_marginal`, and `structured_joint` sampling at one
or more NFE budgets. The default global pilot has 256 samples, length 256, and
NFE budgets 32 and 64.

Every successful shard writes:

- `samples.jsonl`: every decoded sample, complete token ids, initial state,
  active mask, reference ids when available, seeds, batch timing, and metrics;
- `summary.json`: distinct-n, repetition, reference recovery, n-gram JS,
  throughput, measured NFE, and peak-memory summaries;
- `manifest.json`: artifact hashes, Git state, prompt hash, global and shard
  pairing commitments, command, host, and output hashes; and
- `resolved_config.yaml`.

## Spot-safe execution

The runner never appends or resumes partial results. It keeps records in memory
and atomically commits them only after the shard's entire mode/NFE matrix has
finished. A non-empty output directory is rejected. If a Spot VM interrupts a
shard, rerun that shard in a fresh directory.

Runs also require a clean Git tree by default. `--allow-dirty` is an explicit
escape hatch that records the tracked diff hash plus every untracked file's
path, size, and SHA256 and commits their combination in the manifest.

Use small independent shards to bound lost work. For example, this runs shard
3 of 16 (16 paired samples per shard for the default 256-sample pilot):

```bash
python scripts/run_generation_pilot.py \
  --backbone-checkpoint /path/to/released-backbone.pt \
  --backbone-sha256 "$BACKBONE_SHA256" \
  --adapter /path/to/structured-adapter.safetensors \
  --adapter-sha256 "$ADAPTER_SHA256" \
  --adapter-manifest /path/to/adapter-manifest.json \
  --adapter-manifest-sha256 "$ADAPTER_MANIFEST_SHA256" \
  --num-shards 16 \
  --shard-index 3 \
  --output-dir /mnt/experiment/generation/shard-03
```

All shards for one experiment must use identical scientific arguments except
for `--shard-index` and `--output-dir`. Before aggregation, verify that their
manifests have the same `global_pairing_digest`, artifact hashes, prompt hash,
mode/NFE matrix, sequence length, base seed, batch size, and shard count; that
each shard index appears exactly once; and that sample indices are disjoint and
cover `[0, num_samples)`.

Run the fail-closed verifier with every shard directory:

```bash
python scripts/aggregate_generation_shards.py \
  --shard /mnt/experiment/generation/shard-00 \
  --shard /mnt/experiment/generation/shard-01 \
  --output /mnt/experiment/generation/verified-union.json
```

The verifier binds adapter-manifest identity by its verified SHA256 and
structured-decoder semantic digest, not by a machine-specific absolute path.
It also requires identical OS/platform, Python, PyTorch, CUDA runtime,
parameter dtypes, precision policy, and NumPy, safetensors, tokenizers, and
Transformers versions. GPU model identity must match before descriptive timing
is pooled. Deterministic token metrics and shard summaries are recomputed and
compared exactly; the only numeric tolerance is a near-machine-precision check
that a stored reference-LM perplexity equals `exp(mean_nll)`.

## Prompt JSONL schemas

Omit `--prompt-jsonl` for unconditional generation. Otherwise, each non-empty
JSONL line uses exactly one of these schemas:

```json
{"id":"prefix-1","prompt":"A short observed prefix"}
{"id":"infill-1","text":"The complete reference text.","mask_token_indices":[2,3,4]}
{"id":"tokens-1","input_ids":[10,20,30,40],"active_mask":[false,true,true,false],"reference_token_ids":[10,21,31,40]}
```

The explicit token schema must have exactly `--sequence-length` entries. The
text-infilling schema tokenizes the complete reference, masks only the listed
valid positions, and treats EOS padding as observed. Prefix generation masks
all positions after the encoded prefix. Prompts cycle deterministically when
the global sample count exceeds the number of JSONL records.

For the pinned document-local validation datasets, build a prompt file without
loading model weights:

```bash
python scripts/build_infilling_prompts.py \
  --data-config eval_wikitext103_pinned \
  --sequence-length 256 \
  --span-length 32 \
  --num-prompts 256 \
  --selection-seed 31001 \
  --output /mnt/experiment/prompts/wikitext-span32.jsonl
```

The builder takes the first 256 windows in pinned validation order and chooses
each contiguous span by hashing only immutable document metadata, the span
policy, and the selection seed. It never masks the first BOS or final EOS
position. Its adjacent manifest commits the data config, runtime dataset
provenance, tokenizer and dataset revisions, policy, and prompt-file hash.

Add both `--reference-lm gpt2-large` and
`--reference-lm-revision <immutable-hugging-face-commit>` to activate the
optional causal-reference-LM hook. The runner rejects an unpinned reference
model. Reference-LM NLL and perplexity are descriptive generation metrics;
they are not diffusion likelihoods or ELBO estimates.

## Frozen WikiText two-worker queue

`scripts/run_wikitext_generation_queue.py` is the fail-closed controller for
the active K=128 WikiText paper run. It is intentionally not a general launch
template. It binds the complete scientific argv, artifact hashes, immutable
runner commit `09f89c00bbf8c65f679cd40b92609754608817b8`, runner and protocol
file hashes, live PID mapping, and exact 32-shard grid.

The controller first adopts PID 5226 as dynamic shard 00 and PID 5958 as
dynamic shard 01, or validates those completed directories if the processes
have already exited. It then runs dynamic shards 02--15 in pairs, waits at a
phase barrier, and runs static shards 00--15 in pairs. A zero process exit is
not sufficient: the complete shard is cryptographically reloaded and checked
against the frozen paper protocol before the next pair starts. A failed peer
does not kill the other peer; the controller waits for and validates that peer
before halting.

Deploy the controller outside the immutable runner checkout. Copying it into
the checkout would make Git dirty, which is rejected before any new shard is
started. The durable invocation is:

```bash
ROOT=/mnt/contextual-forest/experiments/contextual-forest-generation-paper-v1
PY=/mnt/contextual-forest/venv/bin/python
test ! -e "$ROOT/run-wikitext-queue.py"
test ! -e "$ROOT/logs/wikitext-queue.log"
cp scripts/run_wikitext_generation_queue.py "$ROOT/run-wikitext-queue.py"
set -o noclobber
nohup "$PY" "$ROOT/run-wikitext-queue.py" \
  >"$ROOT/logs/wikitext-queue.log" 2>&1 </dev/null &
```

Before using that redirection, require that the controller log path does not
already exist. Every shard also has its own exclusively created log. Existing
complete directories are revalidated and skipped; any pre-existing directory
without a valid completion manifest, or any orphan log, stops the queue and is
preserved. Retries therefore require a separately reviewed fresh suffixed
directory rather than deletion or reuse.

## Frozen cross-domain continuation

`scripts/run_cross_domain_generation_queue.py` implements the same two-worker,
phase-barrier protocol for the already-pinned arXiv and PubMed prompt bundles.
It is limited to those two named bundles and authenticates the prompt JSONL,
prompt manifest, builder provenance, data configuration, immutable runner,
backbone, adapter pair, scorer, and exact 32-shard task grid before launching.
Each completed shard is then cryptographically and semantically revalidated.

Cross-domain work is a conditional continuation: do not run it concurrently
with the WikiText queue, and do not launch it before the complete WikiText
union and prompt-cluster analysis have been reviewed. From a separate clean
checkout containing the controller, while leaving the immutable generation
runner at its frozen revision, use exactly one of:

```bash
python scripts/run_cross_domain_generation_queue.py --dataset arxiv
python scripts/run_cross_domain_generation_queue.py --dataset pubmed
```

The arXiv and PubMed queues each contain 16 dynamic-adapter shards followed by
16 static-adapter shards. They use 1,024 paired draws, base seeds 92001 and
93001 respectively, and otherwise retain the WikiText sequence length, span,
batch size, NFE grid, candidate support, sampling modes, and reference scorer.
Existing incomplete output directories or logs are never reused.
