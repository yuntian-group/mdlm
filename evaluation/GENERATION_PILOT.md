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
