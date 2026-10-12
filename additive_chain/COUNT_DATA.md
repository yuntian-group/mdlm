# Building count statistics

`python -m additive_chain.prepare_counts` counts adjacent token pairs in
already packed **training** windows. It writes the tensor-only format accepted
by the [generation CLI](CLI.md). It does not download data or model weights.

Input shards are `.npy` integer arrays with shape `[windows, window_length]`.
Document EOS is retained and counted like any other token. A window may span
an article boundary. The command never adds window BOS/EOS, never inserts an
edge between two windows or shards, and rejects MASK/PAD in clean text.

Supply a JSON manifest such as:

```json
{
  "schema": "packed_count_training_v1",
  "split": "train",
  "vocab_size": 50258,
  "mask_id": 50257,
  "eos_id": 50256,
  "window_length": 1024,
  "document_eos_retained": true,
  "window_bos_eos_added": false,
  "tokenizer": {
    "id": "openai-community/gpt2",
    "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e"
  },
  "shards": [
    {"path": "train-000.npy", "sha256": "REPLACE_WITH_ACTUAL_SHA256", "rows": 1000}
  ]
}
```

Paths are relative to the manifest. Replace both the example row count and
hash with the actual shard metadata. Then run on the machine holding the data:

```bash
python -m additive_chain.prepare_counts \
  --manifest train-manifest.json \
  --manifest-sha256 ACTUAL_MANIFEST_SHA256 \
  --output new-count-statistics
```

The output contains `counts.pt` and a receipt with the source manifest,
counts hash, token/window/edge totals, and number of distinct pairs. Raw counts
are independent of the inference strength. The stored default is conditional
mode, strength one, smoothing 0.1; generation selects its strength separately.
Counting memory scales with observed pair types, not a dense vocabulary square.
Input chunks are memory-mapped; the complete sparse count dictionary still
needs CPU memory.

This helper verifies bytes, dimensions and token ranges. A `split: train`
declaration is **not** proof of document-level disjointness. Audit source
document IDs and content hashes against development/test data before fitting.
The receipt intentionally does not claim that audit happened.

## Reconstructing the adaptation/count inputs

`python -m additive_chain.prepare_data` implements the recorded document
selection and packing recipe for the paper's **9,765 training and 2,130
development windows**. It reads a local Hugging Face `save_to_disk()` snapshot
of the complete OpenWebText `train` split, without loading model weights:

```bash
python -m additive_chain.prepare_data \
  --dataset-snapshot /data/openwebtext-train \
  --tokenizer-snapshot /data/gpt2-tokenizer \
  --output /data/new-adaptation-data
```

The dataset is `Skylion007/openwebtext`, configuration `plain_text`, revision
`79d93d786212f7344586290adb811d4ae6a1762c`, with 8,013,769 ordered source rows.
The tokenizer is `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. The preparer verifies its five recorded
file hashes, the native PAD/postprocessor setup and backend digest. It requires
the recorded tokenizer versions, `transformers==4.49.0` and
`tokenizers==0.21.4`. The native dataset preparation used `datasets==2.18.0` and
`numpy==1.26.4`; emitted receipts record the actual installed versions.

If the source dataset is not local, replace `--dataset-snapshot ...` with
`--download-source --cache-dir /data/dataset-cache` to explicitly permit a
pinned dataset download. The tokenizer must still be present locally. This is
a large dataset download; choose a machine/cache with sufficient storage.
No download occurs by default, and no model weights are downloaded.

The exact recipe is:

- Tokenize source text without tokenizer-added special tokens, append one EOS
  per article, concatenate **within each group of 1,000 articles**, pack 1,024
  tokens and drop that group's incomplete tail. No artificial window BOS/EOS
  is added. Literal article EOS is retained. Windows can cross article boundaries.
- Training takes the first 9,765 complete windows (9,999,360 tokens) in source
  order. Groups `[0,1000)`, ..., `[8000,9000)` contribute respectively
  `1100,1133,1142,1091,994,1071,1067,1066,1101` windows. The last group is
  truncated to the fixed token budget. There are 8,995 contributing articles;
  the last is source row 8,996, with some earlier group-tail articles unused.
- Development takes source rows `[7913769,7915769)`, the first 2,000 articles
  of the held-out final 100,000. Its two groups contribute 1,064 and 1,066
  windows, dropping 119 and 717 tokens respectively. Training's development
  selection evaluates the first 32 of these rows, not all 2,130.

Every selected source group is checked against a recorded digest of its ordered
document IDs/text hashes and raw token total. The output includes `train.npy`,
`dev.npy`, companion `train-manifest.json`/`dev-manifest.json` in
`packed_chain_adaptation_v1` format, and `count-manifest.json` accepted by
`prepare_counts`. `prepared-data.json` binds the manifests, array-file hashes,
ordered int32 token digests, group boundaries, tokenizer and observed versions.
It contains no source article text. It reports exact-text overlap between the
contributing training/development documents without filtering either split.

Use the printed `count` manifest hash with the earlier counting command. Raw
count tensors should match for identical token rows, but serialized Torch/NumPy
files can have different bytes; file hashes are not a cross-version numeric
equivalence test.

This reconstructs the small adaptation inputs, **not** the entire 110,254-window
held-out likelihood cache. That larger cache used eight contiguous map-worker
shards and shard-local groups; changing the worker count moves later group
boundaries. Its first two groups coincide with the development recipe above.
Nor is this the original checkpoint's full pretraining corpus. CPU tests cover
source/order checks, packing, budgets and manifests using synthetic text; an
end-to-end corpus reconstruction has not been rerun for this release. Receipts
therefore do not claim a fresh comparison with native packed-token digests.
