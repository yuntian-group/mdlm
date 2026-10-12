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

For the paper's existing adaptation corpus, packing was performed separately
within groups of 1,000 source documents, retaining article EOS and dropping
each group's incomplete trailing window. The selected rows total 9,765 windows
(9,999,360 tokens). Repacking the same documents as one global stream can move
window boundaries and change counts. To reproduce those counts, use the exact
audited packed rows, in their original groups, not a newly concatenated stream.
This command does not itself reproduce the source-document selection or
tokenization pipeline; it starts from those packed arrays.
