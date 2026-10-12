"""Build sparse bigram statistics from explicit, already packed training rows.

This command neither tokenizes documents nor downloads a dataset. Article EOS
tokens are ordinary counted IDs; no edge crosses a supplied window boundary.
The manifest records preprocessing declarations, not proof of train/test
disjointness. Use independently audited training shards for an experiment.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from .backbone import file_sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_manifest(manifest):
    require(isinstance(manifest, dict) and manifest.get('schema') == 'packed_count_training_v1',
            'Expected packed_count_training_v1 manifest')
    require(manifest.get('split') == 'train', 'Only declared training shards may fit counts')
    for name in ('vocab_size', 'window_length'):
        require(type(manifest.get(name)) is int and manifest[name] >= 2, 'Invalid ' + name)
    vocab = manifest['vocab_size']
    require(type(manifest.get('mask_id')) is int and manifest['mask_id'] == vocab - 1
            and type(manifest.get('eos_id')) is int
            and 0 <= manifest['eos_id'] < vocab - 1, 'Invalid MASK/EOS convention')
    require(manifest.get('document_eos_retained') is True
            and manifest.get('window_bos_eos_added') is False,
            'Expected retained document EOS and no added window BOS/EOS')
    tokenizer = manifest.get('tokenizer')
    require(isinstance(tokenizer, dict) and isinstance(tokenizer.get('id'), str)
            and bool(tokenizer['id']) and isinstance(tokenizer.get('revision'), str)
            and bool(tokenizer['revision']), 'Record tokenizer ID and revision')
    shards = manifest.get('shards')
    require(isinstance(shards, list) and bool(shards), 'No training shards')
    for shard in shards:
        require(isinstance(shard, dict) and set(shard) == {'path', 'sha256', 'rows'}
                and isinstance(shard['path'], str) and bool(shard['path'])
                and isinstance(shard['sha256'], str)
                and re.fullmatch('[0-9a-f]{64}', shard['sha256']) is not None
                and type(shard['rows']) is int and shard['rows'] > 0, 'Invalid shard declaration')
    require(len({s['path'] for s in shards}) == len(shards), 'Duplicate training shard')


def build_statistics(manifest, *, base_dir, chunk_rows=256):
    """Count bounded chunks of memory-mapped integer windows, never dense V²."""
    validate_manifest(manifest)
    require(type(chunk_rows) is int and chunk_rows > 0, 'chunk_rows must be positive')
    vocab, length = manifest['vocab_size'], manifest['window_length']
    require(vocab <= math.isqrt(np.iinfo(np.int64).max), 'Pair keys would overflow int64')
    counts = Counter()
    rows = 0
    seen_paths = set()
    for shard in manifest['shards']:
        path = (Path(base_dir) / shard['path']).resolve()
        require(path not in seen_paths, 'Duplicate resolved training shard')
        seen_paths.add(path)
        require(file_sha256(path) == shard['sha256'], 'Training shard SHA256 mismatch')
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        require(isinstance(data, np.ndarray) and data.dtype.kind in 'iu'
                and data.shape == (shard['rows'], length), 'Expected integer [rows, window_length] .npy')
        for start in range(0, len(data), chunk_rows):
            block = np.asarray(data[start:start + chunk_rows])
            require(bool(np.all((block >= 0) & (block < manifest['mask_id']))),
                    'Training rows contain MASK/PAD or out-of-vocabulary IDs')
            block = block.astype(np.int64, copy=False)
            keys, frequencies = np.unique(
                (block[:, :-1] * vocab + block[:, 1:]).reshape(-1), return_counts=True)
            counts.update(dict(zip(keys.tolist(), frequencies.tolist())))
        require(file_sha256(path) == shard['sha256'], 'Training shard changed during counting')
        rows += len(data)
    keys = torch.tensor(sorted(counts), dtype=torch.int64)
    values = torch.tensor([counts[k] for k in keys.tolist()], dtype=torch.float64)
    require(rows * (length - 1) <= 2**53, 'FP64 cannot exactly represent this count total')
    left, right = torch.zeros(vocab, dtype=torch.float64), torch.zeros(vocab, dtype=torch.float64)
    left.index_add_(0, keys // vocab, values)
    right.index_add_(0, keys % vocab, values)
    require(int(values.sum()) == rows * (length - 1), 'Wrong within-window edge total')
    return dict(left_counts=left, right_counts=right, pair_keys=keys, pair_counts=values), rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chunk-rows', type=int, default=256)
    args = p.parse_args(argv)
    raw = args.manifest.read_bytes()
    import hashlib
    require(hashlib.sha256(raw).hexdigest() == args.manifest_sha256, 'Manifest SHA256 mismatch')
    manifest = json.loads(raw)
    require(not args.output.exists(), 'Output directory already exists; refusing to overwrite')
    stats, rows = build_statistics(manifest, base_dir=args.manifest.resolve().parent,
                                  chunk_rows=args.chunk_rows)
    # mkdir without exist_ok also rejects a concurrent writer before any output.
    args.output.mkdir(parents=True)
    artifact = args.output / 'counts.pt'
    with artifact.open('xb') as stream:
        torch.save(dict(format='chain_crf_counts_v1', vocab_size=manifest['vocab_size'],
                        mode='conditional', strength=1., smoothing=.1, **stats), stream)
    receipt = dict(schema='packed_count_statistics_receipt_v1',
        source_manifest=manifest, source_manifest_sha256=args.manifest_sha256,
        count_file='counts.pt', count_sha256=file_sha256(artifact), windows=rows,
        tokens=rows * manifest['window_length'], edges=rows * (manifest['window_length'] - 1),
        distinct_pairs=stats['pair_keys'].numel(), dense_vocabulary_matrix_allocated=False,
        cross_window_edges=False, document_disjointness_independently_verified=False,
        note='Input declarations and bytes authenticated; corpus provenance must be audited separately.')
    with (args.output / 'manifest.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'count_file': str(artifact), 'sha256': receipt['count_sha256'],
                      'windows': rows, 'edges': receipt['edges']}))


if __name__ == '__main__':
    main()
