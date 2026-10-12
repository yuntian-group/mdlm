"""Reconstruct the paper's small adaptation/count corpus from pinned OWT text.

The CLI authenticates source-document order/content and the tokenizer, then
packs each original 1,000-document group separately. No model is loaded.
Dataset downloading is opt-in; tokenization is otherwise entirely local.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .backbone import file_sha256

DATASET_ID = 'Skylion007/openwebtext'
DATASET_CONFIG = 'plain_text'
DATASET_REVISION = '79d93d786212f7344586290adb811d4ae6a1762c'
SOURCE_ROWS = 8013769
TOKENIZER_ID = 'openai-community/gpt2'
TOKENIZER_REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
TOKENIZER_BACKEND_SHA256 = '4b45c809d083d6e520def27be3717f3e2503e087e68683d40bfc51c78baae929'
TOKENIZER_FILES = {
    'config.json': '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
    'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
    'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
    'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
}
# Digest definition: SHA256(canonical JSON [[absolute source row, SHA256(text)], ...]).
# These hashes were derived from authenticated original document ledgers, not
# from newly fetched text. Counts are BEFORE appending one EOS per document.
TRAIN_GROUPS = (
    (0, 1125789, 'a049f3de7d6c7dff5552811ec159bf10f4505c2eceba2d9585cdd23e94fbd735'),
    (1000, 1160097, '4d1d3f8a28f37cdd06a397df2bc509fec5d40acd7832ac57cb7545898e571e04'),
    (2000, 1168917, '6a71f7f83570f35b0e90bd765413a7109979a9425d8194c2a310438c836427c6'),
    (3000, 1116529, '44099db81e8d32290e79e4c0d74ae1c1c2d01d74978f308661acf14394c36e23'),
    (4000, 1016959, 'd24ce9978ed377879b05ef9a8dd6a87ed12644429f209105678ee821e032bf60'),
    (5000, 1096575, '182adbd90c0ff42bb9ddf07dcbac45eeecf4544ed4599da7c6d1691c230dc621'),
    (6000, 1092165, 'e2219cdd4d74949bb76b76b39b3e7b43e81824b8b3a593c95f20db96da3bdee9'),
    (7000, 1091389, 'f1a2e3e25f952dc457fceba90a7d419836f8196f4cf14a3fcd5cb111d1f5dcfc'),
    (8000, 1146861, 'e1d8acb856c8f0950c8d406b3c7294d2618fbd9f5052665702b38a8a697bce12'),
)
DEV_GROUPS = (
    (7913769, 1088655, '2a3fb9e8ccc0efd1193fb6604f6b608d051219b0b7eb0236106a82ce2f3c86f3'),
    (7914769, 1091301, '5736bc406d26a0586aac21f46a8f2dda286843ba2bc9c9f1611eb2fb24184776'),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def token_sha(array):
    """Serialization-independent digest of ordered row-major int32-LE IDs."""
    return hashlib.sha256(array.astype('<i4', copy=False).tobytes(order='C')).hexdigest()


def pack_group(tokenized, *, length=1024, eos_id=50256, mask_id=50257, max_rows=None):
    """Append article EOS, pack ONLY this group, drop its incomplete suffix."""
    require(type(length) is int and length >= 2, 'Invalid window length')
    require(type(eos_id) is int and 0 <= eos_id < mask_id, 'Invalid EOS/MASK IDs')
    require(max_rows is None or type(max_rows) is int and max_rows >= 0, 'Invalid row budget')
    require(all(isinstance(row, list) and all(type(t) is int and 0 <= t <= mask_id
                                           for t in row) for row in tokenized), 'Invalid tokenizer IDs')
    stream = [t for row in tokenized for t in [*row, eos_id]]
    available = len(stream) // length
    used = available if max_rows is None else min(available, max_rows)
    array = np.asarray(stream[:used * length], dtype=np.int32).reshape(used, length)
    require(not bool(np.any(array == mask_id)), 'Selected clean rows contain literal MASK/PAD; do not filter')
    remaining = used * length
    contributions = []
    for row in tokenized:
        take = min(remaining, len(row) + 1)
        contributions.append(take)
        remaining -= take
    return array, dict(available_rows=available, rows=used,
        dropped_group_tail_tokens=len(stream) % length,
        dropped_group_tail_token_sha256=canonical_sha(stream[available * length:]),
        unused_complete_rows=available - used, used_stream_tokens_per_document=contributions)


def prepare_split(dataset, tokenizer, groups, *, row_budget, expected_rows,
                  group_size=1000, length=1024, eos_id=50256, mask_id=50257):
    """Authenticate the selected document groups, preserving source order."""
    require(type(row_budget) is int and row_budget > 0 and expected_rows == row_budget,
            'Invalid split row budget')
    arrays, reports, used_hashes = [], [], set()
    remaining = row_budget
    for start, expected_raw_tokens, expected_digest in groups:
        texts = dataset[start:start + group_size]['text']
        require(len(texts) == group_size and all(isinstance(t, str) for t in texts),
                'Incomplete or non-text source group')
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        observed = canonical_sha([[start + i, h] for i, h in enumerate(hashes)])
        require(observed == expected_digest, 'Source content/order differs at group ' + str(start))
        tokenized = tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                              return_token_type_ids=False)['input_ids']
        require(len(tokenized) == group_size, 'Tokenizer dropped documents')
        require(sum(len(row) for row in tokenized) == expected_raw_tokens,
                'Tokenization differs at group ' + str(start))
        array, report = pack_group(tokenized, length=length, eos_id=eos_id,
                                   mask_id=mask_id, max_rows=remaining)
        used_hashes.update(h for h, n in zip(hashes, report['used_stream_tokens_per_document']) if n)
        contributing = [start + i for i, n in enumerate(report['used_stream_tokens_per_document']) if n]
        report.update(source_interval=[start, start + group_size],
                      source_document_hashes_sha256=observed, raw_tokens=expected_raw_tokens,
                      contributing_documents=len(contributing),
                      last_contributing_document=contributing[-1] if contributing else None,
                      ordered_input_ids_int32le_sha256=token_sha(array))
        arrays.append(array)
        reports.append(report)
        remaining -= len(array)
    require(remaining == 0, 'Document groups do not supply the declared row budget')
    result = np.concatenate(arrays, axis=0)
    require(result.shape == (expected_rows, length), 'Wrong reconstructed split shape')
    return result, reports, used_hashes


def load_tokenizer(snapshot):
    """Use the original tokenization behavior; no network or model loading."""
    for name, digest in TOKENIZER_FILES.items():
        require(file_sha256(Path(snapshot) / name) == digest, 'Tokenizer file differs: ' + name)
    for name, version in (('transformers', '4.49.0'), ('tokenizers', '0.21.4')):
        require(importlib.metadata.version(name) == version, 'Pinned preprocessing requires ' + name + '==' + version)
    import tokenizers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
        (tokenizer.bos_token, tokenizer.bos_token_id), (tokenizer.eos_token, tokenizer.eos_token_id))
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    require((tokenizer.vocab_size, tokenizer.bos_token_id, tokenizer.eos_token_id,
             tokenizer.pad_token_id) == (50257, 50256, 50256, 50257), 'Wrong tokenizer vocabulary')
    require(hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
            == TOKENIZER_BACKEND_SHA256, 'Tokenizer backend differs after native setup')
    return tokenizer


def write_outputs(output, train, dev, train_groups, dev_groups, *, source_fingerprint,
                  overlap_count):
    """Write only fresh packed IDs/metadata, never source text or model tensors."""
    output = Path(output)
    require(not output.exists(), 'Output already exists; refusing to overwrite')
    require(train.dtype == dev.dtype == np.dtype('int32') and
            train.shape == (9765, 1024) and dev.shape == (2130, 1024),
            'Expected the exact paper adaptation split shapes and int32 IDs')
    require(all(bool(np.all((a >= 0) & (a < 50257))) for a in (train, dev)),
            'Packed output contains invalid clean IDs')
    output.mkdir(parents=True)
    shards = {}
    for name, rows in (('train', train), ('dev', dev)):
        target = output / (name + '.npy')
        with target.open('xb') as stream:
            np.save(stream, rows, allow_pickle=False)
        shards[name] = dict(path=target.name, sha256=file_sha256(target), rows=len(rows))
    common = dict(vocab_size=50258, mask_id=50257, eos_id=50256, window_length=1024,
                  document_eos_retained=True, window_bos_eos_added=False,
                  tokenizer=dict(id='gpt2', revision=TOKENIZER_REVISION))
    artifacts = {}
    for name in ('train', 'dev', 'count'):
        is_dev = name == 'dev'
        value = dict(common, schema='packed_count_training_v1' if name == 'count' else
                     'packed_chain_adaptation_v1', split='train' if name == 'count' else name,
                     shards=[shards['train' if name == 'count' else name]],
                     provenance=dict(dataset_id=DATASET_ID, dataset_config=DATASET_CONFIG,
                         dataset_revision=DATASET_REVISION, source_split='train',
                         source_groups=dev_groups if is_dev else train_groups,
                         ordered_input_ids_int32le_sha256=token_sha(dev if is_dev else train),
                         preparer_sha256=file_sha256(__file__),
                         native_packed_token_hash_comparison=False))
        target = output / (name + '-manifest.json')
        with target.open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        artifacts[name] = dict(path=target.name, sha256=file_sha256(target))
    receipt = dict(schema='additive_chain_prepared_data_v1', complete=True,
        source=dict(repository=DATASET_ID, config=DATASET_CONFIG, revision=DATASET_REVISION,
                    split='train', rows=SOURCE_ROWS, observed_fingerprint=source_fingerprint),
        tokenizer=dict(id=TOKENIZER_ID, revision=TOKENIZER_REVISION,
                       files_sha256=TOKENIZER_FILES, backend_sha256=TOKENIZER_BACKEND_SHA256),
        manifests=artifacts, train_groups=train_groups, dev_groups=dev_groups,
        ordered_input_ids_int32le_sha256=dict(train=token_sha(train), dev=token_sha(dev)),
        training_budget_tokens=10000000, selected_training_tokens=int(train.size),
        source_intervals_disjoint=True, overlapping_contributing_text_hashes=overlap_count,
        content_overlap_action='reported; no documents filtered',
        full_heldout_cache_reconstructed=False, native_packed_token_hash_comparison=False,
        validation_scope='first2000 of final100000 source articles; first2130 packed rows',
        training_selection='first9765 windows, packing each1000-document group separately',
        serialization_note='Compare ordered token digests; NPY/Arrow file serialization may differ.',
        preparer_sha256=file_sha256(__file__),
        versions={name: importlib.metadata.version(name) for name in
                  ('numpy', 'datasets', 'transformers', 'tokenizers')})
    with (output / 'prepared-data.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset-snapshot', type=Path,
                        help='Local datasets.save_to_disk() full train split (memory mapped)')
    source.add_argument('--download-source', action='store_true',
                        help='Explicitly permit pinned OWT download, never model weights')
    p.add_argument('--cache-dir', type=Path, help='Required explicit destination for --download-source')
    p.add_argument('--tokenizer-snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    require(not args.output.exists(), 'Output already exists; refusing to overwrite')
    require(not args.download_source or args.cache_dir is not None,
            '--download-source requires an explicit --cache-dir')
    tokenizer = load_tokenizer(args.tokenizer_snapshot)
    from datasets import load_dataset, load_from_disk
    if args.download_source:
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION,
                               split='train', cache_dir=str(args.cache_dir))
    else:
        dataset = load_from_disk(str(args.dataset_snapshot), keep_in_memory=False)
    require(len(dataset) == SOURCE_ROWS and 'text' in dataset.column_names,
            'Expected the full pinned OWT train split in source order')
    train, training, train_hashes = prepare_split(dataset, tokenizer, TRAIN_GROUPS,
                                                 row_budget=9765, expected_rows=9765)
    dev, development, dev_hashes = prepare_split(dataset, tokenizer, DEV_GROUPS,
                                                row_budget=2130, expected_rows=2130)
    receipt = write_outputs(args.output, train, dev, training, development,
                            source_fingerprint=getattr(dataset, '_fingerprint', None),
                            overlap_count=len(train_hashes & dev_hashes))
    print(json.dumps({'output': str(args.output), 'train_rows': len(train),
                      'dev_rows': len(dev), 'manifests': receipt['manifests']}))


if __name__ == '__main__':
    main()
