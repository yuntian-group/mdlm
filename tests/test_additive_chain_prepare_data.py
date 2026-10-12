"""CPU-only synthetic preprocessing tests; never download a dataset/tokenizer."""
import hashlib
import json

import numpy as np
import pytest

from additive_chain import prepare_data as data
from additive_chain.prepare_counts import validate_manifest


class Dataset:
    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, item):
        return {'text': self.texts[item]}


class Tokenizer:
    def __init__(self, rows):
        self.rows = rows

    def __call__(self, texts, **kwargs):
        assert kwargs == dict(add_special_tokens=False, return_attention_mask=False,
                              return_token_type_ids=False)
        return {'input_ids': [self.rows[text] for text in texts]}


def group(texts, rows, start):
    return (start, sum(len(rows[t]) for t in texts), data.canonical_sha([
        [start+i, hashlib.sha256(t.encode()).hexdigest()] for i, t in enumerate(texts)]))


def test_group_tail_is_not_carried_to_next_group_and_literal_eos_retained():
    first, a = data.pack_group([[1, 8], [2, 3]], length=4, eos_id=8, mask_id=9)
    second, b = data.pack_group([[4], [5]], length=4, eos_id=8, mask_id=9)
    assert first.tolist() == [[1, 8, 8, 2]]
    assert second.tolist() == [[4, 8, 5, 8]]
    assert a['dropped_group_tail_tokens'] == 2 and b['dropped_group_tail_tokens'] == 0
    assert a['used_stream_tokens_per_document'] == [3, 1]
    assert a['dropped_group_tail_token_sha256'] == data.canonical_sha([3, 8])


def test_budget_truncates_whole_windows_not_documents():
    rows, report = data.pack_group([[1, 2], [3, 4, 5], [6]],
                                   length=3, eos_id=8, mask_id=9, max_rows=2)
    assert rows.tolist() == [[1, 2, 8], [3, 4, 5]]
    assert report['used_stream_tokens_per_document'] == [3, 3, 0]
    assert report['unused_complete_rows'] == 1


@pytest.mark.parametrize('tokenized', [[[True]], [[-1]], [[10]], [[9]]])
def test_bad_or_selected_mask_token_is_not_silently_filtered(tokenized):
    with pytest.raises(ValueError):
        data.pack_group(tokenized, length=2, eos_id=8, mask_id=9)


def test_source_order_content_and_token_count_authentication():
    texts = ['alpha', 'beta', 'gamma', 'delta']
    token_rows = dict(alpha=[1, 2], beta=[3, 4], gamma=[5], delta=[6])
    groups = [group(texts[:2], token_rows, 0), group(texts[2:], token_rows, 2)]
    packed, reports, hashes = data.prepare_split(Dataset(texts), Tokenizer(token_rows), groups,
        row_budget=2, expected_rows=2, group_size=2, length=4, eos_id=8, mask_id=9)
    assert packed.tolist() == [[1, 2, 8, 3], [5, 8, 6, 8]]
    assert reports[0]['source_interval'] == [0, 2] and len(hashes) == 4
    assert reports[0]['last_contributing_document'] == 1
    with pytest.raises(ValueError, match='Source content/order'):
        data.prepare_split(Dataset(texts[::-1]), Tokenizer(token_rows), groups,
            row_budget=2, expected_rows=2, group_size=2, length=4, eos_id=8, mask_id=9)
    with pytest.raises(ValueError, match='Tokenization differs'):
        data.prepare_split(Dataset(texts), Tokenizer(dict(token_rows, alpha=[1])), groups,
            row_budget=2, expected_rows=2, group_size=2, length=4, eos_id=8, mask_id=9)
    with pytest.raises(ValueError, match='row budget'):
        data.prepare_split(Dataset(texts), Tokenizer(token_rows), groups,
            row_budget=3, expected_rows=3, group_size=2, length=4, eos_id=8, mask_id=9)


def test_paper_population_counts_and_dev_position_within_eight_worker_cache():
    remaining = 9765
    train_rows = []
    for _, raw, _ in data.TRAIN_GROUPS:
        rows = min((raw + 1000) // 1024, remaining)
        train_rows.append(rows)
        remaining -= rows
    assert train_rows == [1100, 1133, 1142, 1091, 994, 1071, 1067, 1066, 1101]
    assert sum(train_rows) == 9765 and remaining == 0
    assert [(raw+1000)//1024 for _, raw, _ in data.DEV_GROUPS] == [1064, 1066]
    assert [(raw+1000)%1024 for _, raw, _ in data.DEV_GROUPS] == [119, 717]
    assert data.DEV_GROUPS[0][0] == data.SOURCE_ROWS - 100000
    # The actual heldout mapper first shards100K into8 contiguous12,500-row
    # pieces. Its first two1K batches therefore have exactly these boundaries.
    first_shard_stop = data.SOURCE_ROWS - 100000 + 100000//8
    assert data.DEV_GROUPS[-1][0] + 1000 < first_shard_stop
    assert data.TRAIN_GROUPS[-1][0] + 1000 < data.DEV_GROUPS[0][0]


def test_token_digest_is_ordered_and_serialization_independent():
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert data.token_sha(values) == data.token_sha(values.astype('>i4'))
    assert data.token_sha(values) != data.token_sha(values[::-1])


def test_output_companions_are_authenticated_and_count_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr(data.importlib.metadata, 'version', lambda name: 'test-only')
    # Synthetic IDs test serialization/schema, not actual OWT reproduction.
    train = np.zeros((9765, 1024), dtype=np.int32)
    dev = np.ones((2130, 1024), dtype=np.int32)
    out = tmp_path / 'prepared'
    receipt = data.write_outputs(out, train, dev, [], [], source_fingerprint='synthetic', overlap_count=0)
    assert receipt['native_packed_token_hash_comparison'] is False
    assert receipt['full_heldout_cache_reconstructed'] is False
    for name, reference in receipt['manifests'].items():
        path = out / reference['path']
        assert data.file_sha256(path) == reference['sha256']
        manifest = json.loads(path.read_text())
        if name == 'count':
            validate_manifest(manifest)
        else:
            assert manifest['schema'] == 'packed_chain_adaptation_v1'
            assert manifest['split'] == name
            # Exercise the actual optimizer input adapter, not a parallel
            # schema-only mock. This does not run any model or optimizer.
            from additive_chain.train_cli import PackedRows
            parsed = PackedRows(path, reference['sha256'], split=name)
            assert len(parsed) == (2130 if name == 'dev' else 9765)
            assert parsed[0].tolist() == ([1] if name == 'dev' else [0]) * 1024
            assert parsed.identity['manifest']['provenance']['dataset_revision'] == data.DATASET_REVISION
        shard = manifest['shards'][0]
        assert data.file_sha256(out / shard['path']) == shard['sha256']
        assert shard['rows'] == (2130 if name == 'dev' else 9765)
    assert json.loads((out / 'prepared-data.json').read_text())['complete'] is True
    with pytest.raises(ValueError, match='already exists'):
        data.write_outputs(out, train, dev, [], [], source_fingerprint='synthetic', overlap_count=0)


def test_write_rejects_wrong_population_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='split shapes'):
        data.write_outputs(tmp_path/'bad', np.zeros((1, 1024), dtype=np.int32),
                           np.zeros((1, 1024), dtype=np.int32), [], [],
                           source_fingerprint=None, overlap_count=0)
    assert not (tmp_path/'bad').exists()


def test_download_requires_explicit_cache_before_loading_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'load_tokenizer', lambda *_: pytest.fail('Must reject before tokenizer load'))
    with pytest.raises(ValueError, match='explicit --cache-dir'):
        data.main(['--download-source', '--tokenizer-snapshot', str(tmp_path/'tokenizer'),
                   '--output', str(tmp_path/'out')])


def test_tokenizer_file_authentication_precedes_dependency_or_model_load(tmp_path):
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(ValueError, match='Tokenizer file differs'):
        data.load_tokenizer(tmp_path)
