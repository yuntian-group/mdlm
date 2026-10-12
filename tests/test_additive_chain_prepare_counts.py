import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from additive_chain.assets import load_count_asset
from additive_chain.counts import SparseCountPotential
from additive_chain.prepare_counts import build_statistics, main, validate_manifest


def fixture(tmp_path):
    # 4 is article EOS, 5 is MASK. The (4,3) window-boundary edge is NOT counted.
    data = np.array([[1, 2, 4], [3, 4, 1]], dtype=np.int32)
    path = tmp_path / 'train.npy'
    np.save(path, data)
    manifest = dict(schema='packed_count_training_v1', split='train', vocab_size=6,
        window_length=3, mask_id=5, eos_id=4, document_eos_retained=True,
        window_bos_eos_added=False, tokenizer={'id': 'tiny', 'revision': 'fixture'},
        shards=[dict(path='train.npy', rows=2, sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    return manifest, path


def test_counts_respect_window_boundaries_and_keep_article_eos(tmp_path):
    manifest, _ = fixture(tmp_path)
    values, rows = build_statistics(manifest, base_dir=tmp_path, chunk_rows=1)
    assert rows == 2
    assert values['pair_keys'].tolist() == [8, 16, 22, 25]
    assert values['pair_counts'].tolist() == [1.] * 4
    assert values['left_counts'].tolist() == [0., 1., 1., 1., 1., 0.]
    assert values['right_counts'].tolist() == [0., 1., 1., 0., 2., 0.]
    large, _ = build_statistics(manifest, base_dir=tmp_path, chunk_rows=99)
    assert all(torch.equal(v, large[k]) for k, v in values.items())


def test_shards_do_not_gain_cross_shard_edges(tmp_path):
    manifest, path = fixture(tmp_path)
    other = tmp_path / 'second.npy'
    other.write_bytes(path.read_bytes())
    manifest['shards'].append({**manifest['shards'][0], 'path': other.name})
    values, rows = build_statistics(manifest, base_dir=tmp_path)
    assert rows == 4 and values['pair_counts'].tolist() == [2.] * 4


@pytest.mark.parametrize('field,value', [('split', 'validation'), ('split', 'test'),
    ('window_bos_eos_added', True), ('document_eos_retained', False), ('mask_id', 2),
    ('mask_id', 5.0),
    ('eos_id', 5), ('vocab_size', True), ('window_length', 1)])
def test_invalid_metadata(tmp_path, field, value):
    manifest, _ = fixture(tmp_path)
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_duplicate_resolved_shards_rejected(tmp_path):
    manifest, _ = fixture(tmp_path)
    manifest['shards'].append({**manifest['shards'][0], 'path': './train.npy'})
    with pytest.raises(ValueError, match='Duplicate resolved'):
        build_statistics(manifest, base_dir=tmp_path)


@pytest.mark.parametrize('invalid', [np.array([[1., 2., 4.]]), np.array([[1, 5, 4]]),
    np.array([[1, -1, 4]]), np.array([[1, 6, 4]]), np.array([[1, 2]])])
def test_invalid_actual_rows(tmp_path, invalid):
    manifest, path = fixture(tmp_path)
    np.save(path, invalid)
    manifest['shards'][0].update(rows=1, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        build_statistics(manifest, base_dir=tmp_path)


def test_content_digest_and_no_overwrite(tmp_path):
    manifest, _ = fixture(tmp_path)
    bad = copy.deepcopy(manifest)
    bad['shards'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA256'):
        build_statistics(bad, base_dir=tmp_path)
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(manifest))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / 'counts'
    argv = ['--manifest', str(source), '--manifest-sha256', digest, '--output', str(output)]
    main(argv)
    receipt = json.loads((output / 'manifest.json').read_text())
    loaded = load_count_asset(output / 'counts.pt', receipt['count_sha256'], vocab_size=6)
    potential = SparseCountPotential.from_counts(**loaded.statistics)
    ids = torch.arange(6)
    probabilities = potential.log_scores(ids[:, None], ids[None, :]).exp()
    assert torch.allclose(probabilities.sum(-1), torch.ones(6, dtype=torch.float64))
    assert receipt['tokens'] == 6 and receipt['edges'] == 4
    assert receipt['document_disjointness_independently_verified'] is False
    before = (output / 'counts.pt').read_bytes()
    with pytest.raises(ValueError, match='refusing to overwrite'):
        main(argv)
    assert (output / 'counts.pt').read_bytes() == before
