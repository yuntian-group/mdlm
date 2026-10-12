"""Tiny local tensor fixtures; no real weights, downloads or CUDA execution."""
import copy
from pathlib import Path

import pytest
import torch

from additive_chain import assets
from additive_chain.backbone import CHECKPOINT_REVISION
from additive_chain.heads import GlobalAdditiveHead, ContextualAdditiveHead


def saved(tmp_path, value, name='asset.pt'):
    path = tmp_path / name
    torch.save(value, path)
    return path, assets.file_sha256(path)


def count_payload():
    return {'format': 'chain_crf_counts_v1', 'vocab_size': 4, 'mode': 'pmi',
        'strength': 7., 'smoothing': .1,
        'left_counts': torch.tensor([2, 3, 0, 0], dtype=torch.float64),
        'right_counts': torch.tensor([0, 2, 3, 0], dtype=torch.float64),
        'pair_keys': torch.tensor([1, 6]), 'pair_counts': torch.tensor([2, 3], dtype=torch.float64)}


def head_payload(kind='global', inherited=None):
    if kind == 'global':
        head = GlobalAdditiveHead(4, 2)
        config = {'vocab_size': 4, 'rank': 2}
    else:
        head = ContextualAdditiveHead(torch.arange(12).reshape(4, 3).float(), 3, 2, 2)
        config = {'vocab_size': 4, 'rank': 2, 'hidden_size': 3, 'context_size': 2}
    training = {'schema': 'additive_ladder_training_identity_v1', 'kind': kind,
        'head_config': config, 'model': {'checkpoint_revision': CHECKPOINT_REVISION},
        'count_strength': .1, 'cap': 500, 'top_p': .9, 'schedule_steps': 32,
        'global_strength': .25 if kind == 'contextual' else 1.,
        'inherited_global': inherited, 'objective': 'joint reveal; candidate-set surrogate; NOT data PPL',
        'support': 'frozen top-p/top-K with gold injection only at selected targets'}
    return {'schema': 'additive_ladder_head_v1', 'kind': kind, 'head_config': config,
        'head_state_dict': head.state_dict(), 'model_revision': CHECKPOINT_REVISION,
        'training_identity': training, 'training_step': 500}


def test_count_import_preserves_raw_statistics_not_stored_potential(tmp_path):
    payload = count_payload()
    value = assets.load_count_asset(*saved(tmp_path, payload), vocab_size=4)
    for key in assets.COUNT_FIELDS:
        assert torch.equal(value.statistics[key], payload[key])
    assert value.identity['stored_mode'] == 'pmi' and value.identity['stored_strength'] == 7
    assert value.identity['observed_pair_total'] == 5
    assert value.identity['training_only_data_provenance_verified'] is False


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(extra='unknown'),
    lambda p: p.update(format='positive_low_rank'),
    lambda p: p.update(vocab_size=5),
    lambda p: p.update(strength=float('nan')),
    lambda p: p.update(left_counts=p['left_counts'].float()),
    lambda p: p['left_counts'].__setitem__(0, 2.5),
    lambda p: p['left_counts'].__setitem__(0, 8),
    lambda p: p['pair_keys'].__setitem__(1, 1),
    lambda p: p['pair_counts'].__setitem__(0, float('nan')),
    lambda p: p.update(left_counts=torch.tensor([0, 0, 0, 5.], dtype=torch.float64),
        right_counts=torch.tensor([5., 0, 0, 0], dtype=torch.float64),
        pair_keys=torch.tensor([12]), pair_counts=torch.tensor([5.], dtype=torch.float64)),
])
def test_malformed_count_rejected(tmp_path, mutation):
    payload = count_payload(); mutation(payload)
    with pytest.raises(ValueError):
        assets.load_count_asset(*saved(tmp_path, payload), vocab_size=4)


def test_load_checks_hash_before_weights_only_and_never_uses_unsafe_fallback(tmp_path, monkeypatch):
    path, digest = saved(tmp_path, count_payload())
    calls = []
    actual = torch.load
    def spy(*args, **kwargs):
        calls.append(kwargs)
        return actual(*args, **kwargs)
    monkeypatch.setattr(assets.torch, 'load', spy)
    with pytest.raises(ValueError, match='SHA256'):
        assets.load_count_asset(path, '0' * 64, vocab_size=4)
    assert not calls
    assets.load_count_asset(path, digest, vocab_size=4)
    assert calls == [{'map_location': 'cpu', 'weights_only': True}]
    monkeypatch.setattr(assets.torch, 'load', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('unsafe custom object')))
    with pytest.raises(RuntimeError, match='unsafe custom object'):
        assets.load_count_asset(path, digest, vocab_size=4)


def test_global_import_exact_parameters_and_rng_preservation(tmp_path):
    payload = head_payload()
    path, digest = saved(tmp_path, payload)
    before = torch.random.get_rng_state().clone()
    loaded = assets.load_head_asset(path, digest, kind='global', vocab_size=4)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not loaded.head.training and all(not p.requires_grad for p in loaded.head.parameters())
    for key, tensor in payload['head_state_dict'].items():
        assert torch.equal(loaded.head.state_dict()[key], tensor)
    assert loaded.identity['sha256'] == digest and loaded.identity['training_step'] == 500


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(schema='legacy_positive_head'),
    lambda p: p.update(training_step=0),
    lambda p: p.update(model_revision='other'),
    lambda p: p['training_identity'].update(objective='marginal PPL'),
    lambda p: p['training_identity']['model'].update(checkpoint_revision='other'),
    lambda p: p['head_state_dict'].update(left_embedding=torch.zeros(4, 2)),
    lambda p: p['head_state_dict'].update({'left_embedding.weight': torch.zeros(4, 2, dtype=torch.float64)}),
    lambda p: p['head_state_dict']['left_embedding.weight'].__setitem__((0, 0), float('inf')),
])
def test_malformed_head_rejected(tmp_path, mutation):
    payload = head_payload(); mutation(payload)
    with pytest.raises((ValueError, RuntimeError)):
        assets.load_head_asset(*saved(tmp_path, payload), kind='global', vocab_size=4)


def composition(global_asset, local_asset=None, **kwargs):
    return assets.validate_composition(global_asset, local_asset, count_strength=.1,
        global_strength=.25, contextual_strength=1. if local_asset else 0.,
        count_mode='conditional', smoothing=.1, cap=500, top_p=.9, steps=32, **kwargs)


def test_contextual_import_binds_exact_inherited_global_and_declares_changes(tmp_path):
    g = assets.load_head_asset(*saved(tmp_path, head_payload(), 'global.pt'), kind='global', vocab_size=4)
    local = head_payload('contextual', {'sha256': g.identity['sha256'], 'step': 500})
    c = assets.load_head_asset(*saved(tmp_path, local, 'local.pt'), kind='contextual', vocab_size=4)
    assert composition(g, c) == {}
    options = dict(count_strength=.2, global_strength=.5, contextual_strength=2.,
        count_mode='pmi', smoothing=.2, cap=1000, top_p=.8, steps=16)
    with pytest.raises(ValueError, match='allow-recalibration'):
        assets.validate_composition(g, c, **options)
    changes = assets.validate_composition(g, c, **options, allow_recalibration=True)
    assert changes['contextual']['global_strength'] == {'training': .25, 'generation': .5}
    assert changes['global']['schedule_steps'] == {'training': 32, 'generation': 16}
    c.training['inherited_global']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='substitute'):
        assets.validate_composition(g, c, **options, allow_recalibration=True)


def test_missing_or_wrong_shaped_context_embedding_rejected(tmp_path):
    p = head_payload('contextual', {'sha256': 'a' * 64, 'step': 500})
    del p['head_state_dict']['token_embeddings']
    with pytest.raises(ValueError, match='embeddings'):
        assets.load_head_asset(*saved(tmp_path, p), kind='contextual', vocab_size=4)
