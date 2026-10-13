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


def supported_payload(kind='global', inherited=None):
    """Synthetic tensors and public-schema fields; no historical weights/paths."""
    value = head_payload(kind, inherited)
    value['training_identity'] = dict(schema='supported_training_continuation_v1', kind=kind,
        head_config=value['head_config'], model={'checkpoint_revision': CHECKPOINT_REVISION},
        pilot_protocol=dict(cap=500, top_p=.9, schedule_steps=32, count_strength=.1,
            rank=2, contextual_size=2, new_head_strength=1., support='unchanged_generation_support',
            gold_injection=False, missing_target_policy='latent_not_supervised',
            empty_supervision_policy='skip_optimizer_completely', normalization='original_selected_targets_R',
            objective='-log q_C(gold_I)/|R|; I = selected targets with gold in C',
            is_data_ppl=False, is_full_reveal_likelihood=False),
        policy=dict(max_updates=5000, min_updates=1000, eval_every=250, patience=4,
            checkpoint_every=250, dev_cases=2, development_selection='original_support_nll_per_R'),
        selection_plan=dict(schema='supported_training_selection_plan_v1', gold_injection=False,
            protocol='training', selection_metric='original_support_nll_per_original_selected_target',
            cap=500, top_p=.9, cases=[dict(case_id=i, validation_index=i+32, j=j, corruption_seed=i+100)
                                    for i, j in enumerate((1, 32))]),
        inherited_global_sha256=None if inherited is None else inherited['sha256'],
        inherited_global_strength=.25 if kind == 'contextual' else 0.,
        test_access=False, inference_support_changed=False)
    return value


def test_supported_import_preserves_identity_and_composes_generation_without_relabeling(tmp_path):
    payload = supported_payload()
    original = copy.deepcopy(payload['training_identity'])
    path, digest = saved(tmp_path, payload)
    before = torch.random.get_rng_state().clone()
    loaded = assets.load_head_asset(path, digest, kind='global', vocab_size=4)
    assert torch.equal(before, torch.random.get_rng_state())
    assert loaded.training == original and loaded.identity['training_schema'] == original['schema']
    assert 'count_strength' not in loaded.training and 'counts' not in loaded.training
    assert composition(loaded) == {}
    for key, tensor in loaded.head.state_dict().items():
        assert torch.equal(tensor, payload['head_state_dict'][key])
    options = dict(count_strength=.1, cap=500, top_p=.9, steps=32, counts_sha256='a'*64)
    with pytest.raises(ValueError, match='exact count-asset bytes'):
        assets.validate_training_predecessor(loaded, **options)
    # Only explicit direct provenance, never data-directory/manifest inference.
    loaded.training['counts'] = {'sha256': 'a'*64}
    assets.validate_training_predecessor(loaded, **options)
    with pytest.raises(ValueError, match='exact count-asset bytes'):
        assets.validate_training_predecessor(loaded, **{**options, 'counts_sha256': 'b'*64})


@pytest.mark.parametrize('mutation', [
    lambda t: t.update(schema='unknown'),
    lambda t: t.update(kind='contextual'),
    lambda t: t.update(model={'checkpoint_revision': 'other'}),
    lambda t: t['pilot_protocol'].update(gold_injection=True),
    lambda t: t['pilot_protocol'].update(missing_target_policy='clamp'),
    lambda t: t['pilot_protocol'].update(empty_supervision_policy='zero_loss_optimizer_step'),
    lambda t: t['pilot_protocol'].update(normalization='supported_targets_I'),
    lambda t: t['pilot_protocol'].update(objective='data PPL'),
    lambda t: t['pilot_protocol'].update(is_data_ppl=True),
    lambda t: t['pilot_protocol'].update(is_full_reveal_likelihood=True),
    lambda t: t['pilot_protocol'].update(new_head_strength=True),
    lambda t: t['pilot_protocol'].update(rank=3),
    lambda t: t['pilot_protocol'].update(schedule_steps='32'),
    lambda t: t['pilot_protocol'].update(top_p=float('nan')),
    lambda t: t['selection_plan'].update(gold_injection=True),
    lambda t: t['selection_plan'].update(cases=[]),
    lambda t: t['selection_plan']['cases'][1].update(validation_index=32),
    lambda t: t['policy'].update(min_updates=6000),
    lambda t: t.update(test_access=True),
    lambda t: t.update(inference_support_changed=True),
    lambda t: t.update(inherited_global_sha256='a'*64),
    lambda t: t.update(inherited_global_strength=.25),
])
def test_supported_metadata_fail_closed(tmp_path, mutation):
    payload = supported_payload(); mutation(payload['training_identity'])
    with pytest.raises(ValueError):
        assets.load_head_asset(*saved(tmp_path, payload), kind='global', vocab_size=4)


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(training_step=5001),
    lambda p: p['head_state_dict'].update({'left_embedding.weight': torch.zeros(5, 2)}),
    lambda p: p['head_state_dict'].pop('right_embedding.weight'),
    lambda p: p['head_state_dict'].update({'left_embedding.weight': torch.zeros(4, 2, dtype=torch.float64)}),
    lambda p: p['head_state_dict']['left_embedding.weight'].__setitem__((0, 0), float('nan')),
])
def test_supported_weight_and_step_contract_and_rng_on_failure(tmp_path, mutation):
    payload = supported_payload(); mutation(payload)
    path, digest = saved(tmp_path, payload)
    before = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, RuntimeError)):
        assets.load_head_asset(path, digest, kind='global', vocab_size=4)
    assert torch.equal(before, torch.random.get_rng_state())


def test_supported_contextual_predecessor_binds_full_export_sha_without_invented_step(tmp_path):
    g = assets.load_head_asset(*saved(tmp_path, head_payload(), 'global.pt'), kind='global', vocab_size=4)
    payload = supported_payload('contextual', {'sha256': g.identity['sha256']})
    c = assets.load_head_asset(*saved(tmp_path, payload, 'contextual.pt'), kind='contextual', vocab_size=4)
    assert composition(g, c) == {}
    assert 'step' not in assets.training_contract(c.training)['inherited_global']
    c.training['inherited_global_sha256'] = 'a'*64
    with pytest.raises(ValueError, match='substitute'):
        composition(g, c, allow_recalibration=True)


def test_supported_predecessor_actual_training_cli_reaches_exact_count_guard(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from additive_chain import train_cli as train
    from additive_chain.backbone import TOKENIZER_REPO, TOKENIZER_REVISION
    rows = NS(manifest={'vocab_size': 50258, 'mask_id': 50257, 'eos_id': 50256,
        'window_length': 1024, 'tokenizer': {'id': TOKENIZER_REPO, 'revision': TOKENIZER_REVISION}})
    monkeypatch.setattr(train, 'PackedRows', lambda *a, **k: rows)
    monkeypatch.setattr(train, 'validate_splits', lambda *a: None)
    embedding = torch.arange(12).reshape(4, 3).float()
    frozen = NS(mask_id=3, model=NS(backbone=NS(backbone=NS(vocab_embed=NS(embedding=embedding)))))
    monkeypatch.setattr(train, 'load_bd3lm', lambda *a, **k: frozen)
    count_path, count_sha = saved(tmp_path, count_payload(), 'counts.pt')
    payload = supported_payload()
    head_path, head_sha = saved(tmp_path, payload, 'head.pt')
    argv = ['--upstream', 'upstream', '--checkpoint', 'checkpoint', '--tokenizer', 'tokenizer',
        '--train-manifest', 'train.json', '--train-manifest-sha256', 'a'*64,
        '--dev-manifest', 'dev.json', '--dev-manifest-sha256', 'b'*64,
        '--count-file', str(count_path), '--count-sha256', count_sha,
        '--global-head', str(head_path), '--global-sha256', head_sha,
        '--kind', 'contextual', '--global-strength', '.25', '--device', 'cpu',
        '--output', str(tmp_path/'training'), '--rank', '2', '--context-size', '2', '--seconds-limit', '120']
    with pytest.raises(ValueError, match='exact count-asset bytes'):
        train.main(argv)
    assert not (tmp_path/'training').exists()
    payload['training_identity']['counts'] = {'sha256': count_sha}
    new_path, new_sha = saved(tmp_path, payload, 'head-bound.pt')
    argv[argv.index(str(head_path))] = str(new_path)
    argv[argv.index(head_sha)] = new_sha
    seen = {}
    def run(**kwargs): seen.update(kwargs); return {'complete': False}
    monkeypatch.setattr(train, 'run_training', run)
    assert train.main(argv) == 0
    assert isinstance(seen['global_head'], GlobalAdditiveHead)
    assert not any(p.requires_grad for p in seen['global_head'].parameters())
    assert seen['inherited']['sha256'] == new_sha and seen['config'].kind == 'contextual'


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
