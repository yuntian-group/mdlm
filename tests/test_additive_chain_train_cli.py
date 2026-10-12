from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from additive_chain import train_cli as train
from additive_chain.assets import load_head_asset
from additive_chain.backbone import CHECKPOINT_REVISION
from additive_chain.counts import SparseCountPotential
from additive_chain.heads import ContextualAdditiveHead, GlobalAdditiveHead


class Rows:
    def __init__(self, split, rows=None):
        self.rows = torch.tensor(rows if rows is not None else [[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2]])
        self.identity = {'split': split, 'synthetic_fixture': True, 'rows': self.rows.tolist()}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Frozen:
    mask_id = 4
    identity = {'checkpoint_revision': CHECKPOINT_REVISION, 'synthetic_fixture': True}

    def __init__(self):
        self.calls = []
        self.model = torch.nn.Linear(1, 1).eval().requires_grad_(False)

    def forward(self, tokens, time):
        self.calls.append((tokens.clone(), time.clone()))
        logits = torch.tensor([1.2, .7, -.4, -1.3, -torch.inf], device=tokens.device)
        hidden = torch.arange(tokens.numel() * 3, dtype=torch.float32, device=tokens.device).reshape(*tokens.shape, 3) / 10
        return SimpleNamespace(log_probs=logits.expand(*tokens.shape, 5), hidden=hidden)


def potential():
    keys = torch.tensor([1, 7, 13, 15])
    values = torch.tensor([3., 2., 4., 1.], dtype=torch.float64)
    left, right = torch.zeros(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64)
    left.index_add_(0, keys // 5, values)
    right.index_add_(0, keys % 5, values)
    return SparseCountPotential.from_counts(left, right, keys, values, strength=.1)


def setup(output, **changes):
    torch.manual_seed(1)
    config = train.TrainingConfig(rank=2, context_size=3, top_k=2, schedule_steps=1,
        max_updates=4, min_updates=4, eval_every=2, dev_blocks=2, patience=4, checkpoint_every=2)
    config = replace(config, **changes)
    head = (GlobalAdditiveHead(5, 2) if config.kind == 'global' else
            ContextualAdditiveHead(torch.arange(15).reshape(5, 3).float() / 10,
                                   hidden_size=3, rank=2, context_size=3))
    return dict(training=Rows('train'), development=Rows('dev'), frozen=Frozen(),
                count_potential=potential(), count_identity={'sha256': 'a' * 64}, head=head,
                config=config, output=output, seconds_limit=120)


def checkpoint(output):
    return torch.load(output / 'last.pt', map_location='cpu', weights_only=True)


def assert_same_science(a, b):
    for key in ('step', 'cursor', 'stale', 'complete', 'generator', 'optimizer', 'head_state_dict',
                'initial_development', 'development'):
        assert train.tree_equal(a[key], b[key]), key
    for x, y in zip(a['training'], b['training']):
        assert {k: v for k, v in x.items() if k != 'seconds'} == {k: v for k, v in y.items() if k != 'seconds'}
    assert a['best']['step'] == b['best']['step']
    assert a['best']['mean_nll'] == b['best']['mean_nll']


def test_real_public_objective_optimizer_exports_and_resume_are_composed(tmp_path):
    full, split = tmp_path / 'full', tmp_path / 'split'
    train.run_training(**setup(full))
    first = train.run_training(**setup(split), max_new_updates=2)
    assert first['step'] == 2 and not first['complete']
    train.run_training(**setup(split), resume=True)
    assert_same_science(checkpoint(full), checkpoint(split))
    saved = checkpoint(split)
    assert saved['step'] == 4 and saved['complete']
    assert saved['identity']['residual_training_coefficient'] == 1
    assert saved['identity']['batch_size'] == 1
    assert saved['identity']['fixed_rng'] == {'initialization': 1, 'corruption': 7919,
                                             'order': 15485863, 'development': 104729}
    assert [r['step'] for r in saved['development']] == [2, 4]
    assert saved['initial_development']['metric'] == train.METRIC
    assert saved['identity']['order_sha256'] == train.value_sha(
        torch.randperm(3, generator=torch.Generator().manual_seed(15485863)).tolist())
    best = Path(saved['best']['immutable_head_path'])
    loaded = load_head_asset(best, saved['best']['immutable_head_sha256'], kind='global', vocab_size=5)
    assert loaded.identity['training_step'] == saved['best']['step']
    assert not loaded.head.training and all(not p.requires_grad for p in loaded.head.parameters())
    assert (split / 'training-history.json').exists()


def test_preflight_is_zero_update_and_resumed_trajectory_is_identical(tmp_path):
    pre, direct = tmp_path / 'pre', tmp_path / 'direct'
    kwargs = setup(pre)
    original = train.cpu_tree(kwargs['head'].state_dict())
    result = train.run_training(**kwargs, phase='preflight')
    saved = checkpoint(pre)
    assert result['passed'] and result['optimizer_updates'] == 0 and result['gradient_norm'] > 0
    assert saved['step'] == saved['cursor'] == 0 and not saved['optimizer']['state']
    assert train.tree_equal(saved['head_state_dict'], original)
    assert torch.equal(saved['generator'], torch.Generator().manual_seed(7919).get_state())
    train.run_training(**setup(pre), resume=True)
    train.run_training(**setup(direct))
    assert_same_science(checkpoint(pre), checkpoint(direct))
    with pytest.raises(ValueError, match='zero-step'):
        train.run_training(**setup(pre), resume=True, phase='preflight')


@pytest.mark.parametrize('at_maximum', [False, True])
def test_failed_development_replays_before_next_update_even_at_maximum(tmp_path, monkeypatch, at_maximum):
    path = tmp_path / 'run'
    kwargs = setup(path, max_updates=2 if at_maximum else 4, min_updates=2)
    original = train.evaluate_rows
    calls = []
    def fail(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('development failed')
        return original(*a, **k)
    monkeypatch.setattr(train, 'evaluate_rows', fail)
    with pytest.raises(RuntimeError, match='development failed'):
        train.run_training(**kwargs)
    saved = checkpoint(path)
    assert saved['step'] == 2 and saved['development'] == [] and saved['best'] is None
    before = saved['generator'].clone()
    observed = []
    def recover(*a, **k):
        observed.append(checkpoint(path)['step'])
        return original(*a, **k)
    monkeypatch.setattr(train, 'evaluate_rows', recover)
    kwargs = setup(path, max_updates=2 if at_maximum else 4, min_updates=2)
    result = train.run_training(**kwargs, resume=True)
    assert observed[0] == 2 and result['complete']
    assert [r['step'] for r in result['development']].count(2) == 1
    if at_maximum:
        assert torch.equal(checkpoint(path)['generator'], before)


def test_failed_export_is_not_committed_and_existing_snapshot_is_idempotent(tmp_path, monkeypatch):
    path = tmp_path / 'run'
    original = train.immutable_head
    def fail(path, payload):
        original(path, payload)
        raise OSError('after immutable export')
    monkeypatch.setattr(train, 'immutable_head', fail)
    with pytest.raises(OSError, match='after immutable'):
        train.run_training(**setup(path))
    saved = checkpoint(path)
    assert saved['step'] == 2 and saved['best'] is None and saved['development'] == []
    digest = train.file_sha256(path / 'head-step000002.pt')
    monkeypatch.setattr(train, 'immutable_head', original)
    train.run_training(**setup(path), resume=True)
    assert train.file_sha256(path / 'head-step000002.pt') == digest
    assert [x['step'] for x in checkpoint(path)['development']] == [2, 4]


def test_dev_pooled_normalization_fixed_population_rng_and_mode_restoration(monkeypatch):
    calls = []
    def step(clean, **kw):
        calls.append((clean.clone(), kw['generator'].get_state().clone()))
        torch.rand(1, generator=kw['generator'])
        index = len(calls) % 2
        return SimpleNamespace(metrics={'block_nll': 6. if index else 4., 'revealed_tokens': 3 if index else 1})
    monkeypatch.setattr(train, 'adaptation_step', step)
    head = GlobalAdditiveHead(5, 2).train()
    config = replace(train.TrainingConfig(), dev_blocks=2)
    data = Rows('dev')
    first = train.evaluate_rows(data, Frozen(), potential(), head, config)
    second = train.evaluate_rows(data, Frozen(), potential(), head, config)
    assert head.training and first['mean_nll'] == second['mean_nll'] == 2.5
    assert len(calls) == 4 and torch.equal(calls[0][0], data[0][None]) and torch.equal(calls[1][0], data[1][None])
    assert torch.equal(calls[0][1], calls[2][1]) and torch.equal(calls[1][1], calls[3][1])
    def failure(*a, **k):
        raise RuntimeError('dev fail')
    monkeypatch.setattr(train, 'adaptation_step', failure)
    with pytest.raises(RuntimeError):
        train.evaluate_rows(data, Frozen(), potential(), head, config)
    assert head.training


def test_patience_and_threshold_commit_only_after_snapshot_and_no_duplicates():
    config = replace(train.TrainingConfig(), max_updates=8, min_updates=4, eval_every=2, patience=2)
    state = {'step': 2, 'best': None, 'stale': 0, 'development': [], 'complete': False}
    events = []
    persist = lambda: (events.append('snapshot') or {'immutable_head_path': 'fixture', 'immutable_head_sha256': 'a' * 64})
    save = lambda: events.append('save')
    train.commit_development(state, config, lambda: {'mean_nll': 1.}, persist, save)
    assert events == ['snapshot', 'save'] and state['best']['step'] == 2
    assert train.commit_development(state, config, lambda: pytest.fail('duplicate evaluation'), persist, save) is None
    state['step'] = 4
    train.commit_development(state, config, lambda: {'mean_nll': 1. - 1e-6}, persist, save)
    assert state['stale'] == 1 and not state['complete']
    state['step'] = 6
    train.commit_development(state, config, lambda: {'mean_nll': 2.}, persist, save)
    assert state['complete'] and state['best']['step'] == 2 and events.count('snapshot') == 1
    with pytest.raises(ValueError, match='Older missing'):
        train.pending_development(dict(state, development=[]), config)


def test_empty_reveal_skips_optimizer_and_preserves_consumed_row_cursor(tmp_path, monkeypatch):
    original = train.adaptation_step
    seen = []
    def maybe_empty(*a, **k):
        # Two initial-development rows, then the first training row is empty.
        seen.append(1)
        if len(seen) == 3:
            torch.rand(1, generator=k['generator'])
            return SimpleNamespace(loss=None, metrics={'revealed_tokens': 0, 'empty_reveal': True})
        return original(*a, **k)
    monkeypatch.setattr(train, 'adaptation_step', maybe_empty)
    result = train.run_training(**setup(tmp_path / 'run'))
    assert result['step'] == 4 and result['cursor'] == 5 and len(result['training']) == 4


def test_contextual_training_freezes_predecessor_and_exports_compatible_identity(tmp_path):
    kwargs = setup(tmp_path / 'local', kind='contextual', global_strength=.25)
    previous = GlobalAdditiveHead(5, 2).eval().requires_grad_(False)
    with torch.no_grad():
        previous.right_embedding.weight.fill_(.1)
    before = train.cpu_tree(previous.state_dict())
    inherited = {'path': 'synthetic-global.pt', 'sha256': 'b' * 64, 'step': 2}
    result = train.run_training(**kwargs, global_head=previous, inherited=inherited)
    assert train.tree_equal(before, train.cpu_tree(previous.state_dict()))
    assert all(p.grad is None for p in previous.parameters())
    best = result['best']
    asset = load_head_asset(best['immutable_head_path'], best['immutable_head_sha256'],
                            kind='contextual', vocab_size=5)
    assert asset.training['inherited_global'] == inherited and asset.training['global_strength'] == .25


def test_resume_rejects_changed_science_and_tampered_best(tmp_path):
    path = tmp_path / 'run'
    train.run_training(**setup(path), max_new_updates=2)
    with pytest.raises(ValueError, match='Resume identity'):
        train.run_training(**setup(path, learning_rate=.002), resume=True)
    with pytest.raises(ValueError, match='empty output'):
        train.run_training(**setup(path))
    best = Path(checkpoint(path)['best']['immutable_head_path'])
    best.write_bytes(b'broken synthetic fixture')
    with pytest.raises(ValueError, match='best head'):
        train.run_training(**setup(path), resume=True)


def test_atomic_writer_failure_preserves_previous_bytes_and_cleans_temporary(tmp_path):
    path = tmp_path / 'last.pt'
    path.write_bytes(b'old')
    def fail(temporary):
        temporary.write_bytes(b'half-written')
        raise OSError('disk failure')
    with pytest.raises(OSError):
        train.atomic(path, fail)
    assert path.read_bytes() == b'old' and list(tmp_path.iterdir()) == [path]


def test_missing_identity_convenience_file_recovers_from_authoritative_checkpoint(tmp_path):
    path = tmp_path / 'run'
    train.run_training(**setup(path), max_new_updates=2)
    (path / 'identity.json').unlink()
    result = train.run_training(**setup(path), resume=True)
    assert result['complete'] and (path / 'identity.json').is_file()


def make_manifest(tmp_path, split, rows=None):
    array = np.array(rows if rows is not None else [[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)
    shard = tmp_path / (split + '.npy')
    np.save(shard, array)
    value = {'schema': 'packed_chain_adaptation_v1', 'split': split,
        'vocab_size': 5, 'window_length': 4, 'mask_id': 4, 'eos_id': 3,
        'document_eos_retained': True, 'window_bos_eos_added': False,
        'tokenizer': {'id': 'synthetic', 'revision': 'fixture'},
        'shards': [{'path': shard.name, 'sha256': train.file_sha256(shard), 'rows': len(array)}]}
    path = tmp_path / (split + '.json')
    path.write_text(json.dumps(value))
    return path, train.file_sha256(path)


def test_packed_inputs_authenticate_and_expose_only_separate_splits(tmp_path):
    training = train.PackedRows(*make_manifest(tmp_path, 'train'), split='train')
    development = train.PackedRows(*make_manifest(tmp_path, 'dev', [[2, 0, 3, 1]]), split='dev')
    train.validate_splits(training, development)
    assert torch.equal(training[0], torch.tensor([0, 1, 2, 3]))
    assert len(training) == 2 and len(development) == 1
    assert training.identity['document_disjointness_independently_verified'] is False
    with pytest.raises(ValueError, match='outside'):
        training[-1]
    with pytest.raises(ValueError, match='separate train/dev'):
        train.PackedRows(*make_manifest(tmp_path, 'test'), split='test')


def test_identical_data_and_manifest_or_shard_tampering_rejected(tmp_path):
    p, h = make_manifest(tmp_path, 'train')
    training = train.PackedRows(p, h, split='train')
    dev = train.PackedRows(*make_manifest(tmp_path, 'dev'), split='dev')
    with pytest.raises(ValueError, match='identical shard'):
        train.validate_splits(training, dev)
    with pytest.raises(ValueError, match='manifest SHA'):
        train.PackedRows(p, '0' * 64, split='train')
    (tmp_path / 'train.npy').write_bytes(b'bad')
    with pytest.raises(ValueError, match='shard SHA'):
        train.PackedRows(p, h, split='train')
    p, h = make_manifest(tmp_path, 'train', [[0, 1, 4, 2]])
    with pytest.raises(ValueError, match='MASK/PAD'):
        train.PackedRows(p, h, split='train')


def test_cli_requires_explicit_output_and_budget_and_has_no_seed_sweep():
    p = train.parser()
    options = {s for a in p._actions for s in a.option_strings}
    assert '--seed' not in options and '--seeds' not in options
    assert next(a for a in p._actions if a.dest == 'output').required
    assert next(a for a in p._actions if a.dest == 'seconds_limit').required
    assert set(train.source_identity()) == set(train.SOURCE_FILES)


def test_graceful_stop_persists_current_row_and_resume_continues(tmp_path):
    output = tmp_path / 'stopped'
    calls = []
    def stop():
        calls.append(1)
        return len(calls) > 1
    state = train.run_training(**setup(output), stop_requested=stop)
    assert state['step'] == checkpoint(output)['step'] == 1 and not state['complete']
    assert state['cursor'] == 1
    resumed = train.run_training(**setup(output), resume=True)
    assert resumed['complete'] and resumed['step'] == 4


def test_output_lock_rejects_concurrent_writer(tmp_path):
    path = tmp_path / 'locked'
    with train.output_lock(path, False):
        # Mimic a initialized run to get past the resume precondition.
        (path / 'last.pt').write_bytes(b'fixture')
        with pytest.raises(ValueError, match='Another process'):
            with train.output_lock(path, True):
                pytest.fail('second owner acquired lock')


def test_failed_optimizer_does_not_publish_half_updated_checkpoint(tmp_path, monkeypatch):
    output = tmp_path / 'failed'
    original = torch.optim.AdamW.step
    def fail(self, *a, **k):
        original(self, *a, **k)
        raise RuntimeError('optimizer interrupted')
    monkeypatch.setattr(torch.optim.AdamW, 'step', fail)
    with pytest.raises(RuntimeError, match='optimizer interrupted'):
        train.run_training(**setup(output))
    saved = checkpoint(output)
    assert saved['step'] == saved['cursor'] == 0
    assert not saved['optimizer']['state']
    assert saved['head_state_dict']['right_embedding.weight'].eq(0).all()
    monkeypatch.setattr(torch.optim.AdamW, 'step', original)
    train.run_training(**setup(output), resume=True)
    direct = tmp_path / 'direct'
    train.run_training(**setup(direct))
    assert_same_science(checkpoint(output), checkpoint(direct))


def test_actual_main_dispatch_uses_public_objective_checkpoint_and_export(tmp_path, monkeypatch):
    from additive_chain.backbone import TOKENIZER_REPO, TOKENIZER_REVISION
    inputs = {'train': Rows('train'), 'dev': Rows('dev')}
    for rows in inputs.values():
        rows.manifest = {'vocab_size': 50258, 'mask_id': 50257, 'eos_id': 50256,
            'window_length': 1024, 'tokenizer': {'id': TOKENIZER_REPO, 'revision': TOKENIZER_REVISION}}
    # Mock only external assets. The CLI configuration, real objective, optimizer,
    # serialization, development selection and actual asset consumer are composed.
    monkeypatch.setattr(train, 'PackedRows', lambda path, digest, split: inputs[split])
    monkeypatch.setattr(train, 'validate_splits', lambda a, b: None)
    monkeypatch.setattr(train, 'load_bd3lm', lambda *a, **k: Frozen())
    def load_count(path, digest, *, vocab_size):
        assert path == Path('counts.pt') and digest == 'a' * 64 and vocab_size == 5
        keys = torch.tensor([1, 7, 13, 15])
        values = torch.tensor([3., 2., 4., 1.], dtype=torch.float64)
        left, right = torch.zeros(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64)
        left.index_add_(0, keys // 5, values); right.index_add_(0, keys % 5, values)
        return SimpleNamespace(statistics={'left_counts': left, 'right_counts': right,
            'pair_keys': keys, 'pair_counts': values}, identity={'sha256': digest})
    monkeypatch.setattr(train.assets, 'load_count_asset', load_count)
    output = tmp_path / 'cli'
    argv = ['--upstream', 'upstream', '--checkpoint', 'checkpoint', '--tokenizer', 'tokenizer',
        '--train-manifest', 'train.json', '--train-manifest-sha256', 'b' * 64,
        '--dev-manifest', 'dev.json', '--dev-manifest-sha256', 'c' * 64,
        '--count-file', 'counts.pt', '--count-sha256', 'a' * 64, '--kind', 'global',
        '--output', str(output), '--seconds-limit', '120', '--device', 'cpu',
        '--rank', '2', '--top-k', '2', '--schedule-steps', '1', '--max-updates', '4',
        '--min-updates', '4', '--eval-every', '2', '--dev-blocks', '2', '--checkpoint-every', '2']
    assert train.main(argv + ['--max-new-updates', '2']) == 0
    assert checkpoint(output)['step'] == 2
    assert train.main(argv + ['--resume']) == 0
    saved = checkpoint(output)
    assert saved['complete'] and saved['step'] == 4
    best = saved['best']
    assert load_head_asset(best['immutable_head_path'], best['immutable_head_sha256'],
                           kind='global', vocab_size=5).training['counts']['sha256'] == 'a' * 64


@pytest.mark.parametrize('changes', [{'kind': 'unknown'}, {'rank': 0}, {'top_k': 0},
    {'global_strength': .25}, {'learning_rate': float('nan')}, {'min_updates': 5001}])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(train.TrainingConfig(), **changes).validate()
