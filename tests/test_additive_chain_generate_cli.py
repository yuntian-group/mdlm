"""CLI routing with mocked loading; no real model, CUDA work or downloads."""
import json
from types import SimpleNamespace as NS

import pytest
import torch

from additive_chain import generate_cli as cli


class Tokenizer:
    bos_token_id = eos_token_id = 0
    def batch_decode(self, rows, skip_special_tokens):
        assert skip_special_tokens is False
        return [' '.join(map(str, row)) for row in rows]


def arguments(tmp_path, *extra):
    return cli.parser().parse_args(['--upstream', str(tmp_path), '--checkpoint', str(tmp_path),
        '--tokenizer', str(tmp_path), '--output', str(tmp_path / 'samples.json'),
        '--samples', '2', '--steps', '1', '--device', 'cpu',
        '--stream-namespace', 'cli-unit-test-only', *extra])


def fake_backbone():
    calls = []
    def forward(tokens, t):
        calls.append(t)
        output = torch.full((*tokens.shape, 66), -torch.inf)
        output[0, torch.arange(tokens.shape[1]), torch.arange(tokens.shape[1]).remainder(65)] = 0
        return output
    return NS(mask_id=65, forward=forward, model=NS(tokenizer=Tokenizer()),
              identity={'fixture': True}, calls=calls)


def test_composed_cli_real_baseline_core_serialization_and_no_overwrite(tmp_path, monkeypatch):
    frozen = fake_backbone(); calls = []
    monkeypatch.setattr(cli, 'load_bd3lm', lambda *a, **k: calls.append((a, k)) or frozen)
    a = arguments(tmp_path)
    value = cli.run(a)
    assert value == json.loads(a.output.read_text())
    assert value['status'] == 'complete' and len(value['attempts']) == 2
    assert len(frozen.calls) == 3  # One excluded warmup, two actual attempts.
    assert all(len(r['token_ids']) == 1024 and r['accepted'] for r in value['attempts'])
    assert value['assets']['backbone'] == {'fixture': True}
    assert value['source_sha256']['generate_cli.py'] == cli.file_sha256(cli.__file__)
    assert value['validation_scope']['actual_checkpoint_cuda_parity_asserted'] is False
    with pytest.raises(ValueError, match='new file'):
        cli.run(a)
    assert len(calls) == 1


@pytest.mark.parametrize('extra', [
    ('--count-strength', '.1'), ('--global-strength', '1'), ('--contextual-strength', '1'),
    ('--count-file', 'counts.pt'), ('--global-sha256', 'a' * 64),
    ('--steps', '0'), ('--top-p', 'nan'), ('--count-strength', 'nan'),
    ('--top-k', '0'), ('--stream-namespace', ' '), ('--device', 'mps'),
])
def test_invalid_requests_fail_before_backbone(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(cli, 'load_bd3lm', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not load')))
    with pytest.raises(ValueError): cli.run(arguments(tmp_path, *extra))


def test_declared_active_stage_arguments_and_cleanup_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.torch.cuda, 'is_available', lambda: True)
    frozen = fake_backbone()
    monkeypatch.setattr(cli, 'load_bd3lm', lambda *a, **k: frozen)
    head = object()
    g = NS(head=head, identity={'kind': 'global'}, training={})
    count = NS(statistics={'fixture': torch.tensor(1)}, identity={'stored_strength': 99.})
    monkeypatch.setattr(cli.assets, 'load_count_asset', lambda *a, **k: count)
    monkeypatch.setattr(cli.assets, 'load_head_asset', lambda *a, **k: g)
    seen = {}
    def validate(*a, **k): seen['composition'] = k; return {'recorded': 'change'}
    monkeypatch.setattr(cli.assets, 'validate_composition', validate)
    class Decoder:
        def __init__(self, **kwargs): seen['decoder'] = kwargs
        def prepare(self, **kwargs): seen['budget'] = kwargs; return {'seconds': 1.}
        def close(self): seen['closed'] = True
    monkeypatch.setattr(cli, 'ChainDecoder', Decoder)
    def generate(*a, **k):
        seen['generation'] = k
        raise RuntimeError('actual attempted computation failed')
    monkeypatch.setattr(cli, 'generate_saved', generate)
    a = arguments(tmp_path, '--device', 'cuda', '--count-file', 'count.pt', '--count-sha256', 'a' * 64,
        '--count-strength', '.1', '--global-head', 'head.pt', '--global-sha256', 'b' * 64,
        '--global-strength', '.25', '--backend', 'static', '--fast-candidates', '--allow-recalibration')
    with pytest.raises(RuntimeError, match='computation failed'): cli.run(a)
    assert seen['closed'] and not a.output.exists()
    assert seen['decoder']['global_head'] is head
    assert seen['decoder']['count_strength'] == .1 and seen['decoder']['global_strength'] == .25
    assert seen['decoder']['count_statistics'] is count.statistics
    assert seen['budget']['max_table_bytes'] == 12 * 1024**3
    assert seen['composition']['allow_recalibration'] is True
    assert seen['generation']['length'] == 1024 and seen['generation']['warmup'] is True


def test_retry_limit_saved_but_exit_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'run', lambda a: {'status': 'retry_limit',
        'attempts': [{'accepted': False}] * 10})
    a = arguments(tmp_path)
    monkeypatch.setattr(cli, 'parser', lambda: NS(parse_args=lambda argv: a))
    assert cli.main([]) == 2
    assert json.loads(capsys.readouterr().out)['raw_attempts'] == 10


def test_raw_cli_composes_with_generation_and_evaluation(tmp_path, monkeypatch):
    from additive_chain.evaluation import summarize_attempts
    frozen = fake_backbone()
    # All-EOS outputs fail acceptance; fixed-raw mode still finishes exactly12.
    def constant(tokens, t):
        result = torch.full((*tokens.shape, 66), -torch.inf)
        result[..., 0] = 0
        return result
    frozen.forward = constant
    monkeypatch.setattr(cli, 'load_bd3lm', lambda *a, **k: frozen)
    a = arguments(tmp_path, '--population', 'raw', '--samples', '12', '--no-warmup')
    value = cli.run(a)
    assert value == json.loads(a.output.read_text())
    assert value['status'] == 'complete' and len(value['attempts']) == 12
    assert value['population'] == 'raw' and not any(r['accepted'] for r in value['attempts'])
    summary = summarize_attempts(value['attempts'], eos_token_id=0,
        samples_requested=12, timing_complete=True, population=value['population'])
    assert summary['population_complete'] and summary['rejected'] == 12
    assert summary['seconds_per_raw_sequence'] > 0
    with pytest.raises(ValueError, match='new file'): cli.run(a)


def test_invalid_programmatic_population_rejected_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'load_bd3lm', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not load')))
    a = arguments(tmp_path)
    a.population = 'unknown'
    with pytest.raises(ValueError, match='population'): cli.run(a)
