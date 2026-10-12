"""CPU contract checks; no checkpoint downloads or claimed GPU reproduction."""
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from additive_chain import backbone


class TinyNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_index = 5
        self.embedding = torch.nn.Embedding(6, 7)
        self.backbone = torch.nn.Module()
        self.backbone.backbone = torch.nn.Module()
        self.backbone.backbone.output_layer = torch.nn.Linear(7, 6)
        self.calls, self.fail = 0, False

    def noise(self, t):
        return -torch.ones_like(t), t

    def _sigma_from_p(self, p):
        return p * .5

    def forward(self, tokens, sigma, sample_mode=False):
        assert sample_mode
        self.calls += 1
        hidden = self.embedding(tokens).tanh() + sigma[:, :, None]
        logits = self.backbone.backbone.output_layer(hidden)
        if self.fail:
            raise RuntimeError('fixture failure')
        return logits.float().log_softmax(-1)


def fixture():
    model = TinyNative()
    tokens = torch.tensor([[5, 1, 5, 2], [0, 5, 5, 5]])
    return model, tokens, backbone.FrozenMDLM(model, expected_length=4)


def test_one_forward_preserves_native_probabilities_and_features():
    model, tokens, frozen = fixture()
    t = torch.tensor([[.2], [.7]])
    with torch.no_grad():
        expected = model(tokens, model._sigma_from_p(model.noise(t)[1]), sample_mode=True)
        hidden = model.embedding(tokens).tanh() + t[:, :, None] * .5
    before = model.calls
    actual = frozen.forward(tokens, t)
    assert model.calls == before + 1
    assert torch.equal(actual.log_probs, expected) and torch.equal(actual.hidden, hidden)
    assert not actual.hidden.requires_grad and not actual.log_probs.requires_grad
    assert not frozen.feature_layer._forward_pre_hooks
    assert not model.training and all(not p.requires_grad for p in model.parameters())


def test_head_can_train_without_backbone_gradients_or_gold_input():
    model, tokens, frozen = fixture()
    head = torch.nn.Linear(7, 3)
    head(frozen.forward(tokens, .5).hidden).square().mean().backward()
    assert head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    assert tuple(inspect.signature(frozen.forward).parameters) == ('tokens', 't')
    with pytest.raises(TypeError):
        frozen.forward(tokens, .5, gold=tokens)


def test_exception_removes_hook_and_allows_next_call():
    model, tokens, frozen = fixture()
    model.fail = True
    with pytest.raises(RuntimeError, match='fixture failure'):
        frozen.forward(tokens, .5)
    assert not frozen._active and not frozen.feature_layer._forward_pre_hooks
    model.fail = False
    assert frozen.forward(tokens, .5).hidden.shape == (2, 4, 7)


@pytest.mark.parametrize('t', [float('nan'), -.1, 1.1, [1., 2., 3.], [[.2, .3], [.4, .5]]])
def test_invalid_time_never_calls_backbone(t):
    model, tokens, frozen = fixture()
    with pytest.raises(ValueError):
        frozen.forward(tokens, t)
    assert model.calls == 0


def test_times_and_training_mode():
    model, tokens, frozen = fixture()
    a = frozen.forward(tokens, .5).log_probs
    assert torch.equal(a, frozen.forward(tokens, [.5, .5]).log_probs)
    assert torch.equal(a, frozen.forward(tokens, [[.5], [.5]]).log_probs)
    model.train()
    with pytest.raises(ValueError, match='evaluation mode'):
        frozen.forward(tokens, .5)


def test_source_verification_and_import_conflict(tmp_path, monkeypatch):
    path = tmp_path / 'diffusion.py'
    path.write_text('fixture')
    digest = backbone.file_sha256(path)
    assert backbone.verify_files(tmp_path, {'diffusion.py': digest}) == {'diffusion.py': digest}
    with pytest.raises(ValueError, match='pinned release'):
        backbone.verify_files(tmp_path, {'diffusion.py': '0' * 64})
    monkeypatch.setitem(sys.modules, 'diffusion', SimpleNamespace(__file__='/elsewhere/diffusion.py'))
    with pytest.raises(ValueError, match='fresh Python process'):
        backbone._check_imports(tmp_path)
    monkeypatch.setitem(sys.modules, 'diffusion', SimpleNamespace(__file__=str(path)))
    backbone._check_imports(tmp_path)


def test_offline_environment_restored_after_failure(monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.delenv('TRANSFORMERS_OFFLINE', raising=False)
    with pytest.raises(RuntimeError):
        with backbone._offline():
            assert backbone.os.environ['HF_HUB_OFFLINE'] == '1'
            raise RuntimeError('test')
    assert backbone.os.environ['HF_HUB_OFFLINE'] == '0'
    assert 'TRANSFORMERS_OFFLINE' not in backbone.os.environ


def test_loader_fails_before_import_or_download_when_assets_missing(tmp_path):
    with pytest.raises(ValueError, match='existing local directories'):
        backbone.load_bd3lm(tmp_path / 'absent', tmp_path, tmp_path)


class NativeGPT2Fixture(SimpleNamespace):
    """Only the observed upstream token contract, not a tokenizer implementation."""
    def __init__(self, **changes):
        values = dict(vocab_size=50257, length=50258, bos_token_id=50256,
                      eos_token_id=50256, unk_token_id=50256, pad_token_id=50257,
                      pad_token='[PAD]', mask_token_id=None, mask_token=None)
        values.update(changes)
        super().__init__(**values)

    def __len__(self):
        return self.length


def tokenizer_fixture_files(directory, monkeypatch):
    # Explicit tiny text fixtures; never model/tokenizer downloads or weights.
    expected = {}
    for name in backbone.TOKENIZER_FILES:
        path = directory / name
        path.write_text('fixture-only ' + name)
        expected[name] = backbone.file_sha256(path)
    monkeypatch.setattr(backbone, 'TOKENIZER_FILES', expected)
    return expected


def test_tokenizer_snapshot_constants_match_released_inventory():
    assert backbone.TOKENIZER_REPO == 'gpt2'
    assert backbone.TOKENIZER_REVISION == '607a30d783dfa663caf39e06633721c8d4cfcd7e'
    assert backbone.TOKENIZER_FILES == {
        'config.json': '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
        'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
        'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
        'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
        'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    }


@pytest.mark.parametrize('changed', list(backbone.TOKENIZER_FILES))
def test_each_tokenizer_file_is_authenticated(tmp_path, monkeypatch, changed):
    expected = tokenizer_fixture_files(tmp_path, monkeypatch)
    assert backbone.verify_tokenizer_files(tmp_path) == expected
    (tmp_path / changed).write_text('tampered fixture')
    with pytest.raises(ValueError, match='File differs from pinned release'):
        backbone.verify_tokenizer_files(tmp_path)


@pytest.mark.parametrize('name', ['special_tokens_map.json', 'added_tokens.json'])
@pytest.mark.parametrize('dangling_symlink', [False, True])
def test_tokenizer_override_files_rejected_even_if_required_hashes_match(
        tmp_path, monkeypatch, name, dangling_symlink):
    tokenizer_fixture_files(tmp_path, monkeypatch)
    path = tmp_path / name
    if dangling_symlink:
        path.symlink_to(tmp_path / 'missing-target')
    else:
        path.write_text('{}')
    with pytest.raises(ValueError, match='Unexpected tokenizer override'):
        backbone.verify_tokenizer_files(tmp_path)


def test_unrelated_snapshot_documentation_and_subdirectories_allowed(tmp_path, monkeypatch):
    expected = tokenizer_fixture_files(tmp_path, monkeypatch)
    (tmp_path / 'README.md').write_text('fixture documentation')
    (tmp_path / 'onnx').mkdir()
    (tmp_path / 'onnx' / 'special_tokens_map.json').write_text('not a root tokenizer override')
    assert backbone.verify_tokenizer_files(tmp_path) == expected


@pytest.mark.parametrize('changes', [
    {'vocab_size': 50258}, {'length': 50257}, {'length': 50259},
    {'bos_token_id': 0}, {'eos_token_id': 0}, {'unk_token_id': 0},
    {'pad_token_id': 50256}, {'pad_token': '<PAD>'},
    {'mask_token_id': 50257}, {'mask_token': '[MASK]'},
])
def test_native_gpt2_contract_checks_added_pad_separately_from_implicit_mask(changes):
    expected = backbone.validate_tokenizer(NativeGPT2Fixture())
    assert expected['vocab_size'] == 50257 and expected['length_with_added_pad'] == 50258
    assert expected['pad_token_id'] == 50257 and expected['mask_token_id'] is None
    with pytest.raises(ValueError, match='tokenizer IDs or vocabulary'):
        backbone.validate_tokenizer(NativeGPT2Fixture(**changes))


def test_loader_rejects_tokenizer_tamper_before_import_or_model_load(tmp_path, monkeypatch):
    tokenizer_fixture_files(tmp_path, monkeypatch)
    (tmp_path / 'vocab.json').write_text('tampered fixture')
    original_verify = backbone.verify_files
    def verify(directory, expected):
        if expected is backbone.TOKENIZER_FILES:
            return original_verify(directory, expected)
        return dict(expected)  # unrelated model/source authentication mocked
    monkeypatch.setattr(backbone, 'verify_files', verify)
    monkeypatch.setattr(backbone.importlib, 'import_module',
                        lambda name: pytest.fail('Tokenizer must fail before any import'))
    with pytest.raises(ValueError, match='vocab.json'):
        backbone.load_bd3lm(tmp_path, tmp_path, tmp_path, device='cpu')


@pytest.mark.parametrize('bad_tokenizer,bad_model', [(False, False), (True, False), (False, True)])
def test_loader_composes_tokenizer_identity_and_ema_before_freezing(
        tmp_path, monkeypatch, bad_tokenizer, bad_model):
    hashes = tokenizer_fixture_files(tmp_path, monkeypatch)
    original_verify = backbone.verify_files
    monkeypatch.setattr(backbone, 'verify_files', lambda directory, expected:
        original_verify(directory, expected) if expected is backbone.TOKENIZER_FILES else dict(expected))
    monkeypatch.setattr(backbone, 'compose_config', lambda *args, **kwargs: {'fixture': True})
    monkeypatch.setitem(sys.modules, 'omegaconf', SimpleNamespace(
        OmegaConf=SimpleNamespace(to_container=lambda config, resolve: dict(config))))
    tokenizer = NativeGPT2Fixture(bos_token_id=0) if bad_tokenizer else NativeGPT2Fixture()
    events, instances = [], []

    class LoaderNative(TinyNative):
        def __init__(self, config, tokenizer):
            assert backbone.os.environ['HF_HUB_OFFLINE'] == '1'
            events.append('construct')
            super().__init__()
            self.mask_index = 0 if bad_model else 50257
            self.vocab_size = 50258
            self.ema = SimpleNamespace(store=lambda parameters: self.check_ema(parameters, 'store'),
                                       copy_to=lambda parameters: self.check_ema(parameters, 'copy'))
            instances.append(self)

        def _get_parameters(self):
            return self.parameters()

        def check_ema(self, parameters, event):
            assert all(parameter.requires_grad for parameter in parameters)
            events.append(event)

        def to(self, device):
            assert device == 'cpu'
            events.append('to')
            return super().to(device)

        def requires_grad_(self, value):
            assert value is False
            events.append('freeze')
            return super().requires_grad_(value)

    modules = {
        'dataloader': SimpleNamespace(get_tokenizer=lambda config: tokenizer),
        'diffusion': SimpleNamespace(Diffusion=LoaderNative),
    }
    monkeypatch.setattr(backbone.importlib, 'import_module', lambda name: modules[name])
    path_before = list(sys.path)
    if bad_tokenizer or bad_model:
        with pytest.raises(ValueError, match='tokenizer IDs|diffusion mask'):
            backbone.load_bd3lm(tmp_path, tmp_path, tmp_path, device='cpu')
        assert events == ([] if bad_tokenizer else ['construct'])
    else:
        actual = backbone.load_bd3lm(tmp_path, tmp_path, tmp_path, device='cpu')
        assert events == ['construct', 'to', 'store', 'copy', 'freeze']
        assert actual.identity['tokenizer_files_sha256'] == hashes
        assert actual.identity['tokenizer_revision'] == backbone.TOKENIZER_REVISION
        assert actual.identity['tokenizer_ids'] == backbone.validate_tokenizer(tokenizer)
        assert actual.identity['model_mask_id'] == 50257
        assert actual.identity['model_vocab_size'] == 50258
        assert actual.identity['ema_applied_before_freezing']
        assert not actual.model.training and all(not p.requires_grad for p in actual.model.parameters())
    assert sys.path == path_before
