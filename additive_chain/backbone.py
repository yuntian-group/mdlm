"""Frozen MDLM features and a local-only loader for the released BD3-LM baseline.

This module does not sample, train, pack data, or evaluate perplexity. Loading
uses the upstream implementation; one forward returns its unchanged log
probabilities and the input to its final output layer.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path
import sys

import torch


UPSTREAM_COMMIT = '1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7'
CHECKPOINT_REPO = 'kuleshov-group/mdlm-owt-noeos'
CHECKPOINT_REVISION = 'b9613c3c54da477d198c998e767ac3647e01225b'
TOKENIZER_REPO = 'gpt2'
TOKENIZER_REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
TOKENIZER_FILES = {
    'config.json': '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
    'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
    'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
    'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
}
CHECKPOINT_FILES = {
    'model.safetensors': '787b45c536ff7a4f847bfd09aaf86f9494e05e4e46be7ad43e2f0b560b062535',
    'config.json': '8f2cfa1afaa48bba538a1d20ae6df25f10023173c1417a44db1cb65fc79ecbfc',
    'configuration_bd3lm.py': '39e24124ab48a3e4be9c686aee35008b12eaae99d92e2f9d0cb282d807acc3d9',
    'modeling_bd3lm.py': 'c752db572ef63916df4987446a9dbb305f750e4e71a1194cc58ac1447d2d7c22',
}
UPSTREAM_CORE_FILES = {
    'diffusion.py': '0d56ac0cc76b7b13a15a252fd1c615c91f286de990706a2dea22b68654e04729',
    'dataloader.py': '35548b1e56f0610b326e39ed0f9dd8d9917eaabad90ffe389d5607d2a44edefc',
    'noise_schedule.py': 'd8c1a23bf70e753f9ee919c618597d0555897a9f445b6673e381ec10ff6dc849',
    'models/ema.py': 'f1417e66a46b2c2144e3fd47bb5c91c819f8c55f6bd18c75b72a75dc36bafb60',
    'models/dit.py': '0ef627114232f6fe7e1fe577d696ea01a0ac670a7865919f62e9708bd1cc0ac1',
    'metrics.py': '8ef7b49b236cc1c66fb822aa0445511ea5db4f9fc75533988be290b04e5775d0',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_files(directory, expected):
    directory = Path(directory).resolve()
    for name, digest in expected.items():
        _require(file_sha256(directory / name) == digest, 'File differs from pinned release: ' + name)
    return dict(expected)


def verify_tokenizer_files(directory):
    """Authenticate the original snapshot, without forbidding unrelated assets.

    These two optional Hugging Face files are absent from the pinned snapshot
    and can override token semantics even when all five required files match.
    A runtime-created PAD token stays in memory; do not save it into this
    authenticated source snapshot.
    """
    directory = Path(directory).resolve()
    for name in ('special_tokens_map.json', 'added_tokens.json'):
        path = directory / name
        _require(not path.exists() and not path.is_symlink(),
                 'Unexpected tokenizer override file: ' + name)
    return verify_files(directory, TOKENIZER_FILES)


def validate_tokenizer(tokenizer):
    """Check the result of the pinned upstream get_tokenizer, before loading weights.

    Native GPT-2 has 50,257 tokens and no MASK or PAD. Upstream adds [PAD]
    at 50,257, but vocab_size still excludes added tokens. Diffusion uses that
    same numeric ID for its implicit mask; it does not add a tokenizer MASK.
    """
    observed = {
        'vocab_size': tokenizer.vocab_size,
        'length_with_added_pad': len(tokenizer),
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'unk_token_id': tokenizer.unk_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'pad_token': tokenizer.pad_token,
        'mask_token_id': tokenizer.mask_token_id,
        'mask_token': tokenizer.mask_token,
    }
    expected = {'vocab_size': 50257, 'length_with_added_pad': 50258,
                'bos_token_id': 50256, 'eos_token_id': 50256, 'unk_token_id': 50256,
                'pad_token_id': 50257, 'pad_token': '[PAD]',
                'mask_token_id': None, 'mask_token': None}
    _require(observed == expected, 'Upstream-created GPT-2 tokenizer IDs or vocabulary changed')
    return observed


@dataclass(frozen=True)
class FrozenOutput:
    log_probs: torch.Tensor
    hidden: torch.Tensor


class FrozenMDLM:
    """Read detached pre-output-norm features from exactly one native forward.

    The API takes only corrupted tokens and time, never gold labels. Features
    remain usable as inputs to trainable heads, while backbone gradients are
    disabled. This synchronous hook interface is intentionally not reentrant.
    """
    def __init__(self, model, *, expected_length=1024, identity=None):
        self.model, self.length, self.identity = model, expected_length, identity or {}
        self.mask_id = model.mask_index
        self._active = False
        model.eval().requires_grad_(False)
        self.feature_layer = model.backbone.backbone.output_layer

    @torch.no_grad()
    def forward(self, tokens, t):
        _require(not self._active, 'FrozenMDLM is not reentrant')
        _require(not self.model.training, 'Frozen backbone must remain in evaluation mode')
        _require(tokens.dtype == torch.long and tokens.ndim == 2 and tokens.shape[1] == self.length
                 and bool(((tokens >= 0) & (tokens <= self.mask_id)).all()), 'Invalid masked token input')
        time = torch.as_tensor(t, dtype=torch.float32, device=tokens.device)
        if time.ndim == 0:
            time = time.expand(tokens.shape[0]).unsqueeze(-1)
        elif time.shape == (tokens.shape[0],):
            time = time.unsqueeze(-1)
        _require(time.shape == (tokens.shape[0], 1) and bool(torch.isfinite(time).all())
                 and bool(((time >= 0) & (time <= 1)).all()), 't must be scalar, [B], or [B,1] in [0,1]')
        features = []

        def capture(_module, inputs):
            _require(len(inputs) >= 1 and isinstance(inputs[0], torch.Tensor), 'Unexpected final-layer inputs')
            features.append(inputs[0].detach())

        self._active = True
        hook = self.feature_layer.register_forward_pre_hook(capture)
        try:
            _, move = self.model.noise(time)
            sigma = self.model._sigma_from_p(move)
            log_probs = self.model.forward(tokens, sigma, sample_mode=True)
            _require(len(features) == 1 and features[0].shape[:2] == tokens.shape,
                     'Expected exactly one final hidden state from one backbone call')
            _require(log_probs.shape == (*tokens.shape, self.mask_id + 1)
                     and log_probs.dtype == torch.float32 and not log_probs.requires_grad,
                     'Native FP32 log-probability interface changed')
            return FrozenOutput(log_probs.detach(), features[0])
        finally:
            hook.remove()
            self._active = False


@contextmanager
def _offline():
    keys = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE')
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update({key: '1' for key in keys})
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _check_imports(upstream):
    # BD3-LM uses unqualified imports. Never silently mix this repository's
    # older diffusion.py/models package with the separately pinned release.
    roots = ('diffusion', 'dataloader', 'metrics', 'models', 'noise_schedule', 'utils')
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in roots and getattr(module, '__file__', None):
            _require(Path(module.__file__).resolve().is_relative_to(upstream),
                     'Conflicting upstream module ' + name + '; load in a fresh Python process')


def compose_config(upstream, checkpoint, tokenizer, *, steps=32, top_p=.9):
    """Use upstream Hydra defaults, with the measured finite-step settings."""
    import hydra
    from omegaconf import OmegaConf

    _require(type(steps) is int and steps > 0, 'steps must be a positive integer')
    _require(0 < top_p <= 1, 'top_p must be in (0,1]')
    for key, function in {'cwd': os.getcwd, 'device_count': torch.cuda.device_count,
                          'eval': eval, 'div_up': lambda x, y: (x + y - 1) // y}.items():
        if not OmegaConf.has_resolver(key):
            OmegaConf.register_new_resolver(key, function)
    with hydra.initialize_config_dir(config_dir=str(Path(upstream).resolve() / 'configs'), version_base=None):
        config = hydra.compose(config_name='config', overrides=[
            'mode=sample_eval', 'algo=mdlm', 'algo.backbone=hf_dit', f'algo.T={steps}',
            'data=openwebtext-split', 'model=small', 'model.length=1024', 'block_size=1024',
            'model.attn_backend=sdpa', f'sampling.nucleus_p={top_p}', 'sampling.first_hitting=false',
            'sampling.kv_cache=false', 'sampling.var_length=false', 'sampling.num_sample_batches=1',
            'loader.eval_batch_size=1', 'loader.eval_global_batch_size=1', 'loader.batch_size=1',
            'loader.num_workers=0', 'trainer.devices=1', 'trainer.num_nodes=1', 'wandb=null',
            'seed=1', 'eval.perplexity_batch_size=1'])
    config.eval.checkpoint_path = str(Path(checkpoint).resolve())
    # Upstream constructs a metrics tokenizer during model initialization;
    # scoring itself is a separate call, not an implicit model download.
    config.eval.gen_ppl_eval_model_name_or_path = str(Path(tokenizer).resolve())
    config.data.tokenizer_name_or_path = str(Path(tokenizer).resolve())
    config.algo.ignore_bos = False
    config.data.insert_train_eos = config.data.insert_valid_eos = True
    config.data.insert_train_special = config.data.insert_valid_special = False
    return config


def load_bd3lm(upstream, checkpoint, tokenizer, *, device='cuda', steps=32, top_p=.9):
    """Load existing local assets; never download or write a checkpoint.

    Use a clean checkout of UPSTREAM_COMMIT and local snapshots of the pinned
    MDLM checkpoint and GPT-2 tokenizer. Core source, checkpoint and tokenizer
    hashes are checked; this does not verify every upstream source file.
    Call before importing either repository's top-level diffusion/model modules.
    """
    upstream, checkpoint, tokenizer = map(lambda p: Path(p).resolve(), (upstream, checkpoint, tokenizer))
    _require(all(p.is_dir() for p in (upstream, checkpoint, tokenizer)), 'Three existing local directories required')
    source_hashes = verify_files(upstream, UPSTREAM_CORE_FILES)
    checkpoint_hashes = verify_files(checkpoint, CHECKPOINT_FILES)
    tokenizer_hashes = verify_tokenizer_files(tokenizer)
    _check_imports(upstream)
    sys.path.insert(0, str(upstream))
    try:
        with _offline():
            dataloader = importlib.import_module('dataloader')
            diffusion = importlib.import_module('diffusion')
            _check_imports(upstream)
            config = compose_config(upstream, checkpoint, tokenizer, steps=steps, top_p=top_p)
            tokenization = dataloader.get_tokenizer(config)
            tokenizer_ids = validate_tokenizer(tokenization)
            model = diffusion.Diffusion(config, tokenizer=tokenization)
            _require(model.mask_index == 50257 and model.vocab_size == 50258,
                     'Native diffusion mask/vocabulary differs from pinned GPT-2 interface')
            model = model.to(device)
            # Preserve native EMA ordering before requires_grad_(False).
            if model.ema:
                model.ema.store(model._get_parameters())
                model.ema.copy_to(model._get_parameters())
    finally:
        sys.path.pop(0)
    from omegaconf import OmegaConf
    return FrozenMDLM(model, identity={
        'checkpoint_repo': CHECKPOINT_REPO, 'checkpoint_revision': CHECKPOINT_REVISION,
        'checkpoint_files_sha256': checkpoint_hashes, 'upstream_commit': UPSTREAM_COMMIT,
        'upstream_core_files_sha256': source_hashes,
        'tokenizer_repo': TOKENIZER_REPO, 'tokenizer_revision': TOKENIZER_REVISION,
        'tokenizer_files_sha256': tokenizer_hashes, 'tokenizer_ids': tokenizer_ids,
        'model_mask_id': model.mask_index, 'model_vocab_size': model.vocab_size,
        'config': OmegaConf.to_container(config, resolve=True),
        'ema_applied_before_freezing': bool(model.ema), 'backbone_frozen': True,
        'feature_location': 'final output-layer input, before output norm/projection'})
