"""Strict local import of count statistics and signed additive-head exports.

The accepted formats are the actual tensor-only count/head interchange formats,
not training-resume files or positive low-rank probability mixtures. Loading
never downloads assets and never falls back to unrestricted pickle execution.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import re

import torch

from .backbone import CHECKPOINT_REVISION, file_sha256
from .heads import GlobalAdditiveHead, ContextualAdditiveHead

COUNT_FIELDS = ('left_counts', 'right_counts', 'pair_keys', 'pair_counts')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), 'Invalid ' + name)
    return float(value)


def load_payload(path, expected_sha256, *, max_bytes=2 * 1024**3):
    """Require explicit bytes, CPU tensors, weights_only=True, and a stable file."""
    path = Path(path).resolve()
    require(re.fullmatch('[0-9a-f]{64}', expected_sha256 or '') is not None,
            'An explicit lowercase SHA256 is required')
    require(path.is_file() and 0 < path.stat().st_size <= max_bytes,
            'Missing, empty or oversized asset file')
    require(file_sha256(path) == expected_sha256, 'Asset SHA256 mismatch')
    value = torch.load(path, map_location='cpu', weights_only=True)
    require(file_sha256(path) == expected_sha256, 'Asset changed during loading')
    require(type(value) is dict, 'Asset must be a dictionary')
    return value, {'path': str(path), 'sha256': expected_sha256}


@dataclass(frozen=True)
class CountAsset:
    statistics: dict
    identity: dict


def load_count_asset(path, expected_sha256, *, vocab_size=50258):
    value, identity = load_payload(path, expected_sha256)
    require(set(value) == {'format', 'vocab_size', 'mode', 'strength', 'smoothing', *COUNT_FIELDS}
            and value['format'] == 'chain_crf_counts_v1', 'Unsupported count asset schema')
    require(type(value['vocab_size']) is int and value['vocab_size'] == vocab_size,
            'Count/model vocabulary mismatch')
    require(value['mode'] in ('conditional', 'pmi'), 'Invalid stored count mode')
    number(value['strength'], 'stored count strength')
    require(0 < number(value['smoothing'], 'stored count smoothing') <= 1,
            'Stored smoothing must lie in (0,1]')
    tensors = {key: value[key] for key in COUNT_FIELDS}
    require(all(isinstance(t, torch.Tensor) and t.device.type == 'cpu'
                and t.layout == torch.strided and not t.requires_grad for t in tensors.values()),
            'Count arrays must be dense detached CPU tensors')
    left, right, keys, counts = (tensors[k] for k in COUNT_FIELDS)
    require(left.dtype == right.dtype == counts.dtype == torch.float64
            and keys.dtype == torch.int64 and left.shape == right.shape == (vocab_size,)
            and keys.ndim == 1 and counts.shape == keys.shape, 'Invalid count tensor shapes/dtypes')
    require(all(bool(torch.isfinite(t).all()) and bool((t >= 0).all())
                and torch.equal(t, t.round()) for t in (left, right, counts)),
            'Count statistics must be finite nonnegative integer-valued FP64')
    require(not bool(((keys < 0) | (keys >= vocab_size**2)).any())
            and not bool((keys[1:] <= keys[:-1]).any()) and bool((counts > 0).all()),
            'Observed pair keys must be sorted, unique and in range, with positive counts')
    observed_left, observed_right = torch.zeros_like(left), torch.zeros_like(right)
    observed_left.index_add_(0, keys // vocab_size, counts)
    observed_right.index_add_(0, keys % vocab_size, counts)
    require(torch.equal(left, observed_left) and torch.equal(right, observed_right),
            'Endpoint counts must exactly equal the pair-count marginals')
    # The last vocabulary row is diffusion's implicit MASK/PAD, never training text.
    require(left[-1] == 0 and right[-1] == 0, 'Raw training counts must not contain MASK/PAD')
    return CountAsset(tensors, {**identity, 'format': value['format'], 'vocab_size': vocab_size,
        'stored_mode': value['mode'], 'stored_strength': value['strength'],
        'stored_smoothing': value['smoothing'], 'observed_pair_types': keys.numel(),
        'observed_pair_total': float(counts.sum()), 'training_only_data_provenance_verified': False})


@dataclass(frozen=True)
class HeadAsset:
    head: torch.nn.Module
    identity: dict
    training: dict


def validate_head_payload(value, *, kind, vocab_size, model_revision):
    require(set(value) == {'schema', 'kind', 'head_config', 'head_state_dict',
                          'model_revision', 'training_identity', 'training_step'}
            and value['schema'] == 'additive_ladder_head_v1'
            and value['kind'] == kind and kind in ('global', 'contextual')
            and value['model_revision'] == model_revision, 'Unsupported additive-logscore head schema/model')
    config, state, training = value['head_config'], value['head_state_dict'], value['training_identity']
    fields = {'vocab_size', 'rank'} | ({'hidden_size', 'context_size'} if kind == 'contextual' else set())
    require(type(config) is dict and set(config) == fields
            and all(type(v) is int and v > 0 for v in config.values())
            and config['vocab_size'] == vocab_size and config['rank'] <= 128,
            'Invalid head configuration')
    require(type(value['training_step']) is int and value['training_step'] > 0,
            'Require a trained head export, not a zero-step initialization')
    require(isinstance(training, dict) and training.get('schema') == 'additive_ladder_training_identity_v1'
            and training.get('kind') == kind and training.get('head_config') == config,
            'Training identity does not describe the exported head')
    require(training.get('model', {}).get('checkpoint_revision') == model_revision,
            'Training backbone revision differs from the exported head')
    require('candidate-set surrogate' in training.get('objective', '')
            and 'NOT data PPL' in training.get('objective', '')
            and 'gold injection only at selected targets' in training.get('support', ''),
            'Expected joint-block candidate-set training, not a likelihood claim or old objective')
    number(training.get('count_strength'), 'training count strength')
    require(type(training.get('cap')) is int and 1 <= training['cap'] <= 2048
            and type(training.get('schedule_steps')) is int and training['schedule_steps'] > 0
            and 0 < number(training.get('top_p'), 'training top-p') <= 1,
            'Missing training support/schedule')
    inherited = training.get('inherited_global')
    if kind == 'global':
        require(inherited is None, 'A global head must not inherit another global head')
    else:
        require(isinstance(inherited, dict) and re.fullmatch('[0-9a-f]{64}', inherited.get('sha256', ''))
                and type(inherited.get('step')) is int and inherited['step'] > 0,
                'Contextual head needs its exact inherited global SHA/step')
        number(training.get('global_strength'), 'training global strength')
    require(isinstance(state, dict) and state and all(isinstance(t, torch.Tensor)
            and t.device.type == 'cpu' and t.layout == torch.strided and t.dtype == torch.float32
            and not t.requires_grad and bool(torch.isfinite(t).all()) for t in state.values()),
            'Head state must contain finite detached FP32 tensors only')
    return config, state, training


def load_head_asset(path, expected_sha256, *, kind, device='cpu', vocab_size=50258,
                    model_revision=CHECKPOINT_REVISION):
    value, identity = load_payload(path, expected_sha256)
    config, state, training = validate_head_payload(value, kind=kind, vocab_size=vocab_size,
                                                    model_revision=model_revision)
    # Module constructors initialize parameters; loading must not shift unrelated RNG streams.
    with torch.random.fork_rng(devices=[]):
        if kind == 'global':
            head = GlobalAdditiveHead(**config)
        else:
            embedding = state.get('token_embeddings')
            require(embedding is not None and embedding.ndim == 2
                    and embedding.shape[0] == vocab_size, 'Missing original frozen token embeddings')
            head = ContextualAdditiveHead(embedding, hidden_size=config['hidden_size'],
                                         rank=config['rank'], context_size=config['context_size'])
        head.load_state_dict(state, strict=True)
    head.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    summary = {**identity, 'schema': value['schema'], 'kind': kind, 'config': config,
               'model_revision': model_revision, 'training_step': value['training_step'],
               'training_objective_is_data_ppl': False}
    return HeadAsset(head, summary, training)


def validate_composition(global_asset, contextual_asset, *, count_strength, global_strength,
                         contextual_strength, count_mode, smoothing, cap, top_p, steps,
                         allow_recalibration=False):
    """Never substitute a contextual predecessor; declare any other recalibration."""
    changes = {}
    active = [(a, s) for a, s in ((global_asset, global_strength),
                                (contextual_asset, contextual_strength)) if s > 0]
    for asset, _ in active:
        require(asset is not None, 'Positive coefficient requires its matching head')
        trained = asset.training
        requested = {'count_strength': count_strength, 'cap': cap, 'top_p': top_p, 'schedule_steps': steps}
        row = {k: {'training': trained[k], 'generation': v} for k, v in requested.items() if trained[k] != v}
        # This existing schema was trained on the conditional, .1-smoothed base.
        if count_mode != 'conditional': row['count_mode'] = {'training': 'conditional', 'generation': count_mode}
        if smoothing != .1: row['smoothing'] = {'training': .1, 'generation': smoothing}
        if asset.identity['kind'] == 'contextual':
            inherited = trained['inherited_global']
            require(global_asset is not None and inherited['sha256'] == global_asset.identity['sha256']
                    and inherited['step'] == global_asset.identity['training_step'],
                    'Contextual head cannot substitute a different inherited global checkpoint')
            if trained['global_strength'] != global_strength:
                row['global_strength'] = {'training': trained['global_strength'], 'generation': global_strength}
        if row: changes[asset.identity['kind']] = row
    require(not changes or allow_recalibration,
            'Training anchor/support differs; require explicit --allow-recalibration')
    return changes
