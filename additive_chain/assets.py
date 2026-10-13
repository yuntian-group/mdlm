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


def training_contract(training):
    """Read either declared objective without rewriting its original identity.

    The continuation export nests its inference anchors in pilot_protocol. Its
    source/data manifests do not establish a direct count-asset SHA, and are
    deliberately not converted into one here.
    """
    require(isinstance(training, dict), 'Missing head training identity')
    schema = training.get('schema')
    if schema == 'additive_ladder_training_identity_v1':
        require('candidate-set surrogate' in training.get('objective', '')
                and 'NOT data PPL' in training.get('objective', '')
                and 'gold injection only at selected targets' in training.get('support', ''),
                'Expected joint-block candidate-set training, not a likelihood claim or old objective')
        result = dict(training)
    elif schema == 'supported_training_continuation_v1':
        protocol = training.get('pilot_protocol')
        require(isinstance(protocol, dict)
                and protocol.get('support') == 'unchanged_generation_support'
                and protocol.get('gold_injection') is False
                and protocol.get('missing_target_policy') == 'latent_not_supervised'
                and protocol.get('empty_supervision_policy') == 'skip_optimizer_completely'
                and protocol.get('normalization') == 'original_selected_targets_R'
                and protocol.get('objective') == '-log q_C(gold_I)/|R|; I = selected targets with gold in C'
                and protocol.get('is_data_ppl') is False
                and protocol.get('is_full_reveal_likelihood') is False
                and type(protocol.get('new_head_strength')) in (int, float)
                and protocol['new_head_strength'] == 1.,
                'Invalid original-support partial-target objective')
        config = training.get('head_config', {})
        require(type(protocol.get('rank')) is int and protocol['rank'] == config.get('rank')
                and (training.get('kind') != 'contextual' or
                     protocol.get('contextual_size') == config.get('context_size')),
                'Training protocol/head configuration differs')
        plan, policy = training.get('selection_plan'), training.get('policy')
        require(isinstance(plan, dict) and isinstance(policy, dict)
                and plan.get('schema') == 'supported_training_selection_plan_v1'
                and plan.get('gold_injection') is False and plan.get('protocol') == 'training'
                and plan.get('selection_metric') == 'original_support_nll_per_original_selected_target'
                and policy.get('development_selection') == 'original_support_nll_per_R'
                and plan.get('cap') == protocol.get('cap') and plan.get('top_p') == protocol.get('top_p')
                and training.get('test_access') is False
                and training.get('inference_support_changed') is False,
                'Invalid original-support selection metadata')
        require(all(type(policy.get(k)) is int and policy[k] > 0 for k in
                    ('max_updates', 'min_updates', 'eval_every', 'patience', 'checkpoint_every', 'dev_cases'))
                and policy['min_updates'] <= policy['max_updates'], 'Invalid training selection policy')
        cases = plan.get('cases')
        require(isinstance(cases, list) and len(cases) == policy['dev_cases']
                and type(protocol.get('schedule_steps')) is int and protocol['schedule_steps'] > 0
                and all(isinstance(c, dict) and type(c.get('case_id')) is int and c['case_id'] == i
                        and type(c.get('validation_index')) is int and c['validation_index'] >= 0
                        and type(c.get('j')) is int and 1 <= c['j'] <= protocol.get('schedule_steps', 0)
                        and type(c.get('corruption_seed')) is int and c['corruption_seed'] >= 0
                        for i, c in enumerate(cases))
                and len({c['validation_index'] for c in cases}) == len(cases),
                'Invalid fixed training selection cases')
        inherited_sha = training.get('inherited_global_sha256')
        inherited = None if inherited_sha is None else {'sha256': inherited_sha}
        result = {k: protocol.get(k) for k in ('count_strength', 'cap', 'top_p', 'schedule_steps')}
        result.update(schema=schema, inherited_global=inherited,
                      global_strength=training.get('inherited_global_strength'),
                      counts=training.get('counts', {}))
    else:
        raise ValueError('Unsupported head training identity schema')
    number(result.get('count_strength'), 'training count strength')
    require(type(result.get('cap')) is int and 1 <= result['cap'] <= 2048
            and type(result.get('schedule_steps')) is int and result['schedule_steps'] > 0
            and 0 < number(result.get('top_p'), 'training top-p') <= 1,
            'Missing training support/schedule')
    return result


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
    require(isinstance(training, dict) and training.get('kind') == kind and training.get('head_config') == config,
            'Training identity does not describe the exported head')
    require(isinstance(training.get('model'), dict)
            and training['model'].get('checkpoint_revision') == model_revision,
            'Training backbone revision differs from the exported head')
    contract = training_contract(training)
    if training['schema'] == 'supported_training_continuation_v1':
        require(value['training_step'] <= training['policy']['max_updates'], 'Head step exceeds training policy')
    inherited = contract.get('inherited_global')
    if kind == 'global':
        require(inherited is None, 'A global head must not inherit another global head')
        if training['schema'] == 'supported_training_continuation_v1':
            require(type(contract['global_strength']) in (int, float) and contract['global_strength'] == 0.,
                    'Global continuation cannot have an inherited coefficient')
    else:
        require(isinstance(inherited, dict) and re.fullmatch('[0-9a-f]{64}', inherited.get('sha256', ''))
                and (training['schema'] == 'supported_training_continuation_v1' or
                     type(inherited.get('step')) is int and inherited['step'] > 0),
                'Contextual head needs its exact inherited global SHA/step')
        number(contract.get('global_strength'), 'training global strength')
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
               'training_schema': training['schema'],
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
        trained = training_contract(asset.training)
        requested = {'count_strength': count_strength, 'cap': cap, 'top_p': top_p, 'schedule_steps': steps}
        row = {k: {'training': trained[k], 'generation': v} for k, v in requested.items() if trained[k] != v}
        # This existing schema was trained on the conditional, .1-smoothed base.
        if count_mode != 'conditional': row['count_mode'] = {'training': 'conditional', 'generation': count_mode}
        if smoothing != .1: row['smoothing'] = {'training': .1, 'generation': smoothing}
        if asset.identity['kind'] == 'contextual':
            inherited = trained['inherited_global']
            require(global_asset is not None and inherited['sha256'] == global_asset.identity['sha256']
                    and ('step' not in inherited or inherited['step'] == global_asset.identity['training_step']),
                    'Contextual head cannot substitute a different inherited global checkpoint')
            if trained['global_strength'] != global_strength:
                row['global_strength'] = {'training': trained['global_strength'], 'generation': global_strength}
        if row: changes[asset.identity['kind']] = row
    require(not changes or allow_recalibration,
            'Training anchor/support differs; require explicit --allow-recalibration')
    return changes


def validate_training_predecessor(asset, *, count_strength, cap, top_p, steps, counts_sha256):
    """Training requires explicit count bytes; generation import alone does not."""
    trained = training_contract(asset.training)
    require(all(trained[k] == v for k, v in dict(count_strength=count_strength, cap=cap,
            top_p=top_p, schedule_steps=steps).items()),
            'Frozen global predecessor was trained with a different count/support/schedule anchor')
    require(re.fullmatch('[0-9a-f]{64}', counts_sha256 or '') is not None
            and isinstance(trained.get('counts'), dict)
            and trained['counts'].get('sha256') == counts_sha256,
            'Contextual training requires a predecessor bound to these exact count-asset bytes')
