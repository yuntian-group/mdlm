"""Exact revealed-block likelihood on a fixed, explicitly prepared support.

For selected sites R and a single shared chain potential, loss =
logZ(unary,pair) - logZ(unary clamped to gold ONLY at R,pair).
Unrevealed sites remain latent; observed singleton constraints remain intact.
Pair scores must be built AFTER support preparation, using returned token IDs.

prepare_training_support(..., missing='inject') is explicitly a TRAINING/DEV
SURROGATE: start with frozen top-p/top-k support, then add missing gold only at
selected R sites. This module never injects gold into a generation sampler.
Its default missing='error' instead rejects out-of-support selected targets.
Injection retains all original candidates (K may grow to K+1); it never
silently replaces a low-probability candidate or changes unseen intermediates.
"""
from dataclasses import dataclass

import torch

from .chain import log_partition


@dataclass
class TrainingSupport:
    candidate_ids: torch.Tensor
    log_unary: torch.Tensor
    vocab_size: int
    gold_slots: torch.Tensor  # [B,L]; -1 outside selected R
    reveal_mask: torch.Tensor
    diagnostics: dict


@dataclass
class BlockRevealLoss:
    per_sequence: torch.Tensor
    total_block_nll: torch.Tensor
    per_revealed_token_nll: torch.Tensor
    revealed_tokens: torch.Tensor
    log_partition: torch.Tensor
    clamped_log_partition: torch.Tensor


def _mask(value, shape, device, name):
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.bool or value.device != device:
        raise ValueError(f'{name} must be bool[B,L] on the score device')


def prepare_training_support(candidates_or_ids, log_unary=None, *, vocab_size=None,
        gold_tokens, reveal_mask, visible_mask=None, missing='error',
        full_log_probs=None, gold_scores_in_unary_scale=None):
    """Prepare support BEFORE constructing pair scores for the objective.

    Accept an existing CompactCandidates-like object, or plain IDs/unaries.
    Valid IDs occupy a prefix and are unique; padding is ID=vocab_size with
    unary=-inf. Full frozen model log probabilities are used only to recover
    the relative score of missing selected gold. Existing normalized candidate
    scores need a row-offset correction; appending a raw full-vocabulary log
    probability directly would be wrong. An explicitly same-scale gold-score
    tensor is an alternative for callers that already know that correction.
    Only R gold values are read/validated; other gold values cannot affect this
    support. visible_mask, when provided, forbids selecting observed tokens.
    """
    if hasattr(candidates_or_ids, 'candidate_ids'):
        if log_unary is not None or vocab_size is not None:
            raise ValueError('Do not mix CompactCandidates and plain score arguments')
        ids, values = candidates_or_ids.candidate_ids, candidates_or_ids.log_unary
        vocab_size = candidates_or_ids.vocab_size
    else:
        ids, values = candidates_or_ids, log_unary
    if (not isinstance(ids, torch.Tensor) or not isinstance(values, torch.Tensor)
            or ids.ndim != 3 or min(ids.shape) < 1 or ids.dtype != torch.long
            or values.shape != ids.shape or values.dtype not in (torch.float32, torch.float64)
            or values.device != ids.device or type(vocab_size) is not int or vocab_size < 1):
        raise ValueError('Expected int64 IDs and floating unaries[B,L,K], plus vocabulary size')
    finite = torch.isfinite(values)
    valid = (ids >= 0) & (ids < vocab_size)
    sizes = finite.sum(-1)
    if bool(torch.isnan(values).any() | torch.isposinf(values).any() | (finite != valid).any()
            | ((~finite) & ids.ne(vocab_size)).any() | sizes.eq(0).any()):
        raise ValueError('Valid IDs require finite scores; padding must be vocab_size/-inf')
    prefix = torch.arange(ids.shape[-1], device=ids.device) < sizes[..., None]
    ordered = ids.sort(-1).values
    if not torch.equal(finite, prefix) or bool(((ordered[..., 1:] == ordered[..., :-1])
                                              & ordered[..., 1:].lt(vocab_size)).any()):
        raise ValueError('Finite candidates must occupy a prefix and have unique token IDs')
    _mask(reveal_mask, ids.shape[:2], ids.device, 'reveal_mask')
    if visible_mask is not None:
        _mask(visible_mask, ids.shape[:2], ids.device, 'visible_mask')
        if bool((visible_mask & reveal_mask).any()):
            raise ValueError('Observed positions must not be selected for a reveal objective')
    if gold_tokens.shape != ids.shape[:2] or gold_tokens.dtype != torch.long or gold_tokens.device != ids.device:
        raise ValueError('gold_tokens must be int64[B,L] on the score device')
    if bool((((gold_tokens < 0) | (gold_tokens >= vocab_size)) & reveal_mask).any()):
        raise ValueError('Selected gold must be a valid vocabulary token')
    if missing not in ('error', 'inject'):
        raise ValueError('missing must be error or inject (training/dev surrogate only)')
    if full_log_probs is not None and gold_scores_in_unary_scale is not None:
        raise ValueError('Supply full_log_probs OR explicit same-scale gold scores, not both')
    found = ((ids == gold_tokens[..., None]) & finite).any(-1)
    absent = reveal_mask & ~found
    count, missing_count = int(reveal_mask.sum()), int(absent.sum())
    diagnostics = {'scope': 'training_or_development_objective_only_not_generation',
        'support_policy': 'strict_frozen_support' if missing == 'error' else 'selected_gold_augmented_surrogate',
        'selected_tokens': count, 'original_out_of_support_tokens': missing_count,
        'original_out_of_support_fraction': missing_count / max(1, count),
        'gold_injected_tokens': 0, 'original_width': ids.shape[-1],
        'prepared_width': ids.shape[-1], 'retained_candidates_removed': 0}
    if missing_count and missing == 'error':
        raise ValueError(f'{missing_count}/{count} selected gold tokens are outside frozen support')
    if missing_count:
        rows, positions = absent.nonzero(as_tuple=True)
        if full_log_probs is not None:
            if (full_log_probs.shape != (*ids.shape[:2], vocab_size)
                    or full_log_probs.device != ids.device or not full_log_probs.is_floating_point()):
                raise ValueError('full_log_probs must be frozen [B,L,V] scores on the candidate device')
            raw = full_log_probs[rows, positions].to(values.dtype)
            selected_ids = ids[rows, positions]
            original_raw = raw.gather(-1, selected_ids.clamp_max(vocab_size-1))
            offset = values[rows, positions, :1] - original_raw[:, :1]
            aligned = original_raw + offset
            keep = finite[rows, positions]
            if not bool(torch.isfinite(aligned[keep]).all()) or not torch.allclose(
                    aligned[keep], values[rows, positions][keep], rtol=1e-5, atol=1e-6):
                raise ValueError('Frozen full scores do not match retained unary relative scores')
            added = raw.gather(-1, gold_tokens[rows, positions, None]).squeeze(-1) + offset.squeeze(-1)
        elif gold_scores_in_unary_scale is not None:
            if (gold_scores_in_unary_scale.shape != ids.shape[:2]
                    or gold_scores_in_unary_scale.device != ids.device
                    or not gold_scores_in_unary_scale.is_floating_point()):
                raise ValueError('Same-scale gold scores must be floating[B,L] on the candidate device')
            added = gold_scores_in_unary_scale[rows, positions].to(values.dtype)
        else:
            raise ValueError('Gold injection needs original full scores or explicit same-scale gold scores')
        if not bool(torch.isfinite(added).all()):
            raise ValueError('Injected selected gold must have finite original model score')
        width = ids.shape[-1] + int(bool((sizes[absent] == ids.shape[-1]).any()))
        new_ids = torch.full((*ids.shape[:2], width), vocab_size, dtype=ids.dtype, device=ids.device)
        new_values = torch.full(new_ids.shape, -torch.inf, dtype=values.dtype, device=values.device)
        new_ids[..., :ids.shape[-1]], new_values[..., :ids.shape[-1]] = ids, values
        new_ids[rows, positions, sizes[absent]] = gold_tokens[rows, positions]
        new_values[rows, positions, sizes[absent]] = added
        # Sort only changed rows for the inference candidate convention.
        changed_ids, order = new_ids[rows, positions].sort(dim=-1, stable=True)
        changed_values = new_values[rows, positions].gather(-1, order)
        changed_values = changed_values - changed_values.logsumexp(-1, keepdim=True)
        new_ids[rows, positions], new_values[rows, positions] = changed_ids, changed_values
        ids, values = new_ids, new_values
        diagnostics.update(gold_injected_tokens=missing_count, prepared_width=width)
    matches = ids.eq(gold_tokens[..., None]) & torch.isfinite(values)
    slots = matches.long().argmax(-1)
    slots = torch.where(reveal_mask, slots, -1)
    return TrainingSupport(ids, values, vocab_size, slots, reveal_mask, diagnostics)


def block_reveal_nll(unary, pair, gold_slots, reveal_mask, lengths=None, *, memory_efficient=True):
    """Return exact block NLL and NLL divided by selected token count.

    Both partition functions share the SAME pair tensor and support. Only the
    numerator's selected R unaries are clamped. No hidden gold is conditioned
    on. Two custom logZ backpropagations safely sum gradients on shared scores.
    A selected target absent from support is an error, never silently dropped.

    Both logZ calls accumulate each local normalizer in float64 before this
    subtraction. The loss scalars are float64; normalized messages and score
    gradients still use the input dtype. This does not upgrade input scores or
    claim that float32 message arithmetic is identical to a float64 oracle.
    """
    _mask(reveal_mask, unary.shape[:2], unary.device, 'reveal_mask')
    if (gold_slots.shape != unary.shape[:2] or gold_slots.dtype != torch.long
            or gold_slots.device != unary.device):
        raise ValueError('gold_slots must be int64[B,L] on the unary device')
    if lengths is not None and bool((reveal_mask &
            (torch.arange(unary.shape[1], device=unary.device)[None] >= lengths[:, None])).any()):
        raise ValueError('Revealed targets must be inside sequence lengths')
    if bool((((gold_slots < 0) | (gold_slots >= unary.shape[-1])) & reveal_mask).any()):
        raise ValueError('Every selected target needs a valid in-support slot')
    selected_score = unary.gather(-1, gold_slots.clamp(0, unary.shape[-1]-1)[..., None]).squeeze(-1)
    if not bool(torch.isfinite(selected_score[reveal_mask]).all()):
        raise ValueError('Every selected target must have finite in-support unary score')
    denied = reveal_mask[..., None] & torch.arange(unary.shape[-1], device=unary.device).ne(gold_slots[..., None])
    clamped = unary.masked_fill(denied, -torch.inf)
    original_logz = log_partition(unary, pair, lengths, memory_efficient=memory_efficient)
    selected_logz = log_partition(clamped, pair, lengths, memory_efficient=memory_efficient)
    per_sequence = original_logz - selected_logz
    total = per_sequence.sum(dtype=torch.float64)
    tokens = reveal_mask.sum()
    return BlockRevealLoss(per_sequence, total, total / tokens.clamp_min(1), tokens,
                           original_logz, selected_logz)
