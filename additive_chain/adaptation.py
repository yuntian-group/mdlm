"""One frozen-backbone adaptation step, not an experiment/training runner.

The learned residual has coefficient one during training. A supplied count
potential already contains its count strength. A contextual stage may inherit
an explicitly frozen global head at an explicitly supplied coefficient.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .counts import SparseCountPotential
from .fast_candidates import extract_compact_candidates
from .heads import ContextualAdditiveHead, GlobalAdditiveHead
from .composition import compose_pair_scores
from .training import BlockRevealLoss, TrainingSupport, block_reveal_nll, prepare_training_support


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _clean_tokens(clean, mask_id):
    _require(type(mask_id) is int and mask_id > 0 and isinstance(clean, torch.Tensor)
             and clean.dtype == torch.long and clean.ndim == 2 and min(clean.shape) > 0,
             'Expected nonempty int64 clean windows[B,L] and final MASK ID')
    _require(bool(((clean >= 0) & (clean < mask_id)).all()),
             'Clean windows must not contain MASK/PAD or out-of-vocabulary IDs')


def draw_corruption(clean, mask_id, generator, schedule_steps=32):
    """Sample j~Uniform{1,...,T}, masks~Bernoulli(j/T), reveal~mask*Bernoulli(1/j).

    Draw order matches the finite-step adaptation schedule: integer step,
    masking uniforms, then reveal uniforms. No BOS/EOS is added or protected.
    Empty reveal sets are legitimate and are never repaired by exposing gold.
    """
    _clean_tokens(clean, mask_id)
    _require(type(schedule_steps) is int and schedule_steps > 0, 'Positive schedule_steps required')
    _require(isinstance(generator, torch.Generator), 'Pass an explicit torch.Generator')
    step = torch.randint(1, schedule_steps + 1, (clean.shape[0], 1),
                         generator=generator, device=clean.device)
    time = step.float() / schedule_steps
    masked = torch.rand(clean.shape, generator=generator, device=clean.device) < time
    reveal = masked & (torch.rand(clean.shape, generator=generator, device=clean.device) < 1. / step)
    return torch.where(masked, mask_id, clean), reveal, time


@torch.no_grad()
def count_pair_scores(support, potential, *, edge_chunk=8):
    """Restricted scale-free count log scores[B,L-1,K,K], in FP32.

    Match the powered sparse-plus-background convention: observed entries use
    log(left*right+correction), other entries use log(left)+log(right), computed
    in FP64 then cast to FP32. The common log_scale is omitted from every edge;
    it cancels in both partition functions. No candidate normalization occurs.
    This portable reference uses searchsorted rather than a CUDA hash kernel.
    Temporary queries are edge-chunked, but the full dense pair tensor remains.
    """
    _require(isinstance(potential, SparseCountPotential), 'A frozen SparseCountPotential is required')
    ids = support.candidate_ids
    _require(support.vocab_size == potential.vocab_size and ids.device == potential.left.device,
             'Count/support vocabulary and device must agree')
    _require(type(edge_chunk) is int and edge_chunk > 0, 'Positive edge_chunk required')
    # Invalid slots are irrelevant under unary=-inf; use real ID0 for lookup.
    safe = ids.masked_fill(ids.eq(support.vocab_size), 0)
    keys = potential.pair_keys
    left, right = potential.left.log(), potential.right.log()
    values = (potential.left[keys // potential.vocab_size]
              * potential.right[keys % potential.vocab_size] + potential.row_values).log()
    batch, length, width = ids.shape
    result = torch.empty((batch, length - 1, width, width), device=ids.device, dtype=torch.float32)
    for start in range(0, length - 1, edge_chunk):
        stop = min(length - 1, start + edge_chunk)
        a, b = safe[:, start:stop, :, None], safe[:, start + 1:stop + 1, None, :]
        score = left[a] + right[b]
        if keys.numel():
            query = a * potential.vocab_size + b
            index = torch.searchsorted(keys, query.contiguous()).clamp_max(keys.numel() - 1)
            score = torch.where(keys[index] == query, values[index], score)
        result[:, start:stop] = score.float()
    _require(bool(torch.isfinite(result).all()), 'Nonfinite count pair score')
    return result


@dataclass
class AdaptationStep:
    objective: BlockRevealLoss | None
    support: TrainingSupport | None
    pair_scores: torch.Tensor | None
    corrupted_tokens: torch.Tensor
    reveal_mask: torch.Tensor
    time: torch.Tensor
    metrics: dict

    @property
    def loss(self):
        return None if self.objective is None else self.objective.per_revealed_token_nll


def loss_on_corruption(clean, corrupted, reveal_mask, time, *, frozen, count_potential,
                       head, global_head=None, global_strength=0., top_k=500,
                       nucleus_p=.9, missing='inject', edge_chunk=8,
                       memory_efficient=True):
    """One exact candidate-set revealed-block loss for a supplied corruption.

    ``frozen.forward(corrupted,time)`` returns detached log_probs and same-pass
    hidden features (e.g. FrozenMDLM). It is called once, on corrupted tokens
    only. All masked intermediates remain latent. Gold augmentation is confined
    to selected reveal targets and is a training/dev surrogate, NOT data PPL.

    ``head`` is the trainable global OR contextual residual, coefficient1.
    ``global_head`` is allowed only for a contextual stage and must already be
    frozen/eval. No module mode, parameter, optimizer or RNG is changed here.
    The caller must skip optimizer.step when result.loss is None (empty reveal).
    """
    _clean_tokens(clean, frozen.mask_id)
    shape, device = clean.shape, clean.device
    _require(isinstance(corrupted, torch.Tensor) and corrupted.shape == shape
             and corrupted.dtype == torch.long and corrupted.device == device,
             'Corrupted tokens must match clean shape/device/dtype')
    _require(bool(((corrupted == clean) | (corrupted == frozen.mask_id)).all()),
             'Corruption may only replace clean tokens with MASK')
    _require(isinstance(reveal_mask, torch.Tensor) and reveal_mask.shape == shape
             and reveal_mask.dtype == torch.bool and reveal_mask.device == device
             and not bool((reveal_mask & corrupted.ne(frozen.mask_id)).any()),
             'Reveal targets must be a bool mask selecting only masked positions')
    _require(isinstance(time, torch.Tensor) and time.shape == (shape[0], 1)
             and time.device == device and time.is_floating_point()
             and bool(torch.isfinite(time).all() & ((time > 0) & (time <= 1)).all()),
             'Time must be floating[B,1] in (0,1] on the token device')
    _require(isinstance(head, (GlobalAdditiveHead, ContextualAdditiveHead))
             and head.vocab_size == frozen.mask_id + 1, 'Wrong trainable additive head/vocabulary')
    _require(type(global_strength) in (int, float) and math.isfinite(global_strength)
             and global_strength >= 0, 'Explicit nonnegative global_strength required')
    if global_head is not None:
        _require(isinstance(head, ContextualAdditiveHead)
                 and isinstance(global_head, GlobalAdditiveHead)
                 and global_head.vocab_size == head.vocab_size and not global_head.training
                 and all(not p.requires_grad for p in global_head.parameters()),
                 'Only a contextual residual may inherit an already frozen/eval global head')
    else:
        _require(global_strength == 0, 'Nonzero global_strength needs its frozen predecessor')
    if not bool(reveal_mask.any()):
        return AdaptationStep(None, None, None, corrupted, reveal_mask, time,
                              {'revealed_tokens': 0, 'empty_reveal': True, 'backbone_calls': 0})
    with torch.no_grad():
        output = frozen.forward(corrupted, time)
        # Detach explicitly even for an externally supplied frozen interface.
        raw, hidden = output.log_probs.detach(), output.hidden.detach()
        candidates = extract_compact_candidates(raw, corrupted, mask_id=frozen.mask_id,
                                                nucleus_p=nucleus_p, top_k=top_k)
        support = prepare_training_support(candidates, gold_tokens=clean, reveal_mask=reveal_mask,
            visible_mask=corrupted.ne(frozen.mask_id), missing=missing, full_log_probs=raw)
        count = count_pair_scores(support, count_potential, edge_chunk=edge_chunk)
        inherited = None if global_head is None or not global_strength else global_head(
            support.candidate_ids, hidden)
    factors = head(support.candidate_ids, hidden)
    if isinstance(head, GlobalAdditiveHead):
        pair = compose_pair_scores(count, global_factors=factors, global_strength=1.)
    else:
        pair = compose_pair_scores(count, global_factors=inherited, global_strength=global_strength,
                                  contextual_factors=factors, contextual_strength=1.)
    objective = block_reveal_nll(support.log_unary.float(), pair, support.gold_slots,
                                reveal_mask, memory_efficient=memory_efficient)
    total = float(objective.total_block_nll.detach())
    if not math.isfinite(total) or total < -1e-3:
        raise FloatingPointError('Invalid candidate-set block likelihood')
    metrics = dict(block_nll=total, per_revealed_token_nll=float(objective.per_revealed_token_nll.detach()),
                   revealed_tokens=int(objective.revealed_tokens), empty_reveal=False,
                   time=time.detach().flatten().tolist(), backbone_calls=1, **support.diagnostics)
    return AdaptationStep(objective, support, pair, corrupted, reveal_mask, time, metrics)


def adaptation_step(clean, *, frozen, count_potential, head, generator,
                    schedule_steps=32, **kwargs):
    """Draw one corruption then return its loss; does not call backward/step."""
    corrupted, reveal, time = draw_corruption(clean, frozen.mask_id, generator, schedule_steps)
    return loss_on_corruption(clean, corrupted, reveal, time, frozen=frozen,
                              count_potential=count_potential, head=head, **kwargs)
