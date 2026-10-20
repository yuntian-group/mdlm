"""Source-conditioned revealed-block training of an additive target CRF.

The translator sees the source and one corrupted target. Its probabilities and
hidden states are frozen. The CRF contains target tokens only, including target
EOS; there is no source-to-target edge. Selected targets are clamped only in
the numerator partition. Other masked targets remain latent.

Candidate selection uses the untempered backbone probabilities. Missing gold
is appended only at selected training targets, before any pair score is built.
The complete energy is then divided by temperature, matching ChainDecoder.
The supplied count potential already contains its count strength (0.2 in the
IWSLT experiment); learned residuals have coefficient one.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .adaptation import count_pair_scores
from .composition import compose_pair_scores
from .fast_candidates import extract_compact_candidates
from .heads import ContextualAdditiveHead, GlobalAdditiveHead
from .training import BlockRevealLoss, TrainingSupport, block_reveal_nll, prepare_training_support
from .translation import MASK_ID, TranslationBatch


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _layout(batch, mask_id):
    _require(isinstance(batch, TranslationBatch), "Expected a TranslationBatch")
    tokens, valid, target = batch.tokens, batch.valid, batch.target
    _require(type(mask_id) is int and mask_id > 0
             and tokens.ndim == 2 and min(tokens.shape) > 0 and tokens.dtype == torch.long,
             "Expected nonempty int64 source/target tokens and final MASK ID")
    _require(valid.shape == tokens.shape and target.shape == tokens.shape
             and valid.dtype == target.dtype == torch.bool
             and valid.device == target.device == tokens.device,
             "Batch masks must be boolean tensors matching tokens")
    _require(bool(((tokens >= 0) & (tokens < mask_id)).all()),
             "Clean source/target tokens must not contain MASK or invalid IDs")
    size = tokens.shape[0]
    _require(len(batch.starts) == len(batch.lengths) == size
             and all(type(x) is int and x >= 1 for x in batch.starts)
             and all(type(x) is int and x >= 1 for x in batch.lengths),
             "Each example needs a nonempty source prefix and target including EOS")
    starts = torch.tensor(batch.starts, dtype=torch.long, device=tokens.device)
    lengths = torch.tensor(batch.lengths, dtype=torch.long, device=tokens.device)
    _require(bool((starts + lengths <= tokens.shape[1]).all()),
             "Target range exceeds the padded batch width")
    position = torch.arange(tokens.shape[1], device=tokens.device)[None]
    expected_valid = position < (starts + lengths)[:, None]
    expected_target = expected_valid & (position >= starts[:, None])
    _require(torch.equal(valid, expected_valid) and torch.equal(target, expected_target),
             "Expected contiguous source prefixes, target suffixes, and right padding")
    return starts, lengths


@dataclass
class TargetCorruption:
    tokens: torch.Tensor
    reveal_mask: torch.Tensor
    time: torch.Tensor
    schedules: torch.Tensor
    steps: torch.Tensor


def draw_target_corruption(batch, generator, *, schedule_steps=(2, 4, 8), mask_id=MASK_ID):
    """Draw T per example, j uniform in 1..T, mask j/T, and reveal 1/j.

    Source and padding are never masked or selected. EOS is an ordinary target
    site. Empty reveal sets are not repaired or retried.
    """
    _layout(batch, mask_id)
    _require(isinstance(generator, torch.Generator), "Pass an explicit torch.Generator")
    _require(isinstance(schedule_steps, (tuple, list)) and len(schedule_steps) > 0
             and all(type(t) is int and t > 0 for t in schedule_steps),
             "schedule_steps must be a nonempty sequence of positive integers")
    size, device = batch.tokens.shape[0], batch.tokens.device
    choices = torch.tensor(schedule_steps, dtype=torch.long, device=device)
    schedules = choices[torch.randint(len(choices), (size,), generator=generator, device=device)]
    # Uniform integer with row-dependent upper bound, without host round trips.
    steps = (torch.rand((size,), generator=generator, device=device) * schedules).long() + 1
    time = (steps.float() / schedules)[:, None]
    masked = (torch.rand(batch.tokens.shape, generator=generator, device=device) < time) & batch.target
    reveal = masked & (torch.rand(batch.tokens.shape, generator=generator, device=device)
                       < steps.float().reciprocal()[:, None])
    return TargetCorruption(batch.tokens.masked_fill(masked, mask_id), reveal, time,
                            schedules, steps)


@dataclass
class ResidualObjective:
    objective: BlockRevealLoss | None
    support: TrainingSupport | None
    unary_scores: torch.Tensor | None
    pair_scores: torch.Tensor | None
    corruption: TargetCorruption
    target_lengths: torch.Tensor
    metrics: dict

    @property
    def loss(self):
        return None if self.objective is None else self.objective.per_revealed_token_nll

    @property
    def total_nll(self):
        return None if self.objective is None else self.objective.total_block_nll

    @property
    def selected_tokens(self):
        return 0 if self.objective is None else int(self.objective.revealed_tokens)


def loss_on_target_corruption(batch, corruption, *, frozen, count_potential, head,
                              global_head=None, global_strength=0., temperature=.85,
                              nucleus_p=.9, top_k=500, edge_chunk=8,
                              memory_efficient=True, mask_id=MASK_ID, count_score_table=None):
    """Compute one target-only loss without changing module modes or parameters.

    ``frozen(tokens, valid)`` must return ``log_probs`` and same-pass ``hidden``.
    The caller keeps the frozen translator in evaluation mode. All its outputs
    and any inherited global head are detached. The new head is the only
    differentiable part. Callers accumulate ``total_nll`` and divide by the
    accumulated ``selected_tokens``; skip optimizer updates for an empty group.

    A small batch bounds the dense [B,target_length,K,K] pair tensor. Right-
    padded targets use explicit lengths, so neither padding nor source edges
    enter the partition functions.

    ``count_score_table`` optionally supplies a frozen FP32 [V,V] lookup table
    built once outside training. Its entries must equal count_pair_scores:
    the count strength is included, the common log_scale and temperature are
    not. The sparse lookup remains the default and uses much less storage.
    """
    starts, lengths = _layout(batch, mask_id)
    _require(isinstance(corruption, TargetCorruption), "Expected a TargetCorruption")
    corrupted, reveal = corruption.tokens, corruption.reveal_mask
    size, device, shape = batch.tokens.shape[0], batch.tokens.device, batch.tokens.shape
    _require(corrupted.shape == shape and corrupted.dtype == torch.long
             and corrupted.device == device
             and bool(((corrupted == batch.tokens) | ((corrupted == mask_id) & batch.target)).all()),
             "Corruption may replace only target tokens with MASK")
    _require(reveal.shape == shape and reveal.dtype == torch.bool and reveal.device == device
             and not bool((reveal & (~batch.target | corrupted.ne(mask_id))).any()),
             "Reveal mask must select only masked, unpadded target positions")
    _require(corruption.time.shape == (size, 1) and corruption.time.device == device
             and corruption.time.is_floating_point()
             and bool(torch.isfinite(corruption.time).all()
                      & ((corruption.time > 0) & (corruption.time <= 1)).all()),
             "Corruption time must be finite [B,1] in (0,1]")
    _require(corruption.steps.shape == corruption.schedules.shape == (size,)
             and corruption.steps.dtype == corruption.schedules.dtype == torch.long
             and corruption.steps.device == corruption.schedules.device == device
             and bool(((corruption.steps >= 1) & (corruption.steps <= corruption.schedules)).all())
             and torch.equal(corruption.time, (corruption.steps.float() / corruption.schedules)[:, None]),
             "Time, integer step, and schedule must agree")
    _require(type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0,
             "Temperature must be finite and positive")
    _require(isinstance(head, (GlobalAdditiveHead, ContextualAdditiveHead))
             and head.vocab_size == mask_id + 1, "Wrong additive head or vocabulary")
    _require(type(global_strength) in (int, float) and math.isfinite(global_strength)
             and global_strength >= 0, "Inherited strength must be finite and nonnegative")
    if global_head is not None:
        _require(isinstance(head, ContextualAdditiveHead)
                 and isinstance(global_head, GlobalAdditiveHead)
                 and global_head.vocab_size == head.vocab_size and not global_head.training
                 and all(not p.requires_grad for p in global_head.parameters()),
                 "Only a contextual head may inherit an already frozen/eval global head")
    else:
        _require(global_strength == 0, "Inherited strength requires a global head")
    _require(not getattr(frozen, "training", False), "The frozen translator must be in evaluation mode")
    if count_score_table is not None:
        _require(isinstance(count_score_table, torch.Tensor)
                 and count_score_table.shape == (mask_id + 1, mask_id + 1)
                 and count_score_table.dtype == torch.float32
                 and count_score_table.device == device and not count_score_table.requires_grad,
                 "Count table must be frozen FP32[V,V] on the token device")
        _require(count_potential.vocab_size == mask_id + 1
                 and count_potential.left.device == device,
                 "Count potential must match the table vocabulary and device")
    if not bool(reveal.any()):
        return ResidualObjective(None, None, None, None, corruption, lengths,
                                 {"revealed_tokens": 0, "empty_reveal": True, "backbone_calls": 0})

    # Gather target sites only. Padded rows use a harmless singleton token;
    # explicit target lengths exclude these nodes and all corresponding edges.
    positions = starts[:, None] + torch.arange(max(batch.lengths), device=device)[None]
    real = torch.arange(max(batch.lengths), device=device)[None] < lengths[:, None]
    gather_positions = positions.clamp_max(shape[1] - 1)
    rows = torch.arange(size, device=device)[:, None]
    clean_target = batch.tokens[rows, gather_positions].masked_fill(~real, 0)
    noisy_target = corrupted[rows, gather_positions].masked_fill(~real, 0)
    target_reveal = reveal[rows, gather_positions] & real
    with torch.no_grad():
        output = frozen(corrupted, batch.valid)
        _require(output.log_probs.shape == (*shape, mask_id + 1)
                 and output.hidden.ndim == 3 and output.hidden.shape[:2] == shape,
                 "Conditional model outputs must match its source/target input")
        raw = output.log_probs[rows, gather_positions].detach()
        hidden = output.hidden[rows, gather_positions].detach().masked_fill(~real[..., None], 0.)
        candidates = extract_compact_candidates(raw, noisy_target, mask_id=mask_id,
                                                nucleus_p=nucleus_p, top_k=top_k)
        support = prepare_training_support(candidates, gold_tokens=clean_target,
            reveal_mask=target_reveal, visible_mask=noisy_target.ne(mask_id),
            missing="inject", full_log_probs=raw)
        if count_score_table is None:
            count = count_pair_scores(support, count_potential, edge_chunk=edge_chunk)
        else:
            safe = support.candidate_ids.masked_fill(support.candidate_ids.eq(mask_id + 1), 0)
            count = count_score_table[safe[:, :-1, :, None], safe[:, 1:, None, :]]
            _require(bool(torch.isfinite(count).all()), "Nonfinite count table entries")
        inherited = None if global_head is None or not global_strength else global_head(
            support.candidate_ids, hidden)
    factors = head(support.candidate_ids, hidden)
    if isinstance(head, GlobalAdditiveHead):
        pair = compose_pair_scores(count, global_factors=factors, global_strength=1.)
    else:
        pair = compose_pair_scores(count, global_factors=inherited, global_strength=global_strength,
                                  contextual_factors=factors, contextual_strength=1.)
    pair = pair / temperature
    unary = support.log_unary.float() / temperature
    objective = block_reveal_nll(unary, pair, support.gold_slots, target_reveal, lengths,
                                memory_efficient=memory_efficient)
    total = float(objective.total_block_nll.detach())
    if not math.isfinite(total) or total < -1e-3:
        raise FloatingPointError("Invalid source-conditioned target block likelihood")
    metrics = dict(block_nll=total,
        per_revealed_token_nll=float(objective.per_revealed_token_nll.detach()),
        revealed_tokens=int(objective.revealed_tokens), empty_reveal=False, backbone_calls=1,
        target_tokens=int(lengths.sum()), target_lengths=batch.lengths,
        schedules=corruption.schedules.detach().tolist(), steps=corruption.steps.detach().tolist(),
        temperature=float(temperature), source_to_target_edges=0, **support.diagnostics)
    return ResidualObjective(objective, support, unary, pair, corruption, lengths, metrics)


def residual_objective(batch, *, generator, schedule_steps=(2, 4, 8), mask_id=MASK_ID, **kwargs):
    """Draw target corruption and return a differentiable block-loss numerator."""
    corruption = draw_target_corruption(batch, generator, schedule_steps=schedule_steps, mask_id=mask_id)
    return loss_on_target_corruption(batch, corruption, mask_id=mask_id, **kwargs)
