"""Selected-target training on unchanged generation candidates.

For a requested reveal set R, supervise I={i in R: gold_i in C_i}. Every
other masked variable, including an out-of-support target in R, stays latent.
This is an explicitly partial-target surrogate, not a likelihood of all the
data or all requested reveals. It does not replace the gold-augmented training
objective. No candidate is inserted, removed, reordered, or renormalized here.
"""
from dataclasses import dataclass

import torch

from .training import TrainingSupport, block_reveal_nll, prepare_training_support


@dataclass
class SupportedTargets:
    support: TrainingSupport
    selected_mask: torch.Tensor
    missing_mask: torch.Tensor
    diagnostics: dict


def prepare_supported_targets(candidates, *, gold_tokens, reveal_mask, visible_mask):
    """Separate available targets from missing ones without changing support.

    The original requested mask is retained for loss normalization. Invalid
    gold IDs at requested positions are errors, not missing-target examples.
    Gold values outside R cannot affect the prepared distribution or objective.
    """
    ids, unary = candidates.candidate_ids, candidates.log_unary
    empty = torch.zeros(ids.shape[:2], dtype=torch.bool, device=ids.device)
    # Validate the full candidate/gold/visible-mask contract, even with empty I.
    prepare_training_support(candidates, gold_tokens=gold_tokens,
        reveal_mask=empty, visible_mask=visible_mask, missing='error')
    if (not isinstance(reveal_mask, torch.Tensor) or reveal_mask.shape != ids.shape[:2]
            or reveal_mask.dtype != torch.bool or reveal_mask.device != ids.device):
        raise ValueError('Reveal mask must be bool[B,L] on the candidate device')
    if visible_mask is None:
        raise ValueError('An explicit visible mask is required')
    if bool((reveal_mask & visible_mask).any()):
        raise ValueError('Visible tokens cannot be supervised reveal targets')
    if bool((((gold_tokens < 0) | (gold_tokens >= candidates.vocab_size)) & reveal_mask).any()):
        raise ValueError('Every originally selected gold target must be a valid token')
    found = (ids.eq(gold_tokens[..., None]) & torch.isfinite(unary)).any(-1)
    inside, missing = reveal_mask & found, reveal_mask & ~found
    support = prepare_training_support(candidates, gold_tokens=gold_tokens,
        reveal_mask=inside, visible_mask=visible_mask, missing='error')
    if support.candidate_ids is not ids or support.log_unary is not unary:
        raise RuntimeError('Target preparation changed the generation support')
    selected, supervised = int(reveal_mask.sum()), int(inside.sum())
    diagnostics = dict(
        support_policy='unchanged_generation_support_selected_in_support_targets',
        original_selected_targets=selected, supervised_targets=supervised,
        out_of_support_targets=int(missing.sum()), gold_injected_tokens=0,
        retained_candidates_removed=0, prepared_width=ids.shape[-1],
        optimizer_update_has_supervision=supervised > 0,
        is_data_likelihood=False, is_full_revealed_block_likelihood=not bool(missing.any()))
    return SupportedTargets(support, reveal_mask.clone(), missing, diagnostics)


def supported_block_loss(targets, pair, *, memory_efficient=True):
    """Exact -log q_C(gold_I), marginalizing every other masked variable.

    Return both |R| and |I| normalizations. The former preserves the original
    example weighting when changing which targets supply supervision. An
    empty I has no information: callers MUST skip optimizer.step entirely,
    including Adam momentum and weight decay, when should_update is False.
    In a batch with some supported targets, retain every requested target in
    the |R| denominator and make one update from the batch objective.
    """
    support = targets.support
    unary = support.log_unary.float() if pair.dtype == torch.float32 else support.log_unary
    value = block_reveal_nll(unary, pair, support.gold_slots, support.reveal_mask,
                             memory_efficient=memory_efficient)
    denominator = targets.selected_mask.sum().clamp_min(1)
    return dict(total_nll=value.total_block_nll,
        per_original_selected_token_nll=value.total_block_nll / denominator,
        per_supported_target_nll=value.per_revealed_token_nll,
        should_update=targets.diagnostics['optimizer_update_has_supervision'],
        selected_targets=int(targets.selected_mask.sum()),
        supervised_targets=int(value.revealed_tokens))
