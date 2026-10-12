"""Stable full-sort top-p/intersect-top-K candidate reference.

The crossing token is excluded and the first token is always retained. This
is an explicit convention; it differs from include-crossing nucleus APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CompactCandidates:
    candidate_ids: torch.Tensor  # [B,L,K], sorted IDs; vocabulary size pads.
    log_unary: torch.Tensor      # [B,L,K], FP64, -inf on padding.
    vocab_size: int
    diagnostics: dict | None = None


@torch.no_grad()
def extract_compact_candidates(log_probs, tokens, *, mask_id, nucleus_p=.9,
                               top_k, row_chunk_size=32):
    """Apply the ORIGINAL nucleus, then the rank cap, with stable ID ties.

The cumulative sum uses probabilities normalized over the full clean
vocabulary, excludes the crossing token, and always keeps the first token.
Visible positions are singleton constraints. All hidden positions are kept.
Full sorts are bounded by row_chunk_size; only K values per row survive.
"""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(nucleus_p, (int, float)) or isinstance(nucleus_p, bool) or not math.isfinite(nucleus_p) or not 0 < nucleus_p <= 1:
        raise ValueError("nucleus_p must lie in (0,1]")
    if type(row_chunk_size) is not int or row_chunk_size < 1:
        raise ValueError("row_chunk_size must be a positive integer")
    if (log_probs.ndim != 3 or not log_probs.is_floating_point()
            or tokens.shape != log_probs.shape[:2] or tokens.dtype != torch.long
            or tokens.device != log_probs.device or min(log_probs.shape) < 1):
        raise ValueError("expected floating [B,L,V] logits and int64 [B,L] tokens")
    vocab = log_probs.shape[-1]
    if type(mask_id) is not int or not 0 <= mask_id < vocab:
        raise ValueError("mask_id must be in the vocabulary")
    if bool(((tokens < 0) | (tokens >= vocab)).any()):
        raise ValueError("tokens outside vocabulary")
    if bool(torch.isnan(log_probs).any() | torch.isposinf(log_probs).any()):
        raise ValueError("logits may contain finite values and -inf only")
    width = min(top_k, vocab)
    ids = torch.full((*tokens.shape, width), vocab, dtype=torch.long, device=tokens.device)
    values = torch.full(ids.shape, -torch.inf, dtype=torch.float64, device=tokens.device)
    ids[..., 0] = tokens
    values[..., 0] = 0.
    masked = tokens.eq(mask_id)
    # One initial index selection; do not transfer individual positions.
    flat_positions = masked.flatten().nonzero().flatten()
    flat_logits = log_probs.reshape(-1, vocab)
    sizes = torch.ones(tokens.shape, dtype=torch.long, device=tokens.device)
    original_sizes = sizes.clone()
    retained_mass = torch.ones(tokens.shape, dtype=torch.float64, device=tokens.device)
    nucleus_mass = retained_mass.clone()
    for start in range(0, flat_positions.numel(), row_chunk_size):
        positions = flat_positions[start:start + row_chunk_size]
        rows = flat_logits[positions].double().clone()
        rows[:, mask_id] = -torch.inf
        normalizer = rows.logsumexp(-1, keepdim=True)
        if not bool(torch.isfinite(normalizer).all()):
            raise ValueError("every hidden position needs a finite clean token")
        rows -= normalizer
        ordered, rank_ids = rows.sort(dim=-1, descending=True, stable=True)
        probability = ordered.exp()
        keep = torch.isfinite(ordered)
        if nucleus_p < 1:
            keep &= probability.cumsum(-1) <= nucleus_p
            keep[:, 0] = True
        original_sizes.flatten()[positions] = keep.sum(-1)
        nucleus_mass.flatten()[positions] = torch.where(keep, probability, 0.).sum(-1)
        chosen = keep[:, :width]
        capped = ordered[:, :width].masked_fill(~chosen, -torch.inf)
        retained_mass.flatten()[positions] = capped.exp().sum(-1)
        capped -= capped.logsumexp(-1, keepdim=True)
        chosen_ids = rank_ids[:, :width].masked_fill(~chosen, vocab)
        chosen_ids, permutation = chosen_ids.sort(dim=-1, stable=True)
        ids.reshape(-1, width)[positions] = chosen_ids
        values.reshape(-1, width)[positions] = capped.gather(-1, permutation)
        sizes.flatten()[positions] = chosen.sum(-1)
    return CompactCandidates(ids, values, vocab, {
        "masked": masked, "support_sizes": sizes, "nucleus_sizes": original_sizes,
        "retained_original_mass": retained_mass, "nucleus_original_mass": nucleus_mass,
    })

