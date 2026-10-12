"""Count-derived unary controls on unchanged diffusion candidate support.

The smoothed count joint satisfies log p(b|a) = PMI(a,b) + log J_R(b).
Keeping only the destination-marginal term tests a frequency correction without
pair interactions or dynamic programming. This module never chooses candidates,
loads a checkpoint, or changes the sampler. Its inputs must come from training
counts, and candidate support must be selected before applying the correction.
"""
from __future__ import annotations

import math
from numbers import Real

import torch


def smoothed_unigram_log_probs(endpoint_counts, *, smoothing=0.1):
    """Return the actual endpoint marginal of the established smoothed joint.

    For N=sum counts and V vocabulary entries, J_R(v) is
    (1-eps)*counts[v]/N + eps*(counts[v]+1)/(N+V).
    This is not merely the add-one backoff distribution. Empty counts give the
    uniform distribution, matching SparseCountPotential.from_counts. Retain all
    V entries, including an unused mask slot if present in the count estimator;
    clean candidate selection separately excludes that slot.
    """
    if (isinstance(smoothing, bool) or not isinstance(smoothing, Real)
            or not math.isfinite(smoothing) or not 0 < smoothing <= 1):
        raise ValueError('Smoothing must be finite and in (0, 1]')
    if (endpoint_counts.ndim != 1 or endpoint_counts.numel() == 0
            or endpoint_counts.is_complex() or endpoint_counts.dtype == torch.bool):
        raise ValueError('Endpoint counts must be a nonempty real vector')
    counts = endpoint_counts.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
        raise ValueError('Endpoint counts must be finite and nonnegative')
    total = counts.sum()
    if not bool(torch.isfinite(total)):
        raise ValueError('Total endpoint count must be finite')
    n = float(total)
    eps = float(smoothing) if n else 1.0
    backoff = (counts + 1.0) / (n + counts.numel())
    marginal = (1.0 - eps) * counts / (n if n else 1.0) + eps * backoff
    return marginal.log()


def add_count_unigram(log_unary, candidate_ids, log_marginal,
                      incoming_edge_mask, *, strength=1.0):
    """Add lambda log J_R(token) exactly once per incoming chain edge.

    Inputs: log_unary and candidate_ids[B,L,K], log_marginal[V] from
    smoothed_unigram_log_probs, and explicit bool incoming_edge_mask[B,L].
    For a complete linear chain the first position has no incoming edge; all
    later positions have one. When applying to a component, its first position
    can still have an incoming edge from an observed boundary. Supplying this
    mask prevents a component boundary from silently changing the model.

    IDs -1 and V are padding and receive zero correction. Visible singleton
    scores only add constants. Zero strength returns the same unary tensor.
    Renormalization and independent sampling remain the caller's responsibility.
    """
    if (isinstance(strength, bool) or not isinstance(strength, Real)
            or not math.isfinite(strength) or strength < 0):
        raise ValueError('Strength must be finite and nonnegative')
    if (log_unary.ndim != 3 or min(log_unary.shape) < 1
            or not log_unary.is_floating_point()
            or candidate_ids.shape != log_unary.shape
            or candidate_ids.dtype != torch.long
            or candidate_ids.device != log_unary.device):
        raise ValueError('Expected matching floating unaries and int64 IDs[B,L,K]')
    if (incoming_edge_mask.shape != log_unary.shape[:2]
            or incoming_edge_mask.dtype != torch.bool
            or incoming_edge_mask.device != log_unary.device):
        raise ValueError('Expected bool incoming-edge mask[B,L] on the unary device')
    if (log_marginal.ndim != 1 or log_marginal.numel() == 0
            or log_marginal.dtype != torch.float64
            or log_marginal.device != log_unary.device):
        raise ValueError('Expected FP64 log marginal[V] on the unary device')
    vocab = log_marginal.numel()
    if bool(((candidate_ids < -1) | (candidate_ids > vocab)).any()):
        raise ValueError('Candidate IDs must be tokens, -1, or V padding')
    if (not bool(torch.isfinite(log_marginal).all())
            or abs(float(torch.logsumexp(log_marginal, 0))) > 1e-10):
        raise ValueError('Expected a finite normalized log marginal')
    if strength == 0:
        return log_unary
    valid = candidate_ids.ge(0) & candidate_ids.lt(vocab)
    safe_ids = candidate_ids.clamp(0, vocab - 1)
    correction = float(strength) * log_marginal[safe_ids]
    correction = torch.where(valid & incoming_edge_mask[..., None], correction, 0.0)
    return log_unary + correction.to(log_unary.dtype)
