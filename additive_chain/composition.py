"""Explicit count + global + contextual log-score composition.

This is a dense correctness/training reference, not a fused GPU implementation.
It allocates [B,L-1,K,K] scores. Supply count log probabilities already looked
up at the candidate token pairs; no full-vocabulary V-by-V table is required.
"""
from __future__ import annotations

import math
from numbers import Real

import torch

from .chain import sample_ffbs


def _coefficient(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or value < 0):
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    return float(value)


def factors_to_pair_scores(factors, *, dtype=None):
    """Return G_i[a,b]=left_i[a] dot right_{i+1}[b].

Both node-factor tensors have [B,L,K,R] shape. No positive-potential
factorization, probability normalization, or additional rank scaling occurs.
dtype defaults to the input dtype; an explicit float64 requests a float64
reference contraction of the supplied factors. Autocast is disabled.
"""
    left, right = factors
    if (left.ndim != 4 or min(left.shape) < 1 or right.shape != left.shape
            or left.device != right.device or left.dtype != right.dtype
            or left.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Factors must be matching float32/float64 [B,L,K,R] tensors')
    if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
        raise ValueError('Factor scores must be finite')
    dtype = left.dtype if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Reference pair scores must use float32 or float64')
    with torch.autocast(device_type=left.device.type, enabled=False):
        return torch.einsum('blar,blcr->blac', left[:, :-1].to(dtype), right[:, 1:].to(dtype))


def compose_pair_scores(count_log_scores, *, count_strength=1.,
                        global_factors=None, global_strength=0.,
                        contextual_factors=None, contextual_strength=0.):
    """Compose lambda*C + mu*G + nu*L with one shared support and orientation.

count_log_scores[B,L-1,K,K] contains UNSCALED log p_count(destination|source).
It must be finite (use a declared positive smoothing scheme). Its dtype/device
define those of the output. A zero tensor permits learned-only or independent
models. All strengths are explicit external coefficients; heads do not absorb
them. Counts may depend on edge position, e.g. a reset policy at boundaries,
but the caller must use the same rule for training and inference.

The function never truncates/reorders candidate sets. Compute factors/count
lookups AFTER any explicit training-only support augmentation. Observed tokens
must remain singleton-clamped in the unaries, including boundary neighbors.
"""
    if (count_log_scores.ndim != 4 or count_log_scores.shape[0] < 1
            or count_log_scores.shape[-1] < 1
            or count_log_scores.shape[-2] != count_log_scores.shape[-1]
            or count_log_scores.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(count_log_scores).all())):
        raise ValueError('Count log scores must be finite [B,L-1,K,K] float32/float64')
    count_strength = _coefficient(count_strength, 'count_strength')
    terms = ((global_factors, _coefficient(global_strength, 'global_strength')),
             (contextual_factors, _coefficient(contextual_strength, 'contextual_strength')))
    result = count_log_scores * count_strength
    expected = (count_log_scores.shape[0], count_log_scores.shape[1]+1,
                count_log_scores.shape[-1])
    for factors, strength in terms:
        if factors is None:
            if strength:
                raise ValueError('Nonzero residual strength requires its node factors')
            continue
        left, right = factors
        if (left.shape[:3] != expected or left.device != result.device
                or right.shape[:3] != expected or right.device != result.device):
            raise ValueError('Residual factors must share the count candidate layout/device')
        residual = factors_to_pair_scores(factors, dtype=result.dtype)
        result = result + strength * residual
    return result


def sample_tokens(candidate_ids, log_unary, pair, *, vocab_size,
                  lengths=None, num_samples=1, generator=None):
    """Joint FFBS samples[S,B,L] mapped from support slots to token IDs.

Finite unary entries require unique IDs in [0,vocab_size). Invalid slots
must have unary=-inf and ID=vocab_size. Padded sequence positions (lengths)
produce -1 tokens, distinct from candidate padding. The sampler receives no
gold targets and cannot augment support. It samples the declared restricted
CRF exactly up to floating-point arithmetic, not the full vocabulary model.
"""
    if (type(vocab_size) is not int or vocab_size < 1
            or candidate_ids.ndim != 3 or candidate_ids.dtype != torch.long
            or candidate_ids.shape != log_unary.shape
            or candidate_ids.device != log_unary.device):
        raise ValueError('Need int64 candidate IDs[B,L,K] and matching unaries')
    finite = torch.isfinite(log_unary)
    valid = candidate_ids.ge(0) & candidate_ids.lt(vocab_size)
    ordered = candidate_ids.sort(-1).values
    if bool((finite != valid).any() | ((~finite) & candidate_ids.ne(vocab_size)).any()
            | ((ordered[..., 1:] == ordered[..., :-1]) & ordered[..., 1:].lt(vocab_size)).any()):
        raise ValueError('Finite candidates need unique real IDs; padding uses vocab_size/-inf')
    slots = sample_ffbs(log_unary, pair, lengths, num_samples=num_samples, generator=generator)
    ids = candidate_ids[None].expand(num_samples, -1, -1, -1)
    tokens = ids.gather(-1, slots.clamp_min(0)[..., None]).squeeze(-1)
    return torch.where(slots.ge(0), tokens, -1)
