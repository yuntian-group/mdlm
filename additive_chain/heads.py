"""Signed additive log-score factors for count, global, and contextual chains.

The CRF energy is sum(unary) + sum(count_score + lambda_g*G + lambda_l*L).
G(a,b)=A[a] dot B[b]. L_i(a,b)=U_i(a) dot V_{i+1}(b). Low rank applies ONLY
to these log-score residuals; exp(G) is not replaced by a low-rank positive
potential. There is no strength, softmax, softplus, or temperature in a head.
Stage strengths are external validation-selected coefficients.

node_factors returns two contiguous FP32 [B,L,K,R] arrays: the left factor at
node i and right factor at node i+1 define edge i, in source,destination order.
This factor layout avoids materializing pair matrices until a dense reference
consumer explicitly requests them. Count scores and residual factors are added
in log space; they are not alternative, mutually exclusive parameterizations.

All heads start at exactly zero residual score. One nonzero factor supplies
nonzero first-step gradients to the zero factor. The other factor's first-step
gradient is correctly zero, rather than evidence of a disconnected objective.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import torch
from torch import Tensor, nn


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _candidate_inputs(ids, vocab_size, device):
    if (ids.ndim != 3 or min(ids.shape) < 1 or ids.dtype != torch.long
            or ids.device != device):
        raise ValueError('candidate_ids must be nonempty int64 [B,L,K] on the head device')
    # Both frozen conventions are accepted: -1 or vocab_size denotes padding.
    if bool(((ids < -1) | (ids > vocab_size)).any()):
        raise ValueError('Candidate IDs must be real tokens, -1, or vocab_size padding')
    valid = ids.ge(0) & ids.lt(vocab_size)
    return ids.clamp(0, vocab_size - 1), valid[..., None]


class GlobalAdditiveHead(nn.Module):
    """Global signed log-bilinear residual; no dependence on context or gold.

    A starts N(0, 1/rank), i.e. standard deviation 1/sqrt(rank), B starts zero.
    Scores are exactly A dot B, with NO further 1/sqrt(rank) score scaling.
    vocab_size is the count/model vocabulary dimension used by the candidate
    packet; MASK must never occur as a finite candidate even if its embedding
    row is present. Candidate support validation remains the caller's job.
    """

    def __init__(self, vocab_size: int, rank: int = 64):
        super().__init__()
        self.vocab_size = _positive_int(vocab_size, 'vocab_size')
        self.rank = _positive_int(rank, 'rank')
        self.left_embedding = nn.Embedding(vocab_size, rank)
        self.right_embedding = nn.Embedding(vocab_size, rank)
        nn.init.normal_(self.left_embedding.weight, std=rank ** -.5)
        nn.init.zeros_(self.right_embedding.weight)

    def node_factors(self, candidate_ids: Tensor, hidden: Tensor | None = None):
        safe, valid = _candidate_inputs(candidate_ids, self.vocab_size,
                                        self.left_embedding.weight.device)
        if self.left_embedding.weight.dtype != torch.float32 or self.right_embedding.weight.dtype != torch.float32:
            raise ValueError('Keep additive-head parameters in FP32')
        left = self.left_embedding(safe) * valid
        right = self.right_embedding(safe) * valid
        return left.contiguous(), right.contiguous()

    forward = node_factors


class ContextualAdditiveHead(nn.Module):
    """Independent local residual from frozen token embeddings and masked h.

    U_i(a) = P_L(E_a) * (1 + g_L(LN(h_i))) + s_L(LN(h_i))
    V_i(a) = P_R(E_a) * (1 + g_R(LN(h_i))) + s_R(LN(h_i)).

    P_R and s_R start at zero; P_L, g_L and g_R are nonzero. Thus the initial
    residual is zero but the first update can learn contextual right factors.
    Feature LayerNorm is NOT a normalization of token/pair probabilities.
    Gates and residual scores are signed and unbounded; no positivity wrapper.
    Shifts allow unary effects as well as pair interactions: this is deliberate
    and should not be presented as a dependency-only capacity control.

    The embedding buffer is cloned/detached at construction; hidden states are
    detached at every call. Pass h from the SAME corrupted x_t used for base
    logits. Never rerun the encoder on gold-filled or reveal-clamped inputs.
    This module has no gold input. Data/support construction stays outside it.
    """

    def __init__(self, frozen_token_embeddings: Tensor, hidden_size: int | None = None,
                 rank: int = 64, context_size: int = 128):
        super().__init__()
        if (not isinstance(frozen_token_embeddings, Tensor)
                or frozen_token_embeddings.ndim != 2
                or min(frozen_token_embeddings.shape) < 1
                or not frozen_token_embeddings.is_floating_point()
                or not bool(torch.isfinite(frozen_token_embeddings).all())):
            raise ValueError('Frozen embeddings must be a finite floating [V,D] tensor')
        self.vocab_size, self.embedding_size = frozen_token_embeddings.shape
        self.hidden_size = _positive_int(hidden_size if hidden_size is not None
                                        else self.embedding_size, 'hidden_size')
        self.rank = _positive_int(rank, 'rank')
        self.context_size = _positive_int(context_size, 'context_size')
        self.register_buffer('token_embeddings', frozen_token_embeddings.detach().float().clone())
        self.hidden_norm = nn.LayerNorm(self.hidden_size)
        self.context = nn.Sequential(nn.Linear(self.hidden_size, context_size), nn.SiLU())
        self.left_token = nn.Linear(self.embedding_size, rank, bias=False)
        self.right_token = nn.Linear(self.embedding_size, rank, bias=False)
        self.left_gate = nn.Linear(context_size, rank)
        self.right_gate = nn.Linear(context_size, rank)
        self.left_shift = nn.Linear(context_size, rank)
        self.right_shift = nn.Linear(context_size, rank)
        nn.init.normal_(self.left_token.weight, std=self.embedding_size ** -.5)
        nn.init.zeros_(self.right_token.weight)
        for gate in (self.left_gate, self.right_gate):
            nn.init.normal_(gate.weight, std=.02)
            nn.init.zeros_(gate.bias)
        for shift in (self.left_shift, self.right_shift):
            nn.init.zeros_(shift.weight)
            nn.init.zeros_(shift.bias)
        # Keep the supplied table's device, with learned parameters in FP32.
        self.to(device=self.token_embeddings.device, dtype=torch.float32)

    def node_factors(self, candidate_ids: Tensor, hidden: Tensor):
        safe, valid = _candidate_inputs(candidate_ids, self.vocab_size, self.token_embeddings.device)
        if (not isinstance(hidden, Tensor) or hidden.shape != (*candidate_ids.shape[:2], self.hidden_size)
                or hidden.device != candidate_ids.device or not hidden.is_floating_point()):
            raise ValueError('hidden must be floating [B,L,H] on the head device')
        if self.left_token.weight.dtype != torch.float32 or self.token_embeddings.dtype != torch.float32:
            raise ValueError('Keep additive-head parameters and frozen embeddings in FP32')
        # The frozen backbone may use BF16 autocast; do not inherit it here.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            context = self.context(self.hidden_norm(hidden.detach().float()))
            # Project the frozen V-by-D table BEFORE candidate gathering.
            # Gathering [B,L,K,D] first would retain ~1.6 GB at B1/L1024/K500/
            # D768 merely for each projection's input. This retains V-by-R
            # tables and B-by-L-by-K-by-R factors instead, with exact gradients.
            left_table = self.left_token(self.token_embeddings)
            right_table = self.right_token(self.token_embeddings)
            left = (left_table[safe] * (1. + self.left_gate(context)[..., None, :])
                    + self.left_shift(context)[..., None, :])
            right = (right_table[safe] * (1. + self.right_gate(context)[..., None, :])
                     + self.right_shift(context)[..., None, :])
        return (left * valid).contiguous(), (right * valid).contiguous()

    forward = node_factors


def concatenate_scaled_factors(terms: Sequence[tuple[float, tuple[Tensor, Tensor]]]):
    """One factor pair for sum_m lambda_m L_m R_m^T.

    Scale LEFT by lambda and leave RIGHT unchanged, then concatenate the rank
    axis. This supports lambda=0 without sqrt derivatives or a hidden bypass.
    Coefficients are externally selected nonnegative scalars, not parameters.
    The frozen count score is added separately; it is never factor-concatenated.
    """
    if not terms:
        raise ValueError('At least one factor term is required')
    left_parts, right_parts = [], []
    shape = device = None
    for coefficient, factors in terms:
        if (isinstance(coefficient, bool) or not isinstance(coefficient, Real)
                or not math.isfinite(float(coefficient)) or coefficient < 0):
            raise ValueError('Stage coefficients must be finite nonnegative scalars')
        left, right = factors
        if (left.ndim != 4 or min(left.shape) < 1 or right.shape != left.shape
                or left.device != right.device or left.dtype != torch.float32
                or right.dtype != torch.float32):
            raise ValueError('Factors must be matching FP32 [B,L,K,R] arrays')
        if shape is not None and (left.shape[:3] != shape or left.device != device):
            raise ValueError('Factor terms must share candidate layout and device')
        shape, device = left.shape[:3], left.device
        left_parts.append(left * float(coefficient))
        right_parts.append(right)
    return torch.cat(left_parts, -1).contiguous(), torch.cat(right_parts, -1).contiguous()
