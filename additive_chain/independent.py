"""Unary-only capacity controls for the additive chain ladder.

These heads add scores at individual positions; they introduce no pair term.
The inherited count/global chain, if any, is unchanged. Thus an independent
residual above a count model is NOT itself a fully factorized joint model.
All hidden states/optional token embeddings come from the same frozen masked
backbone pass. Candidate IDs remain the shared baseline support; this module
does not select candidates or inject gold.

This module only transforms score tensors; it does not load models, choose
candidate support, or run an experiment.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


def _dimension(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def parameter_count(vocab_size, hidden_size, width, *, embedding_size=None):
    """Count trainable tensor elements, excluding frozen embedding buffers."""
    for name, value in [('vocab_size', vocab_size), ('hidden_size', hidden_size), ('width', width)]:
        _dimension(value, name)
    # Input projection H*w + bias w. Output either V*w+bias V or E*w.
    if embedding_size is None:
        return (hidden_size + 1 + vocab_size) * width + vocab_size
    _dimension(embedding_size, 'embedding_size')
    return (hidden_size + 1 + embedding_size) * width


def choose_width(target_parameters, vocab_size, hidden_size, *, embedding_size=None):
    """Nearest usable width; expose exact mismatch rather than claim equality."""
    _dimension(target_parameters, 'target_parameters')
    base = parameter_count(vocab_size, hidden_size, 1, embedding_size=embedding_size)
    slope = parameter_count(vocab_size, hidden_size, 2, embedding_size=embedding_size) - base
    real = 1 + (target_parameters - base) / slope
    choices = {max(1, math.floor(real)), max(1, math.ceil(real))}
    width = min(choices, key=lambda w: (abs(parameter_count(vocab_size, hidden_size, w,
                         embedding_size=embedding_size) - target_parameters), w))
    actual = parameter_count(vocab_size, hidden_size, width, embedding_size=embedding_size)
    return dict(width=width, target_parameters=target_parameters, actual_parameters=actual,
                relative_parameter_difference=(actual-target_parameters)/target_parameters,
                architecture='full_vocabulary_output' if embedding_size is None else 'frozen_embedding_output')


class IndependentAdditiveHead(nn.Module):
    """Positionwise signed residual, zero initialized at the output projection.

    Dense variant: score_i(v) = W_v SiLU(P LN(h_i)+b) + c_v.
    Frozen-embedding variant: score_i(v) = <E_v Q, SiLU(P LN(h_i)+b)>.
    Layer normalization has no learned affine parameters. The embedding table
    is an immutable detached buffer; all active capacity is counted above.
    """
    def __init__(self, vocab_size, hidden_size, width, *, frozen_token_embeddings=None):
        super().__init__()
        self.vocab_size = _dimension(vocab_size, 'vocab_size')
        self.hidden_size = _dimension(hidden_size, 'hidden_size')
        self.width = _dimension(width, 'width')
        self.input = nn.Linear(hidden_size, width)
        self.embedding_size = None
        if frozen_token_embeddings is None:
            self.output = nn.Linear(width, vocab_size)
            self.register_buffer('token_embeddings', None)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            embedding = frozen_token_embeddings
            if (embedding.ndim != 2 or embedding.shape[0] != vocab_size or embedding.shape[1] < 1
                    or not embedding.is_floating_point() or not bool(torch.isfinite(embedding).all())):
                raise ValueError('Expected finite floating frozen token embeddings[V,E]')
            self.embedding_size = embedding.shape[1]
            self.register_buffer('token_embeddings', embedding.detach().float().clone())
            self.output = nn.Linear(self.embedding_size, width, bias=False)
            nn.init.zeros_(self.output.weight)
            self.to(embedding.device)

    def _features(self, hidden):
        if (hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size
                or not hidden.is_floating_point() or hidden.device != self.input.weight.device
                or min(hidden.shape) < 1 or self.input.weight.dtype != torch.float32):
            raise ValueError('Expected floating hidden[B,L,H] and FP32 control on same device')
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError('Nonfinite frozen hidden state')
        return F.silu(self.input(F.layer_norm(hidden.detach().float(), (self.hidden_size,))))

    def forward(self, candidate_ids, hidden):
        if (candidate_ids.ndim != 3 or candidate_ids.dtype != torch.long
                or candidate_ids.shape[:2] != hidden.shape[:2]
                or candidate_ids.device != hidden.device or min(candidate_ids.shape) < 1):
            raise ValueError('Expected int64 candidate IDs[B,L,K] matching hidden[B,L,H]')
        if bool(((candidate_ids < -1) | (candidate_ids > self.vocab_size)).any()):
            raise ValueError('Candidate must be a token, -1, or V padding')
        valid = candidate_ids.ge(0) & candidate_ids.lt(self.vocab_size)
        safe = candidate_ids.clamp(0, self.vocab_size - 1)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            features = self._features(hidden)
            if self.token_embeddings is None:
                table, bias = self.output.weight, self.output.bias
                values = (table[safe] * features[..., None, :]).sum(-1) + bias[safe]
            else:
                # Project before gathering to avoid a [B,L,K,E] retained tensor.
                table = self.output(self.token_embeddings)
                values = (table[safe] * features[..., None, :]).sum(-1)
        return torch.where(valid, values, 0.)

    def full_scores(self, hidden):
        """Reference/all-vocabulary scores, not a support-selection operation."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            features = self._features(hidden)
            if self.token_embeddings is None:
                return self.output(features)
            return features @ self.output(self.token_embeddings).T

    def active_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def add_independent_residual(log_unary, candidate_ids, hidden, head, strength=1.):
    if isinstance(strength, bool) or not isinstance(strength, Real) or not math.isfinite(strength) or strength < 0:
        raise ValueError('Strength must be a finite nonnegative real scalar')
    if log_unary.shape != candidate_ids.shape or not log_unary.is_floating_point():
        raise ValueError('Unaries must match candidate IDs[B,L,K]')
    if strength == 0:
        return log_unary
    return log_unary + float(strength) * head(candidate_ids, hidden).to(log_unary.dtype)
