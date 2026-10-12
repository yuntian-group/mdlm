"""Exact dense-pair linear-chain reference; no torch-struct or CUDA extension.

Scores are unary[B,L,K] and pair[B,L-1,K_left,K_right]. For length n,
log weight(x) = sum_{i<n} unary[i,x_i] + sum_{i<n-1} pair[i,x_i,x_{i+1}].
Varying supports and observed singleton constraints are represented by -inf.
Lengths count real positions (1..L); padded node/edge marginals are zero and
sampled padded slots are -1. There is no top-k selection or renormalization of
the input law in this module: callers supply their chosen support/scores.

The default log_partition backward returns exact node/edge marginals without
retaining the forward DP autograd graph. Saved tensors are the input scores,
lengths and O(BLK) forward messages; pair gradients still require O(BLK^2).
This is a correctness/training reference, NOT a fused high-speed GPU backend.
Use memory_efficient=False for higher-order derivatives. marginals() uses
ordinary autograd and is itself differentiable (including marginal losses).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable


@dataclass
class ChainMarginals:
    log_partition: torch.Tensor  # [B]
    node: torch.Tensor           # [B,L,K]; padded positions zero
    edge: torch.Tensor | None    # [B,L-1,K_left,K_right]; padded edges zero


def _validate(unary, pair, lengths):
    if unary.ndim != 3 or min(unary.shape) < 1:
        raise ValueError('unary must have nonempty [B,L,K] shape')
    b, length, states = unary.shape
    if pair.shape != (b, length-1, states, states):
        raise ValueError('pair must have [B,L-1,K_left,K_right] shape')
    if unary.dtype not in (torch.float32, torch.float64) or pair.dtype != unary.dtype:
        raise ValueError('unary and pair must share float32 or float64 dtype')
    if pair.device != unary.device:
        raise ValueError('unary and pair must share a device')
    if bool(torch.isnan(unary).any() | torch.isposinf(unary).any()
            | torch.isnan(pair).any() | torch.isposinf(pair).any()):
        raise ValueError('scores may contain finite values and -inf only')
    if lengths is None:
        lengths = torch.full((b,), length, dtype=torch.long, device=unary.device)
    if (not isinstance(lengths, torch.Tensor) or lengths.shape != (b,)
            or lengths.dtype != torch.long or lengths.device != unary.device
            or bool(((lengths < 1) | (lengths > length)).any())):
        raise ValueError('lengths must be int64[B] on the score device, between 1 and L')
    return lengths


def _safe_logsumexp(value, dim, keepdim=False):
    """Return -inf with zero gradient for an entirely unreachable slice."""
    reachable = torch.isfinite(value).any(dim=dim, keepdim=True)
    safe = torch.where(reachable, value, torch.zeros_like(value))
    result = torch.logsumexp(safe, dim=dim, keepdim=True)
    result = torch.where(reachable, result, torch.full_like(result, -torch.inf))
    return result if keepdim else result.squeeze(dim)


def _normalize(value):
    norm = _safe_logsumexp(value, -1, keepdim=True)
    # Keep impossible rows at -inf without forming -inf - -inf. The forward
    # pass subsequently rejects chains with no complete legal path.
    return value - torch.where(torch.isfinite(norm), norm, 0.), norm.squeeze(-1)


def _forward(unary, pair, lengths):
    alpha, scale = _normalize(unary[:, 0])
    messages, logz = [alpha], scale
    for pos in range(1, unary.shape[1]):
        active = lengths > pos
        edge = torch.where(active[:, None, None], pair[:, pos-1], 0.)
        emission = torch.where(active[:, None], unary[:, pos], 0.)
        proposal = emission + _safe_logsumexp(alpha[:, :, None] + edge, -2)
        proposal, scale = _normalize(proposal)
        alpha = torch.where(active[:, None], proposal, alpha)
        logz = logz + torch.where(active, scale, 0.)
        messages.append(alpha)
    if not bool(torch.isfinite(logz).all()):
        raise ValueError('Every active chain must have at least one finite complete path')
    return torch.stack(messages, 1), logz


def _backward(unary, pair, lengths):
    beta = torch.zeros_like(unary[:, -1])
    messages = [beta]
    for pos in range(unary.shape[1]-2, -1, -1):
        active = lengths > pos+1
        edge = torch.where(active[:, None, None], pair[:, pos], 0.)
        emission = torch.where(active[:, None], unary[:, pos+1], 0.)
        proposal = _safe_logsumexp(edge + emission[:, None, :] + beta[:, None, :], -1)
        proposal, _ = _normalize(proposal)
        beta = torch.where(active[:, None], proposal, 0.)
        messages.append(beta)
    return torch.stack(messages[::-1], 1)


def _edge_posterior(unary, pair, lengths, alpha, beta, pos):
    real = lengths > pos+1
    value = (alpha[:, pos, :, None] + pair[:, pos]
             + unary[:, pos+1, None, :] + beta[:, pos+1, None, :])
    flat, _ = _normalize(value.flatten(1))
    return torch.where(real[:, None, None], flat.reshape_as(value).exp(), 0.)


def _posterior(unary, pair, lengths, alpha, beta, *, return_edges):
    active = torch.arange(unary.shape[1], device=unary.device)[None, :] < lengths[:, None]
    node, _ = _normalize(alpha + beta)
    node = torch.where(active[:, :, None], node.exp(), 0.)
    edge_marginals = None
    if return_edges:
        edges = []
        for pos in range(pair.shape[1]):
            edges.append(_edge_posterior(unary, pair, lengths, alpha, beta, pos))
        edge_marginals = torch.stack(edges, 1) if edges else pair.clone()
    return node, edge_marginals


class _MemoryEfficientLogPartition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unary, pair, lengths):
        alpha, logz = _forward(unary, pair, lengths)
        ctx.save_for_backward(unary, pair, lengths, alpha)
        return logz

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_logz):
        unary, pair, lengths, alpha = ctx.saved_tensors
        beta = _backward(unary, pair, lengths)
        grad_unary, grad_pair = None, None
        if ctx.needs_input_grad[0]:
            node, _ = _posterior(unary, pair, lengths, alpha, beta, return_edges=False)
            grad_unary = node * grad_logz[:, None, None]
        if ctx.needs_input_grad[1]:
            # One required pair-gradient allocation, with only O(BK^2) edge
            # scratch; do not stack an additional full edge-marginal tensor.
            grad_pair = torch.empty_like(pair)
            for pos in range(pair.shape[1]):
                posterior = _edge_posterior(unary, pair, lengths, alpha, beta, pos)
                grad_pair[:, pos] = posterior * grad_logz[:, None, None]
        return grad_unary, grad_pair, None


def log_partition(unary, pair, lengths=None, *, memory_efficient=True):
    """Return logZ[B]; default backward is exact first-order marginals."""
    lengths = _validate(unary, pair, lengths)
    if memory_efficient:
        return _MemoryEfficientLogPartition.apply(unary, pair, lengths)
    return _forward(unary, pair, lengths)[1]


def marginals(unary, pair, lengths=None, *, return_edges=True):
    """Differentiable exact node/edge probabilities using normalized FP DP."""
    lengths = _validate(unary, pair, lengths)
    alpha, logz = _forward(unary, pair, lengths)
    beta = _backward(unary, pair, lengths)
    node, edge = _posterior(unary, pair, lengths, alpha, beta, return_edges=return_edges)
    return ChainMarginals(logz, node, edge)


def path_score(unary, pair, states, lengths=None):
    """Score one path[B,L]. Illegal supported-state paths have score -inf.

    Active slots must be 0..K-1; padded slots are ignored (and may be -1).
    """
    lengths = _validate(unary, pair, lengths)
    if states.shape != unary.shape[:2] or states.dtype != torch.long or states.device != unary.device:
        raise ValueError('states must be int64[B,L] on the score device')
    active = torch.arange(unary.shape[1], device=unary.device)[None] < lengths[:, None]
    if bool((((states < 0) | (states >= unary.shape[-1])) & active).any()):
        raise ValueError('Active path slots must be in 0..K-1')
    safe = states.clamp(0, unary.shape[-1]-1)
    score = torch.where(active, unary.gather(-1, safe[..., None]).squeeze(-1), 0.).sum(-1)
    if pair.shape[1]:
        offsets = safe[:, :-1] * unary.shape[-1] + safe[:, 1:]
        selected = pair.flatten(-2).gather(-1, offsets[..., None]).squeeze(-1)
        score = score + torch.where(active[:, 1:], selected, 0.).sum(-1)
    return score


def joint_nll(unary, pair, states, lengths=None, *, reduction='none', memory_efficient=True):
    """Exact logZ minus path score. An excluded target yields +inf, not a
    renormalized/ignored target. Evaluation must not silently add missing gold.
    """
    if reduction not in ('none', 'sum', 'mean'):
        raise ValueError('reduction must be none, sum or mean')
    value = log_partition(unary, pair, lengths, memory_efficient=memory_efficient)
    value = value - path_score(unary, pair, states, lengths)
    return value if reduction == 'none' else getattr(value, reduction)()


@torch.no_grad()
def sample_ffbs(unary, pair, lengths=None, *, num_samples=1, generator=None):
    """Exact joint draws[S,B,L] in support-slot coordinates; padding is -1.

    A single fixed generator controls categorical draws. Backward sampling is
    p(x_i | x_{i+1}, scores), NOT independent sampling from node marginals.
    """
    lengths = _validate(unary, pair, lengths)
    if type(num_samples) is not int or num_samples < 1:
        raise ValueError('num_samples must be a positive integer')
    alpha, _ = _forward(unary, pair, lengths)
    batch, length, states = unary.shape
    result = torch.full((num_samples, batch, length), -1, dtype=torch.long, device=unary.device)
    for pos in range(length-1, -1, -1):
        rows = (lengths > pos).nonzero().flatten()
        if not rows.numel():
            continue
        logits = alpha[rows, pos][None].expand(num_samples, -1, -1)
        if pos < length-1:
            continuation = lengths[rows] > pos+1
            next_slot = result[:, rows, pos+1].clamp_min(0)
            edge = pair[rows, pos][None].expand(num_samples, -1, -1, -1)
            selected = edge.gather(-1, next_slot[:, :, None, None].expand(-1, -1, states, 1)).squeeze(-1)
            logits = logits + torch.where(continuation[None, :, None], selected, 0.)
        probability = logits.softmax(-1)
        drawn = torch.multinomial(probability.reshape(-1, states), 1, generator=generator)
        result[:, rows, pos] = drawn.reshape(num_samples, rows.numel())
    return result


# Short aliases without a second implementation.
logZ = log_partition
ffbs = sample_ffbs
