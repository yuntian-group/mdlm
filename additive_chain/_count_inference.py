"""Segmented count inference on exact retained token support.

SparseCountPotential is an immutable sparse-plus-separable potential. Visible
singleton supports separate latent spans; both boundary edges are retained.
No candidate construction, tail aggregation, or full-vocabulary fallback.
"""
from dataclasses import dataclass
import math

import torch

from . import _count_kernels as nf


# Cache one immutable count potential. Retaining the owner prevents object-id
# reuse; mutation is rejected instead of silently using a stale hash table.
_TABLES = {}


@dataclass
class Packet:
    candidate_ids: torch.Tensor
    unary: torch.Tensor
    masked: torch.Tensor


def count_log_components(potential):
    """CPU-testable exact score conversion; no count normalization is changed."""
    from .counts import SparseCountPotential
    potential = SparseCountPotential.from_reference(potential)
    if not math.isfinite(potential.log_scale):
        raise ValueError('finite shared log scale required')
    left, right = potential.left, potential.right
    if bool((left <= 0).any() | (right <= 0).any()):
        raise ValueError('positive count backoff factors required')
    keys = potential.pair_keys
    rows, columns = keys // potential.vocab_size, keys % potential.vocab_size
    weights = left[rows] * right[columns] + potential.row_values
    if bool((weights <= 0).any() | ~torch.isfinite(weights).all()):
        raise ValueError('positive finite observed count potentials required')
    return keys, weights.double().log(), left.double().log(), right.double().log()


def prepare_packet(candidates):
    """Only recode padding; no probabilities are discarded or pooled."""
    ids, unary = candidates.candidate_ids, candidates.log_unary
    if (ids.ndim != 3 or min(ids.shape) < 1 or unary.shape != ids.shape or ids.dtype != torch.long
            or not unary.is_floating_point() or ids.device != unary.device
            or type(candidates.vocab_size) is not int or candidates.vocab_size < 1
            or not 1 <= ids.shape[-1] <= 2048):
        raise ValueError('expected [B,L,K] candidates with 1 <= K <= 2048')
    finite = torch.isfinite(unary)
    if bool(torch.isnan(unary).any() | torch.isposinf(unary).any()
            | (finite != ((ids >= 0) & (ids < candidates.vocab_size))).any()
            | finite.sum(-1).eq(0).any()):
        raise ValueError('finite scores must correspond exactly to valid IDs')
    return Packet(ids.masked_fill(~finite, -1).contiguous(), unary.float().contiguous(),
                  finite.sum(-1).gt(1))


def _factors(packet, potential):
    from .counts import SparseCountPotential
    if not isinstance(potential, SparseCountPotential):
        raise ValueError('Use SparseCountPotential.from_reference once before CUDA inference')
    if potential.left.device != packet.unary.device:
        raise ValueError('Count potential and candidates must share the CUDA device')
    signature = potential.signature()
    cached = _TABLES.get(id(potential))
    if cached is not None and cached[1] != signature:
        raise ValueError('Count potential mutated after caching; construct a new immutable potential')
    if cached is None:
        keys, values, left_bg, right_bg = count_log_components(potential)
        if not keys.is_cuda:
            raise ValueError('CUDA count potential required')
        hash_keys, hash_values = nf._build_count_hash(keys, values.float(), potential.vocab_size)
        _TABLES.clear()
        cached = (potential, signature, hash_keys, hash_values, left_bg, right_bg)
        _TABLES[id(potential)] = cached
    _, _, keys, values, left_bg, right_bg = cached
    empty = packet.unary.new_empty(0)
    return nf.PairFactors(empty, empty, packet.candidate_ids, keys, values,
                          left_bg, right_bg, 1, potential.vocab_size, 1)


def clear_count_cache():
    _TABLES.clear()


def prepare_count_cache(potential):
    """Explicit preparation for warmup accounting; returns cache memory bytes."""
    device = potential.left.device
    ids = torch.zeros((1, 1, 1), dtype=torch.long, device=device)
    packet = Packet(ids, torch.zeros_like(ids, dtype=torch.float32), ids.bool())
    factors = _factors(packet, potential)
    return sum(t.numel() * t.element_size() for t in
               (factors.keys, factors.values, factors.left_bg, factors.right_bg))


def _chunks(packet, factors, segment_batch_size):
    """Condition on singleton separators and batch contiguous latent runs."""
    if type(segment_batch_size) is not int or segment_batch_size < 1:
        raise ValueError('positive segment batch size required')
    unary, masked = packet.unary, packet.masked
    batch, length, states = unary.shape
    clamps = unary.argmax(-1)
    starts, ends = masked.clone(), masked.clone()
    starts[:, 1:] &= ~masked[:, :-1]
    ends[:, :-1] &= ~masked[:, 1:]
    rows, start = starts.nonzero(as_tuple=True)
    end = ends.nonzero(as_tuple=True)[1]
    lengths = end - start + 1
    for offset in range(0, rows.numel(), segment_batch_size):
        rb = rows[offset:offset + segment_batch_size]
        rs = start[offset:offset + segment_batch_size]
        rl = lengths[offset:offset + segment_batch_size]
        width = int(rl.max())
        positions = rs[:, None] + torch.arange(width, device=unary.device)[None, :]
        valid = positions < (rs + rl)[:, None]
        packed = unary[rb[:, None], positions.clamp_max(length - 1)].clone()
        has_left = rs > 0
        if has_left.any():
            br, bp = rb[has_left], rs[has_left] - 1
            packed[has_left, 0] += factors.boundary(br, bp, clamps[br, bp], left_boundary=True)
        has_right = rs + rl < length
        if has_right.any():
            br, bp = rb[has_right], (rs + rl - 1)[has_right]
            packed[has_right, rl[has_right] - 1] += factors.boundary(
                br, bp, clamps[br, bp + 1], left_boundary=False)
        dummy = unary.new_full((states,), -torch.inf)
        dummy[0] = 0.
        packed = torch.where(valid[..., None], packed, dummy).contiguous()
        yield packed, rb.contiguous(), rs.contiguous(), rl.contiguous(), positions, valid


@torch.no_grad()
def infer_nina_count(candidates, potential, generator=None, *, segment_batch_size=32,
                     sampling='joint'):
    if nf.triton is None or not candidates.log_unary.is_cuda or torch.version.hip is not None:
        raise ValueError('NVIDIA CUDA and Triton required; no unmeasured CPU fallback')
    if sampling not in ('joint', 'log_marginals'):
        raise ValueError('sampling must be joint or log_marginals')
    packet = prepare_packet(candidates)
    factors = _factors(packet, potential)
    if sampling == 'joint':
        result = packet.unary.argmax(-1)
    else:
        result = packet.unary.new_full(packet.unary.shape, -torch.inf)
        result.scatter_(-1, packet.unary.argmax(-1, keepdim=True), 0.)
    for unary, rows, starts, lengths, positions, valid in _chunks(packet, factors, segment_batch_size):
        alpha = nf._messages(unary, factors, rows, starts, lengths)
        if sampling == 'joint':
            noise = torch.empty_like(alpha, dtype=torch.float64).exponential_(generator=generator).log_()
            values = nf._sample_with_noise(alpha, factors, rows, starts, lengths, noise)
        else:
            beta = nf._messages(unary, factors, rows, starts, lengths, backward=True)
            values = alpha + beta - torch.logsumexp(alpha[:, -1], -1)[:, None, None]
        batches = rows[:, None].expand_as(positions)
        result[batches[valid], positions[valid]] = values[valid]
    return result


@torch.no_grad()
def sample_nina_count(candidates, potential, generator=None, *, segment_batch_size=32):
    """Return token IDs (not candidate-slot IDs), matching sample_compact_count."""
    slots = infer_nina_count(candidates, potential, generator,
                            segment_batch_size=segment_batch_size, sampling='joint')
    return candidates.candidate_ids.gather(-1, slots[..., None]).squeeze(-1)


@torch.no_grad()
def nina_count_marginals(candidates, potential, *, segment_batch_size=32):
    return infer_nina_count(candidates, potential, segment_batch_size=segment_batch_size,
                            sampling='log_marginals').exp()

