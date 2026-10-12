"""Optional exact-support NVIDIA CUDA/Triton additive inference.

Supply Candidates and SparseCountPotential from this package. Learned factors
already include external coefficients. No encoder calls, support changes,
training, gold injection, or automatic CPU fallback are performed. This module
is explicit opt-in; importing additive_chain does not import Triton.
"""
import torch

from . import _additive_kernels as af
from . import _count_inference as count_reference


def clear_count_cache():
    """Release the shared immutable count cache before scorer/model handoff."""
    count_reference.clear_count_cache()


def _check_factors(packet, left, right):
    if left is None and right is None:
        return None
    if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
            or left.ndim != 4 or left.shape[:3] != packet.unary.shape
            or right.shape != left.shape or not 1 <= left.shape[-1] <= 128
            or left.dtype != torch.float32 or right.dtype != torch.float32
            or left.device != packet.unary.device or right.device != left.device):
        raise ValueError('Learned factors must be FP32 [B,L,K,R], 1<=R<=128, on the candidate device')
    # One host synchronization covers both input validity and the exact nested
    # count-only case. At zero initialization (or external coefficient zero),
    # avoid a dot-product kernel whose layout can change FP32 reduction order
    # even though its mathematical residual is identically zero.
    finite, active = torch.stack((torch.isfinite(left).all() & torch.isfinite(right).all(),
                                  left.ne(0).any() & right.ne(0).any())).tolist()
    if not finite:
        raise ValueError('Learned factors must be finite')
    if not active:
        return None
    valid = packet.candidate_ids.ge(0)[..., None]
    return (left * valid).contiguous(), (right * valid).contiguous()


def prepare_additive_factors(packet, potential, left=None, right=None):
    learned = _check_factors(packet, left, right)
    base = count_reference._factors(packet, potential)
    if learned is None:
        left, right, rank, kind = base.left, base.right, 1, 1
    else:
        left, right = learned
        rank, kind = left.shape[-1], 2
    return af.PairFactors(left, right, packet.candidate_ids, base.keys, base.values,
                          base.left_bg, base.right_bg, kind, base.vocab, rank)


@torch.no_grad()
def infer_additive_count(candidates, potential, left=None, right=None, generator=None,
                         *, segment_batch_size=32, sampling='joint'):
    if (af.triton is None or not candidates.log_unary.is_cuda
            or torch.version.hip is not None):
        raise ValueError('Additive fused inference requires NVIDIA CUDA and Triton')
    if sampling not in ('joint', 'log_marginals'):
        raise ValueError('sampling must be joint or log_marginals')
    packet = count_reference.prepare_packet(candidates)
    if potential.vocab_size != candidates.vocab_size:
        raise ValueError('Count and candidate vocabularies must agree')
    factors = prepare_additive_factors(packet, potential, left, right)
    if sampling == 'joint':
        result = packet.unary.argmax(-1)
    else:
        result = packet.unary.new_full(packet.unary.shape, -torch.inf)
        result.scatter_(-1, packet.unary.argmax(-1, keepdim=True), 0.)
    for unary, rows, starts, lengths, positions, valid in count_reference._chunks(
            packet, factors, segment_batch_size):
        alpha = af._messages(unary, factors, rows, starts, lengths)
        if sampling == 'joint':
            noise = torch.empty_like(alpha, dtype=torch.float64).exponential_(generator=generator).log_()
            values = af._sample_with_noise(alpha, factors, rows, starts, lengths, noise)
        else:
            beta = af._messages(unary, factors, rows, starts, lengths, backward=True)
            # Normalize every node locally to avoid accumulating a full-chain
            # logZ subtraction error; alpha/beta represent the same exact law.
            # log_softmax subtracts the maximum before normalizing. Computing
            # x-logsumexp(x) directly loses mass accuracy when an entire long
            # chain carries a large common FP32 offset.
            values = torch.log_softmax(alpha + beta, -1)
        batches = rows[:, None].expand_as(positions)
        result[batches[valid], positions[valid]] = values[valid]
    return result


def sample_additive_count(candidates, potential, left=None, right=None, generator=None,
                          *, segment_batch_size=32):
    slots = infer_additive_count(candidates, potential, left, right, generator,
                                 segment_batch_size=segment_batch_size)
    return candidates.candidate_ids.gather(-1, slots[..., None]).squeeze(-1)


def additive_count_marginals(candidates, potential, left=None, right=None,
                             *, segment_batch_size=32):
    return infer_additive_count(candidates, potential, left, right,
        segment_batch_size=segment_batch_size, sampling='log_marginals').exp()


def dense_pair_reference(candidates, potential, left=None, right=None, *, max_cells=4_000_000):
    """Small CPU/CUDA score oracle only; refuses large all-edge allocations.

    Returns [B,L-1,K_source,K_destination] FP32, with neutral padding. The score
    conversion matches the frozen Nina count hash exactly, before adding dots.
    Invalid candidates remain forbidden by the original -inf unary scores.
    """
    packet = count_reference.prepare_packet(candidates)
    learned = _check_factors(packet, left, right)
    b, length, k = packet.unary.shape
    if b * max(0, length-1) * k * k > max_cells:
        raise ValueError('Dense oracle budget exceeded; use tiled production kernels')
    keys, values, left_bg, right_bg = count_reference.count_log_components(potential)
    a, c = packet.candidate_ids[:, :-1, :, None], packet.candidate_ids[:, 1:, None, :]
    valid = a.ge(0) & c.ge(0)
    sa, sc = a.clamp_min(0), c.clamp_min(0)
    score = left_bg[sa] + right_bg[sc]
    if keys.numel():
        query = (sa * potential.vocab_size + sc).contiguous()
        index = torch.searchsorted(keys, query).clamp_max(keys.numel()-1)
        score = torch.where(keys[index] == query, values[index], score)
    score = torch.where(valid, score.float(), 0.)
    if learned is not None:
        l, r = learned
        score = score + torch.einsum('blar,blcr->blac', l[:, :-1], r[:, 1:])
    return score

