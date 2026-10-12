"""Optional cached count/global base plus dynamic contextual log scores.

The cache is independent of the corrupted input; the caller supplies fresh
contextual endpoint factors from the one masked-input backbone pass. This
module never builds a cache inside the denoising loop, changes candidates,
injects target tokens, or writes weights. Coefficients are already included
in the fixed table and contextual left factors, exactly as in the fused law.

Inference only; no autograd or automatic CPU fallback. The portable backend
requires its own CUDA and full-model parity checks before benchmark claims.
See STATIC_CONTEXTUAL_BACKEND.md for the API, provenance, and validation scope.
"""
import torch

from . import _additive_kernels as kernels
from . import cuda as additive
from . import _dense_inference as dense
from . import _count_inference as reference
from . import static


def prepare_contextual_factors(packet, potential, cache, left, right):
    cache.assert_current(potential)
    learned = additive._check_factors(packet, left, right)
    if learned is None:
        return None
    table = cache.table
    if table is None:
        raise ValueError('Active contextual residual requires a prepared count/global table')
    if (table.log_scores.shape != (potential.vocab_size, potential.vocab_size)
            or table.log_scores.dtype != torch.float32
            or table.log_scores.device != packet.unary.device
            or not table.log_scores.is_contiguous()):
        raise ValueError('Contiguous matching FP32 static table required')
    # Dummy fields keep the common kernel-call interface. kind3 never reads
    # hash keys or background factors; values contains the dense fixed table.
    dummy_keys = packet.candidate_ids.new_empty((0,))
    dummy_bg = packet.unary.new_empty((0,))
    l, r = learned
    return kernels.PairFactors(l, r, packet.candidate_ids, dummy_keys,
        table.log_scores, dummy_bg, dummy_bg, 3, potential.vocab_size, l.shape[-1])


@torch.no_grad()
def infer_static_contextual(candidates, potential, cache, left=None, right=None,
                            generator=None, *, segment_batch_size=32,
                            sampling='joint'):
    if sampling not in ('joint', 'log_marginals'):
        raise ValueError('sampling must be joint or log_marginals')
    if candidates.vocab_size != potential.vocab_size:
        raise ValueError('Candidate/count vocabularies must agree')
    packet = reference.prepare_packet(candidates)
    cache.assert_current(potential)
    if cache.table is None:
        # A zero global term is a valid predecessor, not an unsupported model.
        # No implicit V^2 allocation here: use the existing count+local path.
        # A caller wanting cached speed can explicitly prepare a count table.
        additive._check_factors(packet, left, right)
        return additive.infer_additive_count(candidates, potential, left, right,
            generator, segment_batch_size=segment_batch_size, sampling=sampling)
    factors = prepare_contextual_factors(packet, potential, cache, left, right)
    if factors is None:
        # Identically zero local score uses its exact predecessor implementation
        # and consumes precisely the predecessor RNG stream.
        return static.infer_static_global(candidates, potential, cache, generator,
            segment_batch_size=segment_batch_size, sampling=sampling)
    if kernels.triton is None or not packet.unary.is_cuda or torch.version.hip is not None:
        raise ValueError('Cached contextual inference requires NVIDIA CUDA and Triton')
    if sampling == 'joint':
        result = packet.unary.argmax(-1)
    else:
        result = packet.unary.new_full(packet.unary.shape, -torch.inf)
        result.scatter_(-1, packet.unary.argmax(-1, keepdim=True), 0.)
    for unary, rows, starts, lengths, positions, valid in reference._chunks(
            packet, factors, segment_batch_size):
        alpha = kernels._messages(unary, factors, rows, starts, lengths)
        if sampling == 'joint':
            noise = torch.empty_like(alpha, dtype=torch.float64).exponential_(generator=generator).log_()
            values = kernels._sample_with_noise(alpha, factors, rows, starts, lengths, noise)
        else:
            beta = kernels._messages(unary, factors, rows, starts, lengths, backward=True)
            values = torch.log_softmax(alpha + beta, -1)
        batches = rows[:, None].expand_as(positions)
        result[batches[valid], positions[valid]] = values[valid]
    return result


def sample_static_contextual(candidates, potential, cache, left=None, right=None,
                             generator=None, *, segment_batch_size=32):
    slots = infer_static_contextual(candidates, potential, cache, left, right,
        generator, segment_batch_size=segment_batch_size)
    return candidates.candidate_ids.gather(-1, slots[..., None]).squeeze(-1)


def static_contextual_marginals(candidates, potential, cache, left=None, right=None,
                                *, segment_batch_size=32):
    return infer_static_contextual(candidates, potential, cache, left, right,
        segment_batch_size=segment_batch_size, sampling='log_marginals').exp()


def dense_pair_reference(candidates, potential, cache, left=None, right=None,
                         *, max_cells=4_000_000):
    """Bounded independent score oracle, not a production inference fallback."""
    if candidates.vocab_size != potential.vocab_size:
        raise ValueError('Candidate/count vocabularies must agree')
    packet = reference.prepare_packet(candidates)
    batch, length, cap = packet.unary.shape
    if batch * max(0, length-1) * cap * cap > max_cells:
        raise ValueError('Dense oracle budget exceeded')
    cache.assert_current(potential)
    if cache.table is None:
        return additive.dense_pair_reference(candidates, potential, left, right, max_cells=max_cells)
    factors = prepare_contextual_factors(packet, potential, cache, left, right)
    pair = dense.dense_pair_scores(cache.table, packet.candidate_ids[:, :-1, :, None],
                                  packet.candidate_ids[:, 1:, None, :])
    if factors is not None:
        pair = pair + torch.einsum('blar,blcr->blac', factors.left[:, :-1], factors.right[:, 1:])
    return pair
