"""Explicit static count/global inference on unchanged candidate supports.

S[a,b] = count_log_score[a,b] + (coefficient * A[a]) dot B[b].

The frozen global log-score residual is corruption-independent, so its FP32
V-by-V table can be prepared once. This is an optional inference backend,
not a default, a training objective, a candidate filter or a new model. A
context-dependent residual is not supported here. Preparation/compilation must
be reported separately from steady-state decoding time. See STATIC_BACKEND.md.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time

import torch

from . import cuda as additive
from .heads import GlobalAdditiveHead
from . import _dense_inference as dense
from . import _count_inference as count_reference


def prepare_static_count_cache(potential, **budgets):
    """Build an explicitly owned FP32 table; no process-global cache is used.

    Pass the result to sample_static_count/static_count_marginals. Treat its
    tensors as immutable. Delete the result after sampling to release memory.
    The caller times this one-time preparation separately from decoding.
    """
    return dense.build_dense_count_table(potential, **budgets)


def sample_static_count(candidates, potential, cache, generator=None, *, segment_batch_size=32):
    """Exact retained-support joint sampling from an explicitly prepared table."""
    if cache is None:
        raise ValueError('Prepare and pass an explicit static count cache')
    return dense.sample_dense_count(candidates, potential, generator, table=cache,
                                    segment_batch_size=segment_batch_size)


def static_count_marginals(candidates, potential, cache, *, segment_batch_size=32):
    """Singleton probabilities of that same chain, normalized locally."""
    if cache is None:
        raise ValueError('Prepare and pass an explicit static count cache')
    logp = dense.infer_dense_count(candidates, potential, table=cache,
        segment_batch_size=segment_batch_size, sampling='log_marginals')
    return torch.softmax(logp, -1)


def _head_signature(head):
    return tuple((id(t), t._version, tuple(t.shape), str(t.dtype), str(t.device), t.requires_grad)
                 for t in (head.left_embedding.weight, head.right_embedding.weight))


def _validate_head(potential, head, coefficient):
    if type(coefficient) not in (int, float) or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Global coefficient must be finite and nonnegative')
    if not isinstance(head, GlobalAdditiveHead):
        raise ValueError('Only a frozen GlobalAdditiveHead is supported; contextual scores are not static')
    left, right = head.left_embedding.weight, head.right_embedding.weight
    if (left.shape != right.shape or left.shape != (potential.vocab_size, head.rank)
            or not 1 <= head.rank <= 128 or left.dtype != torch.float32
            or right.dtype != torch.float32 or left.device != potential.left.device
            or right.device != left.device or left.requires_grad or right.requires_grad):
        raise ValueError('Frozen FP32 global weights of matching vocabulary/device and rank<=128 required')
    if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
        raise ValueError('Global weights must be finite')
    return left, right


@contextmanager
def _ieee_matmul(device):
    # Preserve the caller process setting; restore the setting even if building fails.
    # CPU has no TF32 setting. Do not silently inherit a lower precision mode.
    if device.type != 'cuda':
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@dataclass
class StaticGlobalCache:
    table: dense.DenseCountTable | None
    potential: object
    head: GlobalAdditiveHead
    coefficient: float
    potential_signature: tuple
    head_signature: tuple
    diagnostics: dict
    closed: bool = False

    def assert_current(self, potential):
        if self.closed:
            raise RuntimeError('Static global cache has been closed')
        if (potential is not self.potential or dense._signature(potential) != self.potential_signature
                or _head_signature(self.head) != self.head_signature):
            raise ValueError('Frozen count/global source changed; explicitly rebuild the cache')

    def prepared_factors(self, packet):
        """Keep the common compact candidate/boundary interface; no [B,L,K,R]."""
        self.assert_current(self.potential)
        if self.table is None:
            raise ValueError('Zero residual must use unchanged count-only inference')
        if packet.unary.device != self.table.log_scores.device:
            raise ValueError('Candidate and static table devices must match')
        return dense.DenseFactors(packet.candidate_ids, self.table)

    def close(self):
        """Release the large table before external scoring or another model."""
        self.table = None
        self.closed = True


@torch.no_grad()
def prepare_static_global_cache(potential, head, coefficient, *,
                                max_table_bytes=12 * 1024**3,
                                max_temporary_bytes=128 * 1024**2,
                                reserve_bytes=2 * 1024**3):
    """Explicit setup; no global cache, files, weights saved, or hidden rebuilds.

    Temporary budget is shared sequentially by count construction and the
    learned row product. Count CSR-to-key conversion has its own conservative
    O(E) estimate in diagnostics. Matmul-library workspace is not bounded by
    this tensor budget; reserve_bytes is required for that and inference.
    """
    if any(type(x) is not int or x < 0 for x in
           (max_table_bytes, max_temporary_bytes, reserve_bytes)):
        raise ValueError('Memory budgets must be nonnegative integer bytes')
    left, right = _validate_head(potential, head, coefficient)
    source_signature, head_signature = dense._signature(potential), _head_signature(head)
    coefficient = float(coefficient)
    inactive = coefficient == 0 or not bool(left.ne(0).any() & right.ne(0).any())
    if inactive:
        return StaticGlobalCache(None, potential, head, coefficient, source_signature,
            head_signature, {'backend': 'unchanged_count_zero_global', 'table_bytes': 0,
            'global_coefficient': coefficient, 'global_rank': head.rank,
            'persistent_files_written': False, 'zero_dispatch_bitwise': True})
    vocab, rank = left.shape
    # Scaled [rows,R], FP32 product [rows,V], and conservatively two boolean
    # [rows,V] finite-check buffers. Reduction/BLAS workspace uses reserve.
    one_row_bytes = 4 * rank + 6 * vocab
    if max_temporary_bytes < one_row_bytes:
        raise MemoryError('Temporary budget cannot hold one learned product row')
    rows_per_chunk = min(vocab, max_temporary_bytes // one_row_bytes)
    device = left.device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    # Always build an unshared table: never mutate an existing count cache.
    table = dense.build_dense_count_table(potential, max_table_bytes=max_table_bytes,
        max_temporary_bytes=max_temporary_bytes, reserve_bytes=reserve_bytes)
    with _ieee_matmul(device), torch.autocast(device.type, enabled=False):
        for start in range(0, vocab, rows_per_chunk):
            stop = min(vocab, start + rows_per_chunk)
            scaled_left = left[start:stop] * coefficient
            residual = torch.mm(scaled_left, right.T)
            if not bool(torch.isfinite(residual).all()):
                raise ValueError('Nonfinite learned product; refusing invalid static table')
            table.log_scores[start:stop].add_(residual)
            if not bool(torch.isfinite(table.log_scores[start:stop]).all()):
                raise ValueError('Nonfinite combined static pair score')
            # Explicit release keeps simultaneous chunk temporaries bounded.
            del scaled_left, residual
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    if dense._signature(potential) != source_signature or _head_signature(head) != head_signature:
        raise ValueError('Source mutated while building the static table')
    diagnostics = dict(table.diagnostics)
    diagnostics.update(backend='experimental_static_count_plus_global_log_table',
        global_coefficient=coefficient, global_rank=rank,
        learned_rows_per_build_chunk=rows_per_chunk,
        learned_chunk_tensor_bytes=rows_per_chunk * one_row_bytes,
        learned_temporary_scope='scaled left rows plus product rows plus two finite-check boolean rows; excludes reduction/BLAS workspace',
        score_convention='FP32_count_log + FP32_matmul(coefficient_times_FP32_A,FP32_B)',
        matmul_precision='IEEE FP32; TF32 disabled',
        build_seconds_including_device_synchronization=time.perf_counter() - started,
        build_timing_scope='one-time setup, excluded from per-sequence timing but reported separately',
        gpu_law_gate_passed=False, speedup_measured=False,
        same_support=True, no_extra_backbone_pass=True,
        persistent_files_written=False, zero_dispatch_bitwise=False)
    table.diagnostics = diagnostics
    return StaticGlobalCache(table, potential, head, coefficient, source_signature,
                             head_signature, diagnostics)


@torch.no_grad()
def infer_static_global(candidates, potential, cache, generator=None, *,
                        segment_batch_size=32, sampling='joint'):
    cache.assert_current(potential)
    if sampling not in ('joint', 'log_marginals'):
        raise ValueError('sampling must be joint or log_marginals')
    if cache.table is None:
        # Do not send an exactly zero global head through a different recurrence.
        return additive.infer_additive_count(candidates, potential, generator=generator,
            segment_batch_size=segment_batch_size, sampling=sampling)
    if dense.triton is None or not candidates.log_unary.is_cuda or torch.version.hip is not None:
        raise ValueError('Static global FFBS requires NVIDIA CUDA and Triton')
    if candidates.vocab_size != potential.vocab_size:
        raise ValueError('Count and candidate vocabularies must agree')
    packet = count_reference.prepare_packet(candidates)
    factors = cache.prepared_factors(packet)
    if sampling == 'joint':
        result = packet.unary.argmax(-1)
    else:
        result = packet.unary.new_full(packet.unary.shape, -torch.inf)
        result.scatter_(-1, packet.unary.argmax(-1, keepdim=True), 0.)
    for unary, rows, starts, lengths, positions, valid in count_reference._chunks(
            packet, factors, segment_batch_size):
        alpha = dense.dense_messages(unary, factors, rows, starts, lengths)
        if sampling == 'joint':
            noise = torch.empty_like(alpha, dtype=torch.float64).exponential_(generator=generator).log_()
            values = dense.dense_sample_with_noise(alpha, factors, rows, starts, lengths, noise)
        else:
            beta = dense.dense_messages(unary, factors, rows, starts, lengths, backward=True)
            values = torch.log_softmax(alpha + beta, -1)
        batches = rows[:, None].expand_as(positions)
        result[batches[valid], positions[valid]] = values[valid]
    return result


def sample_static_global(candidates, potential, cache, generator=None, *, segment_batch_size=32):
    slots = infer_static_global(candidates, potential, cache, generator,
                               segment_batch_size=segment_batch_size)
    return candidates.candidate_ids.gather(-1, slots[..., None]).squeeze(-1)


def static_global_marginals(candidates, potential, cache, *, segment_batch_size=32):
    return infer_static_global(candidates, potential, cache,
        segment_batch_size=segment_batch_size, sampling='log_marginals').exp()
