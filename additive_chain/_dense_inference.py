# Pair/message/sampling portions derived from Nina Zhang, Apache-2.0.
# yuntian-group/mdlm commit 18175679c93d7792936e1497b7c5be72a765f2de.
# See STATIC_BACKEND.md and CUDA_BACKEND.md for attribution and validation scope.
"""Optional dense static count lookup and segmented FFBS.

Table entries omit only the globally shared edge log_scale. Observed pairs use
log(left[a]*right[b] + correction[a,b]); unobserved pairs use the separable
log background. The FP32 table costs 4*V*V bytes and is built explicitly under
memory budgets. No file, model, data or checkpoint loading is performed.

The table lookup replaces hash probes; the log-sum-exp recurrence, retained
candidate support, visible boundaries and FP64 sampling noise are unchanged.
Actual target-device numerical and trajectory validation remains necessary.
"""
from dataclasses import dataclass
import math

import torch

from . import _count_inference as reference

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

_TABLES = {}


@dataclass
class DenseCountTable:
    log_scores: torch.Tensor
    omitted_log_scale: float
    diagnostics: dict
    owner: object
    source_signature: tuple

    @property
    def vocab_size(self):
        return self.log_scores.shape[0]


def _signature(potential):
    # Compatibility inputs may omit pair_keys; the package derives
    # them from CSR. Bind the actual CSR identity/version, not the identity of
    # a newly derived keys tensor (which would invalidate the cache each call).
    tensors = [potential.left, potential.right,
               potential.sparse if hasattr(potential, 'sparse') else potential.row_values]
    keys = getattr(potential, 'pair_keys', None)
    if keys is not None:
        tensors.append(keys)
    return tuple((id(t), t._version, str(t.device)) for t in tensors) + (potential.log_scale,)


@torch.no_grad()
def build_dense_count_table(potential, *, max_table_bytes=12 * 1024**3,
                            max_temporary_bytes=128 * 1024**2,
                            reserve_bytes=2 * 1024**3):
    """Prepare once during explicitly excluded setup/warmup, never per step.

    max_temporary_bytes bounds table-fill chunks, not the O(E) CSR-to-key
    conversion when a raw potential has no cached pair_keys. Its conservative
    24*E-byte peak estimate is reported separately; available GPU memory is
    checked after conversion, before allocating the dense table.
    """
    from .counts import SparseCountPotential
    synthesized_pair_keys = getattr(potential, 'pair_keys', None) is None
    p = SparseCountPotential.from_reference(potential)
    vocab = p.vocab_size
    required = vocab * vocab * 4
    if any(type(x) is not int or x < 0 for x in
           (max_table_bytes, max_temporary_bytes, reserve_bytes)):
        raise ValueError('Memory budgets must be nonnegative integer bytes')
    if required > max_table_bytes:
        raise MemoryError(f'Dense count table requires {required} bytes; budget {max_table_bytes}')
    if max_temporary_bytes < vocab * 12:
        raise MemoryError('Temporary budget must accommodate one FP64 row and FP32 conversion')
    if p.left.dtype != torch.float64 or p.right.dtype != torch.float64:
        raise ValueError('Frozen count factors must be FP64')
    if not math.isfinite(p.log_scale) or bool((p.left <= 0).any() | (p.right <= 0).any()
            | ~torch.isfinite(p.left).all() | ~torch.isfinite(p.right).all()):
        raise ValueError('Finite shared scale and strictly positive count factors required')
    if p.left.device.type == 'cpu' and required > 64 * 1024**2:
        raise MemoryError('CPU construction is limited to small correctness fixtures (64 MiB)')
    if p.left.is_cuda:
        free, total = torch.cuda.mem_get_info(p.left.device)
        if required + max_temporary_bytes + reserve_bytes > free:
            raise MemoryError('Dense table plus construction scratch and reserve exceeds free GPU memory')
    else:
        free, total = None, None
    rows_per_chunk = max(1, min(vocab, max_temporary_bytes // (vocab * 12)))
    # Bound observed-edge temporaries as well (two indices, gathered values,
    # product, correction, logarithm and FP32 write values).
    pairs_per_chunk = max(1, min(262_144, max_temporary_bytes // 80))
    left_log, right_log = p.left.log(), p.right.log()
    scores = torch.empty((vocab, vocab), dtype=torch.float32, device=p.left.device)
    for start in range(0, vocab, rows_per_chunk):
        stop = min(vocab, start + rows_per_chunk)
        scores[start:stop] = (left_log[start:stop, None] + right_log[None, :]).float()
    flat = scores.view(-1)
    for start in range(0, p.pair_keys.numel(), pairs_per_chunk):
        keys = p.pair_keys[start:start + pairs_per_chunk]
        rows, cols = keys // vocab, keys % vocab
        weight = p.left[rows] * p.right[cols] + p.row_values[start:start + pairs_per_chunk]
        if bool((weight <= 0).any() | ~torch.isfinite(weight).all()):
            raise ValueError('Observed count edge must have positive finite weight')
        flat[keys] = weight.log().float()
    return DenseCountTable(scores, p.log_scale, {
        'backend': 'experimental_dense_static_count_log_table', 'vocab_size': vocab,
        'table_dtype': 'float32', 'table_bytes': required, 'max_table_bytes': max_table_bytes,
        'max_temporary_bytes': max_temporary_bytes, 'reserve_bytes': reserve_bytes,
        'temporary_budget_scope': 'table-fill chunks; excludes prior CSR-to-key conversion',
        'synthesized_pair_keys': synthesized_pair_keys,
        'synthesized_key_conversion_peak_bytes_estimate':
            24 * p.pair_keys.numel() if synthesized_pair_keys else 0,
        'synthesized_key_estimate_scope': 'conservative 24 bytes per observed edge; not measured',
        'rows_per_build_chunk': rows_per_chunk, 'pairs_per_build_chunk': pairs_per_chunk,
        'observed_edges': p.pair_keys.numel(), 'free_device_bytes_before_build': free,
        'total_device_bytes': total, 'omitted_shared_log_scale': p.log_scale,
        'build_timing_scope': 'explicit setup/warmup; caller must measure separately',
        'persistent_files_written': False}, potential, _signature(potential))


def clear_dense_count_cache():
    _TABLES.clear()


def prepare_dense_count_cache(potential, **budgets):
    """Cache one immutable potential per object; explicitly release when done."""
    signature = _signature(potential)
    cached = _TABLES.get(id(potential))
    if cached is None or cached[1] != signature:
        table = build_dense_count_table(potential, **budgets)
        _TABLES[id(potential)] = (potential, signature, table)
    else:
        table = cached[2]
        if table.log_scores.numel() * table.log_scores.element_size() > budgets.get('max_table_bytes', 12 * 1024**3):
            raise MemoryError('Existing dense table exceeds requested budget')
    return table


def dense_pair_scores(table, a, b):
    """Boundary/small-fixture lookup; invalid padded candidates are neutral."""
    a, b = torch.broadcast_tensors(a, b)
    vocab = table.vocab_size
    valid = (a >= 0) & (b >= 0) & (a < vocab) & (b < vocab)
    offset = flat_pair_offsets(a.clamp(0, vocab - 1), b.clamp(0, vocab - 1), vocab)
    score = table.log_scores.view(-1)[offset]
    return torch.where(valid, score, 0.)


def flat_pair_offsets(a, b, vocab):
    """Promote before multiplication even when a caller supplies int32 IDs."""
    return a.to(torch.int64) * vocab + b.to(torch.int64)


@dataclass
class DenseFactors:
    ids: torch.Tensor
    table: DenseCountTable

    def boundary(self, rows, positions, states, *, left_boundary):
        if left_boundary:
            a, b = self.ids[rows, positions, states, None], self.ids[rows, positions + 1]
        else:
            a, b = self.ids[rows, positions], self.ids[rows, positions + 1, states, None]
        return dense_pair_scores(self.table, a, b)


if triton is not None:
    @triton.jit
    def _flat_offset(a, b, vocab: tl.constexpr):
        return a.to(tl.int64) * vocab + b.to(tl.int64)

    @triton.jit
    def _pair_scores(a, b, table, vocab: tl.constexpr):
        valid = (a >= 0) & (b >= 0) & (a < vocab) & (b < vocab)
        # 50,258**2 exceeds signed int32: promote BEFORE multiplication.
        return tl.load(table + _flat_offset(a, b, vocab), valid, other=0.)

    @triton.jit
    def _offset_probe_kernel(a, b, output, size: tl.constexpr, vocab: tl.constexpr,
                             block: tl.constexpr):
        index = tl.program_id(0) * block + tl.arange(0, block)
        ai = tl.load(a + index, index < size, other=0)
        bi = tl.load(b + index, index < size, other=0)
        tl.store(output + index, _flat_offset(ai, bi, vocab), index < size)

    @triton.jit
    def _pair_tile(ids, table, row, pos, a, b,
                   original_length: tl.constexpr, states: tl.constexpr, vocab: tl.constexpr):
        ai = tl.load(ids + (row * original_length + pos) * states + a, a < states, other=-1)
        bi = tl.load(ids + (row * original_length + pos + 1) * states + b, b < states, other=-1)
        return _pair_scores(ai[:, None], bi[None, :], table, vocab)

    @triton.jit
    def _message_kernel(unary, messages, ids, table, rows, starts, lengths, pos, width,
                        original_length: tl.constexpr, states: tl.constexpr, vocab: tl.constexpr,
                        state_block: tl.constexpr, tile: tl.constexpr, backward: tl.constexpr):
        run = tl.program_id(0).to(tl.int64)
        selected = tl.program_id(1) * tile + tl.arange(0, tile)
        all_states = tl.arange(0, state_block)
        run_length = tl.load(lengths + run)
        row = tl.load(rows + run).to(tl.int64)
        start = tl.load(starts + run).to(tl.int64)
        if backward:
            if pos < run_length - 1:
                running_max = tl.full((tile,), -float('inf'), tl.float32)
                running_sum = tl.full((tile,), 0., tl.float32)
                for chunk in range((states + 127) // 128):
                    reduction = chunk * 128 + tl.arange(0, 128)
                    scores = _pair_tile(ids, table, row, start + pos, selected, reduction,
                                        original_length, states, vocab)
                    nxt = (run * width + pos + 1) * states + reduction
                    incoming = tl.load(messages + nxt, reduction < states, other=-float('inf'))
                    incoming += tl.load(unary + nxt, reduction < states, other=-float('inf'))
                    scores += incoming[None, :]
                    scores = tl.where(reduction[None, :] < states, scores, -float('inf'))
                    block_max = tl.max(scores, axis=1)
                    next_max = tl.maximum(running_max, block_max)
                    shift = tl.where(next_max == -float('inf'), 0., next_max)
                    running_sum = running_sum * tl.exp(running_max - shift) + tl.sum(
                        tl.exp(scores - shift[:, None]), axis=1)
                    running_max = next_max
                result = tl.log(running_sum) + running_max
            else:
                result = tl.full((tile,), 0., tl.float32)
        else:
            if pos < run_length:
                running_max = tl.full((tile,), -float('inf'), tl.float32)
                running_sum = tl.full((tile,), 0., tl.float32)
                for chunk in range((states + 127) // 128):
                    reduction = chunk * 128 + tl.arange(0, 128)
                    scores = _pair_tile(ids, table, row, start + pos - 1, reduction, selected,
                                        original_length, states, vocab)
                    incoming = tl.load(messages + (run * width + pos - 1) * states + reduction,
                                       reduction < states, other=-float('inf'))
                    scores += incoming[:, None]
                    scores = tl.where(reduction[:, None] < states, scores, -float('inf'))
                    block_max = tl.max(scores, axis=0)
                    next_max = tl.maximum(running_max, block_max)
                    shift = tl.where(next_max == -float('inf'), 0., next_max)
                    running_sum = running_sum * tl.exp(running_max - shift) + tl.sum(
                        tl.exp(scores - shift[None, :]), axis=0)
                    running_max = next_max
                result = tl.log(running_sum) + running_max
                result += tl.load(unary + (run * width + pos) * states + selected,
                                  selected < states, other=-float('inf'))
            else:
                previous = tl.load(messages + (run * width + pos - 1) * states + all_states,
                                   all_states < states, other=-float('inf'))
                maximum = tl.max(previous, axis=0)
                shift = tl.where(maximum == -float('inf'), 0., maximum)
                total = tl.log(tl.sum(tl.exp(previous - shift), axis=0)) + shift
                result = tl.where(selected == 0, total, -float('inf'))
        tl.store(messages + (run * width + pos) * states + selected, result, selected < states)

    @triton.jit
    def _sample_kernel(alpha, noise, output, invalid, ids, table, rows, starts, lengths, width,
                       original_length: tl.constexpr, states: tl.constexpr,
                       vocab: tl.constexpr, state_block: tl.constexpr):
        run = tl.program_id(0).to(tl.int64)
        a = tl.arange(0, state_block)
        row = tl.load(rows + run).to(tl.int64)
        start = tl.load(starts + run).to(tl.int64)
        run_length = tl.load(lengths + run)
        selected = tl.full((), 0, tl.int32)
        bad = tl.full((), False, tl.int1)
        for offset in range(width):
            pos = (width - 1 - offset).to(tl.int64)
            address = (run * width + pos) * states + a
            logits = tl.load(alpha + address, a < states, other=-float('inf'))
            if pos < run_length - 1:
                ai = tl.load(ids + (row * original_length + start + pos) * states + a,
                             a < states, other=-1)
                bi = tl.load(ids + (row * original_length + start + pos + 1) * states + selected)
                logits += _pair_scores(ai, bi, table, vocab)
            maximum = tl.max(logits, axis=0)
            bad = bad | (maximum == -float('inf')) | (tl.sum(
                ((a < states) & ((logits != logits) | (logits == float('inf')))).to(tl.int32), 0) > 0)
            scores = logits.to(tl.float64) - tl.load(noise + address, a < states, other=0.)
            selected = tl.argmax(scores, axis=0, tie_break_left=True).to(tl.int32)
            selected = tl.where(bad, 0, selected)
            tl.store(output + run * width + pos, selected)
        tl.store(invalid + run, bad.to(tl.int32))


def dense_messages(unary, factors, rows, starts, lengths, *, backward=False):
    batch, width, states = unary.shape
    result = torch.empty_like(unary)
    result[:, -1 if backward else 0] = 0. if backward else unary[:, 0]
    positions = range(width - 2, -1, -1) if backward else range(1, width)
    with torch.cuda.device(unary.device):
        for pos in positions:
            _message_kernel[(batch, triton.cdiv(states, 32))](
                unary, result, factors.ids, factors.table.log_scores, rows, starts, lengths,
                pos, width, factors.ids.shape[1], states, factors.table.vocab_size,
                max(32, triton.next_power_of_2(states)), 32, backward,
                num_warps=4, num_stages=1, enable_fp_fusion=False)
    return result


def pair_offset_probe(a, b, vocab):
    """GPU address gate sharing the lookup's arithmetic; allocates no table."""
    if triton is None or not a.is_cuda or a.shape != b.shape or a.device != b.device:
        raise ValueError('Probe requires same-shape CUDA integer IDs')
    if a.dtype not in (torch.int32, torch.int64) or b.dtype not in (torch.int32, torch.int64):
        raise ValueError('Integer IDs required')
    a, b = a.contiguous(), b.contiguous()
    result = torch.empty_like(a, dtype=torch.int64)
    _offset_probe_kernel[(triton.cdiv(a.numel(), 256),)](a, b, result, a.numel(), vocab, 256)
    return result


def dense_sample_with_noise(alpha, factors, rows, starts, lengths, noise):
    batch, width, states = alpha.shape
    result = torch.empty((batch, width), dtype=torch.long, device=alpha.device)
    invalid = torch.empty(batch, dtype=torch.int32, device=alpha.device)
    with torch.cuda.device(alpha.device):
        _sample_kernel[(batch,)](alpha, noise, result, invalid, factors.ids, factors.table.log_scores,
            rows, starts, lengths, width, factors.ids.shape[1], states, factors.table.vocab_size,
            max(32, triton.next_power_of_2(states)), num_warps=4, enable_fp_fusion=False)
    if invalid.any().item():
        raise RuntimeError('Invalid dense-count sampling distribution')
    return result


@torch.no_grad()
def infer_dense_count(candidates, potential, generator=None, *, table=None,
                      segment_batch_size=32, sampling='joint'):
    if triton is None or not candidates.log_unary.is_cuda or torch.version.hip is not None:
        raise ValueError('Static FFBS requires NVIDIA CUDA and Triton')
    if sampling not in ('joint', 'log_marginals'):
        raise ValueError('sampling must be joint or log_marginals')
    packet = reference.prepare_packet(candidates)
    table = prepare_dense_count_cache(potential) if table is None else table
    if (table.vocab_size != candidates.vocab_size or table.log_scores.device != packet.unary.device
            or table.owner is not potential or table.source_signature != _signature(potential)):
        raise ValueError('Dense table must match candidates and the exact immutable count potential')
    factors = DenseFactors(packet.candidate_ids, table)
    result = packet.unary.argmax(-1) if sampling == 'joint' else packet.unary.new_full(packet.unary.shape, -torch.inf)
    if sampling != 'joint':
        result.scatter_(-1, packet.unary.argmax(-1, keepdim=True), 0.)
    for unary, rows, starts, lengths, positions, valid in reference._chunks(packet, factors, segment_batch_size):
        alpha = dense_messages(unary, factors, rows, starts, lengths)
        if sampling == 'joint':
            noise = torch.empty_like(alpha, dtype=torch.float64).exponential_(generator=generator).log_()
            values = dense_sample_with_noise(alpha, factors, rows, starts, lengths, noise)
        else:
            beta = dense_messages(unary, factors, rows, starts, lengths, backward=True)
            values = alpha + beta - torch.logsumexp(alpha[:, -1], -1)[:, None, None]
        batches = rows[:, None].expand_as(positions)
        result[batches[valid], positions[valid]] = values[valid]
    return result


def sample_dense_count(candidates, potential, generator=None, *, table=None, segment_batch_size=32):
    slots = infer_dense_count(candidates, potential, generator, table=table,
                             segment_batch_size=segment_batch_size)
    return candidates.candidate_ids.gather(-1, slots[..., None]).squeeze(-1)


def dense_count_marginals(candidates, potential, *, table=None, segment_batch_size=32):
    return infer_dense_count(candidates, potential, table=table,
                            segment_batch_size=segment_batch_size, sampling='log_marginals').exp()
