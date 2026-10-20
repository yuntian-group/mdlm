# Portions derived from Nina Zhang, chain_crf/fused_pairs.py, Apache-2.0.
# yuntian-group/mdlm commit 18175679c93d7792936e1497b7c5be72a765f2de.
# See CUDA_BACKEND.md for attribution, changes, and validation scope.
"""Count plus signed log-bilinear CUDA kernels, derived from Nina Zhang.

Changes to the count kernels: kind2 adds both count and learned scores; zero
learned contributions delegate to the unchanged count recurrence and sampler.
kind3 reads a cached count/global table and adds fresh contextual dot scores.
No support pooling or probability-rank approximation. Inference only.
"""
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None



if triton is not None:
    @triton.jit
    def _count_hash(key, capacity: tl.constexpr):
        # Mix both halves for large vocabularies; normal token pairs fit uint32.
        x = key.to(tl.uint32) ^ (key.to(tl.uint64) >> 32).to(tl.uint32)
        x = (x ^ (x >> 16)) * 0x7feb352d
        x = (x ^ (x >> 15)) * 0x846ca68b
        return (x ^ (x >> 16)) & (capacity - 1)


    @triton.jit
    def _insert_count_hash(keys, values, table_keys, table_values,
                           entries: tl.constexpr, capacity: tl.constexpr,
                           block: tl.constexpr):
        index = tl.program_id(0) * block + tl.arange(0, block)
        pending = index < entries
        key = tl.load(keys + index, pending, other=0)
        value = tl.load(values + index, pending, other=0.)
        stored_key = key.to(table_keys.dtype.element_ty)
        empty = tl.full((block,), -1, table_keys.dtype.element_ty)
        slot = _count_hash(key, capacity)
        while tl.sum(pending.to(tl.int32), 0) > 0:
            # atomic_cas has no mask: inactive lanes perform a no-op CAS at 0.
            previous = tl.atomic_cas(table_keys + tl.where(pending, slot, 0),
                                     empty, tl.where(pending, stored_key, empty))
            inserted = pending & ((previous == -1) | (previous == stored_key))
            tl.store(table_values + slot, value, inserted)
            pending = pending & ~inserted
            slot = (slot + 1) & (capacity - 1)


    @triton.jit
    def _count_scores(a, b, keys, values, left_bg, right_bg,
                      vocab: tl.constexpr, capacity: tl.constexpr):
        valid = (a >= 0) & (b >= 0)
        wanted = a.to(tl.int64) * vocab + b
        stored_key = wanted.to(keys.dtype.element_ty)
        slot = _count_hash(wanted, capacity)
        pending = valid
        score = tl.full(wanted.shape, 0., tl.float32)
        found = tl.full(wanted.shape, False, tl.int1)
        while tl.sum(pending.to(tl.int32)) > 0:
            key = tl.load(keys + slot, pending, other=-1)
            hit = pending & (key == stored_key)
            score = tl.where(hit, tl.load(values + slot, hit, other=0.), score)
            found = found | hit
            pending = pending & ~hit & (key != -1)
            slot = (slot + 1) & (capacity - 1)
        bg = tl.load(left_bg + a, a >= 0, other=0.) + tl.load(right_bg + b, b >= 0, other=0.)
        return tl.where(valid, tl.where(found, score, bg).to(tl.float32), 0.)


    @triton.jit
    def _count_query_kernel(a, b, keys, values, left_bg, right_bg, output,
                            size: tl.constexpr, vocab: tl.constexpr,
                            capacity: tl.constexpr, block: tl.constexpr):
        index = tl.program_id(0) * block + tl.arange(0, block)
        ai = tl.load(a + index, index < size, other=-1)
        bi = tl.load(b + index, index < size, other=-1)
        score = _count_scores(ai, bi, keys, values, left_bg, right_bg, vocab, capacity)
        tl.store(output + index, score, index < size)


    @triton.jit(do_not_specialize=['original_length'])
    def _pair_tile(left, right, ids, keys, values, left_bg, right_bg,
                   original_row, edge_pos, a, b,
                   original_length, states: tl.constexpr,
                   rank: tl.constexpr, rank_block: tl.constexpr,
                   kind: tl.constexpr, vocab: tl.constexpr,
                   capacity: tl.constexpr):
        # kind2 is COUNT + SIGNED LOG DOT, not an either/or switch.
        score = tl.full((a.shape[0], b.shape[0]), 0., tl.float32)
        if (kind == 0 or kind == 2) or kind == 3:
            r = tl.arange(0, rank_block)
            x = tl.load(left + ((original_row * original_length + edge_pos) * states
                                + a[:, None]) * rank + r[None, :],
                        (a[:, None] < states) & (r[None, :] < rank), other=0.)
            y = tl.load(right + ((original_row * original_length + edge_pos + 1) * states
                                 + b[None, :]) * rank + r[:, None],
                        (b[None, :] < states) & (r[:, None] < rank), other=0.)
            score = tl.dot(x, y, input_precision='ieee')
        if kind == 1 or kind == 2:
            ai = tl.load(ids + (original_row * original_length + edge_pos) * states + a,
                         a < states, other=-1)
            bi = tl.load(ids + (original_row * original_length + edge_pos + 1) * states + b,
                         b < states, other=-1)
            score += _count_scores(ai[:, None], bi[None, :], keys, values, left_bg, right_bg,
                                   vocab, capacity)
        if kind == 3:
            ai = tl.load(ids + (original_row * original_length + edge_pos) * states + a,
                         a < states, other=-1).to(tl.int64)
            bi = tl.load(ids + (original_row * original_length + edge_pos + 1) * states + b,
                         b < states, other=-1).to(tl.int64)
            valid = (ai[:, None] >= 0) & (ai[:, None] < vocab) & (bi[None, :] >= 0) & (bi[None, :] < vocab)
            score += tl.load(values + ai[:, None] * vocab + bi[None, :], valid, other=0.)
        return score


    @triton.jit(do_not_specialize=['original_length', 'pos', 'width'])
    def _message_kernel(unary, messages, left, right, ids, keys, values, left_bg, right_bg,
                        rows, starts, lengths, pos, width,
                        original_length, states: tl.constexpr,
                        rank: tl.constexpr, rank_block: tl.constexpr, kind: tl.constexpr,
                        vocab: tl.constexpr, capacity: tl.constexpr,
                        state_block: tl.constexpr, tile: tl.constexpr, backward: tl.constexpr):
        run = tl.program_id(0).to(tl.int64)
        selected = tl.program_id(1) * tile + tl.arange(0, tile)
        all_states = tl.arange(0, state_block)
        run_length = tl.load(lengths + run)
        row = tl.load(rows + run).to(tl.int64)
        start = tl.load(starts + run).to(tl.int64)
        if backward:
            if pos < run_length - 1:
                # One center over ALL states, shared by every output tile.
                # Local reduction shifts alone do not stop O(length) offsets.
                previous = tl.load(messages + (run * width + pos + 1) * states + all_states,
                                   all_states < states, other=-float('inf'))
                center = tl.max(previous, axis=0)
                center = tl.where(center == -float('inf'), 0., center)
                running_max = tl.full((tile,), -float('inf'), tl.float32)
                running_sum = tl.full((tile,), 0., tl.float32)
                for chunk in range((states + 127) // 128):
                    reduction_states = chunk * 128 + tl.arange(0, 128)
                    scores = _pair_tile(left, right, ids, keys, values, left_bg, right_bg,
                                        row, start + pos, selected, reduction_states,
                                        original_length, states, rank, rank_block, kind,
                                        vocab, capacity)
                    nxt = (run * width + pos + 1) * states + reduction_states
                    incoming = tl.load(messages + nxt, reduction_states < states, other=-float('inf'))
                    incoming = incoming - center
                    incoming += tl.load(unary + nxt, reduction_states < states, other=-float('inf'))
                    scores += incoming[None, :]
                    scores = tl.where(reduction_states[None, :] < states, scores, -float('inf'))
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
                previous = tl.load(messages + (run * width + pos - 1) * states + all_states,
                                   all_states < states, other=-float('inf'))
                center = tl.max(previous, axis=0)
                center = tl.where(center == -float('inf'), 0., center)
                running_max = tl.full((tile,), -float('inf'), tl.float32)
                running_sum = tl.full((tile,), 0., tl.float32)
                for chunk in range((states + 127) // 128):
                    reduction_states = chunk * 128 + tl.arange(0, 128)
                    scores = _pair_tile(left, right, ids, keys, values, left_bg, right_bg,
                                        row, start + pos - 1, reduction_states, selected,
                                        original_length, states, rank, rank_block, kind,
                                        vocab, capacity)
                    incoming = tl.load(messages + (run * width + pos - 1) * states + reduction_states,
                                       reduction_states < states, other=-float('inf'))
                    scores += (incoming - center)[:, None]
                    scores = tl.where(reduction_states[:, None] < states, scores, -float('inf'))
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
                # Neutral edges and point-mass padding reproduce packed chains.
                previous = tl.load(messages + (run * width + pos - 1) * states + all_states,
                                   all_states < states, other=-float('inf'))
                padding_max = tl.max(previous, axis=0)
                padding_shift = tl.where(padding_max == -float('inf'), 0., padding_max)
                total = tl.log(tl.sum(tl.exp(previous - padding_shift), axis=0)) + padding_shift
                result = tl.where(selected == 0, total, -float('inf'))
        tl.store(messages + (run * width + pos) * states + selected, result, selected < states)


    @triton.jit(do_not_specialize=['original_length', 'width'])
    def _sample_kernel(alpha, noise, output, invalid, left, right, ids, keys, values,
                       left_bg, right_bg, rows, starts, lengths, width,
                       original_length, states: tl.constexpr,
                       rank: tl.constexpr, rank_block: tl.constexpr, kind: tl.constexpr,
                       vocab: tl.constexpr, capacity: tl.constexpr,
                       state_block: tl.constexpr):
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
                column = tl.full((state_block,), 0., tl.float32)
                if (kind == 0 or kind == 2) or kind == 3:
                    r = tl.arange(0, rank_block)
                    x = tl.load(left + ((row * original_length + start + pos) * states
                                        + a[:, None]) * rank + r[None, :],
                                (a[:, None] < states) & (r[None, :] < rank), other=0.)
                    y = tl.load(right + ((row * original_length + start + pos + 1) * states
                                         + selected) * rank + r,
                                r < rank, other=0.)
                    column = tl.sum(x * y[None, :], axis=1)
                if kind == 1 or kind == 2:
                    ai = tl.load(ids + (row * original_length + start + pos) * states + a,
                                 a < states, other=-1)
                    bi = tl.load(ids + (row * original_length + start + pos + 1) * states + selected)
                    column += _count_scores(ai, bi, keys, values, left_bg, right_bg,
                                           vocab, capacity)
                if kind == 3:
                    ai = tl.load(ids + (row * original_length + start + pos) * states + a,
                                 a < states, other=-1).to(tl.int64)
                    bi = tl.load(ids + (row * original_length + start + pos + 1) * states + selected).to(tl.int64)
                    valid = (ai >= 0) & (ai < vocab) & (bi >= 0) & (bi < vocab)
                    column += tl.load(values + ai * vocab + bi, valid, other=0.)
                logits += column
            maximum = tl.max(logits, axis=0)
            bad = bad | (maximum == -float('inf')) | (tl.sum(
                ((a < states) & ((logits != logits) | (logits == float('inf')))).to(tl.int32), 0) > 0)
            perturbation = tl.load(noise + address, a < states, other=0.)
            scores = logits.to(tl.float64) - perturbation
            selected = tl.argmax(scores, axis=0, tie_break_left=True).to(tl.int32)
            selected = tl.where(bad, 0, selected)
            tl.store(output + run * width + pos, selected)
        tl.store(invalid + run, bad.to(tl.int32))


@dataclass
class PairFactors:
    left: torch.Tensor
    right: torch.Tensor
    ids: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    left_bg: torch.Tensor
    right_bg: torch.Tensor
    kind: int
    vocab: int
    rank: int

    def kernel_args(self):
        return (self.left, self.right, self.ids, self.keys, self.values,
                self.left_bg, self.right_bg)

    def constants(self):
        return (self.ids.shape[1], self.ids.shape[2], self.rank,
                max(32, triton.next_power_of_2(self.rank)), self.kind, self.vocab,
                self.keys.numel())

    def boundary(self, rows, positions, states, *, left_boundary):
        """Add count and learned factors at both visible endpoints."""
        learned = 0.
        if (self.kind == 0 or self.kind == 2) or self.kind == 3:
            if left_boundary:
                learned = (self.left[rows, positions, states, None, :] *
                           self.right[rows, positions + 1]).sum(-1)
            else:
                learned = (self.left[rows, positions] *
                           self.right[rows, positions + 1, states, None, :]).sum(-1)
        if self.kind == 0:
            return learned
        if left_boundary:
            a, b = self.ids[rows, positions, states, None], self.ids[rows, positions + 1]
        else:
            a, b = self.ids[rows, positions], self.ids[rows, positions + 1, states, None]
        if self.kind == 3:
            valid = (a >= 0) & (a < self.vocab) & (b >= 0) & (b < self.vocab)
            count = self.values[a.clamp(0, self.vocab-1), b.clamp(0, self.vocab-1)]
            count = torch.where(valid, count, 0.)
        else:
            count = _lookup_count_pairs(a, b, self.keys, self.values,
                                        self.left_bg, self.right_bg, self.vocab)
        return count + learned


def _build_count_hash(keys, values, vocab):
    """Immutable linear-probing map with load <= 0.5 and exact key checks.

    Int32 storage preserves all uint32 key bits. Reserve all-ones for an empty
    slot; use int64 if the vocabulary could produce that sentinel or wider keys.
    For 17.7M observed pairs this uses 512 MiB, independent of sequence length.
    """
    capacity = triton.next_power_of_2(max(2, 2 * keys.numel()))
    dtype = torch.int32 if vocab * vocab < 2**32 else torch.int64
    table_keys = torch.full((capacity,), -1, dtype=dtype, device=keys.device)
    table_values = torch.empty(capacity, dtype=torch.float32, device=keys.device)
    if keys.numel():
        with torch.cuda.device(keys.device):
            _insert_count_hash[(triton.cdiv(keys.numel(), 256),)](
                keys, values, table_keys, table_values, keys.numel(), capacity, 256)
    return table_keys, table_values


def _lookup_count_pairs(a, b, keys, values, left_bg, right_bg, vocab):
    """Small boundary queries; bulk pair queries remain inside message kernels."""
    a, b = torch.broadcast_tensors(a, b)
    a, b = a.contiguous(), b.contiguous()
    output = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    if output.numel():
        with torch.cuda.device(a.device):
            _count_query_kernel[(triton.cdiv(output.numel(), 256),)](
                a, b, keys, values, left_bg, right_bg, output,
                output.numel(), vocab, keys.numel(), 256, enable_fp_fusion=False)
    return output


def _messages(unary, factors, rows, starts, lengths, *, backward=False):
    if factors.kind == 1:
        # Preserve the pinned count-only recurrence bit-for-bit when every
        # learned contribution is zero; do not merely rely on adding zero in
        # a differently laid-out dot-product kernel.
        from . import _count_kernels as frozen_count
        return frozen_count._messages(unary, factors, rows, starts, lengths,
                                      backward=backward)
    batch, width, states = unary.shape
    result = torch.empty_like(unary)
    if backward:
        result[:, -1] = 0.
        positions = range(width - 2, -1, -1)
    else:
        result[:, 0] = unary[:, 0]
        positions = range(1, width)
    tile = 32
    with torch.cuda.device(unary.device):
        for pos in positions:
            _message_kernel[(batch, triton.cdiv(states, tile))](
                unary, result, *factors.kernel_args(), rows, starts, lengths,
                pos, width, *factors.constants(), max(32, triton.next_power_of_2(states)),
                tile, backward, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return result


def _sample_with_noise(alpha, factors, rows, starts, lengths, log_noise):
    if factors.kind == 1:
        from . import _count_kernels as frozen_count
        return frozen_count._sample_with_noise(alpha, factors, rows, starts,
                                               lengths, log_noise)
    batch, width, states = alpha.shape
    result = torch.empty((batch, width), dtype=torch.long, device=alpha.device)
    invalid = torch.empty(batch, dtype=torch.int32, device=alpha.device)
    with torch.cuda.device(alpha.device):
        _sample_kernel[(batch,)](
            alpha, log_noise, result, invalid, *factors.kernel_args(), rows, starts,
            lengths, width, *factors.constants(), max(32, triton.next_power_of_2(states)),
            num_warps=4, enable_fp_fusion=False)
    if invalid.any().item():
        raise RuntimeError('Invalid CRF sampling distribution: NaN, +inf, or no reachable state')
    return result

