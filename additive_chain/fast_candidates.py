"""Exact capped candidates without sorting all 50k vocabulary entries.

The original support rule is unchanged: normalize the FULL clean vocabulary,
apply exclude-crossing top-p, intersect top-K, renormalize, then sort token IDs.
For FP32 inputs an int64 lexicographic key selects precisely the original
descending-score/ascending-ID order. FP64/other inputs use the frozen code.

Rare subtraction-rounding ambiguity or a cumulative probability extremely
close to p triggers the frozen full-sort path. This guards support boundaries,
not a looser sampling approximation. Original nucleus sizes/masses are NOT
computed by the fast path and are explicitly unavailable in diagnostics.
"""
import math

import torch

from .candidates import CompactCandidates, extract_compact_candidates as _reference


def _diagnostics(masked, sizes, retained_mass, fallback_rows, backend):
    return {'masked': masked, 'support_sizes': sizes,
            'retained_original_mass': retained_mass,
            'nucleus_sizes': None, 'nucleus_original_mass': None,
            'diagnostic_scope': 'retained_support_only',
            'extraction_backend': backend, 'reference_fallback_rows': fallback_rows}


def _fallback(log_probs, tokens, *, mask_id, nucleus_p, top_k, row_chunk_size):
    result = _reference(log_probs, tokens, mask_id=mask_id, nucleus_p=nucleus_p,
                        top_k=top_k, row_chunk_size=row_chunk_size)
    old = result.diagnostics
    return CompactCandidates(result.candidate_ids, result.log_unary, result.vocab_size,
        _diagnostics(old['masked'], old['support_sizes'], old['retained_original_mass'],
                     int(old['masked'].sum()), 'frozen_reference'))


@torch.no_grad()
def extract_compact_candidates(log_probs, tokens, *, mask_id, nucleus_p=.9,
                               top_k, row_chunk_size=32):
    if type(top_k) is not int or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    if (not isinstance(nucleus_p, (int, float)) or isinstance(nucleus_p, bool)
            or not math.isfinite(nucleus_p) or not 0 < nucleus_p <= 1):
        raise ValueError('nucleus_p must lie in (0,1]')
    if type(row_chunk_size) is not int or row_chunk_size < 1:
        raise ValueError('row_chunk_size must be positive')
    if (log_probs.ndim != 3 or not log_probs.is_floating_point()
            or tokens.shape != log_probs.shape[:2] or tokens.dtype != torch.long
            or tokens.device != log_probs.device or min(log_probs.shape) < 1):
        raise ValueError('expected floating [B,L,V] logits and int64 [B,L] tokens')
    vocab = log_probs.shape[-1]
    if type(mask_id) is not int or not 0 <= mask_id < vocab:
        raise ValueError('mask_id must be in the vocabulary')
    if bool(((tokens < 0) | (tokens >= vocab)).any()):
        raise ValueError('tokens outside vocabulary')
    if bool(torch.isnan(log_probs).any() | torch.isposinf(log_probs).any()):
        raise ValueError('logits may contain finite values and -inf only')
    if log_probs.dtype != torch.float32 or vocab >= 2**31:
        return _fallback(log_probs, tokens, mask_id=mask_id, nucleus_p=nucleus_p,
                         top_k=top_k, row_chunk_size=row_chunk_size)
    width = min(top_k, vocab)
    ids = torch.full((*tokens.shape, width), vocab, dtype=torch.long, device=tokens.device)
    values = torch.full(ids.shape, -torch.inf, dtype=torch.float64, device=tokens.device)
    ids[..., 0], values[..., 0] = tokens, 0.
    masked = tokens.eq(mask_id)
    positions_all = masked.flatten().nonzero().flatten()
    flat_logits = log_probs.reshape(-1, vocab)
    sizes = torch.ones(tokens.shape, dtype=torch.long, device=tokens.device)
    retained_mass = torch.ones(tokens.shape, dtype=torch.float64, device=tokens.device)
    rank_tie = vocab - 1 - torch.arange(vocab, device=tokens.device, dtype=torch.int64)
    fallback_rows = 0
    # A conservative positive-sum rounding bound; near-boundary rows use the
    # same full-length scan and sort as the frozen reference instead.
    epsilon = torch.finfo(torch.float64).eps
    cumulative_guard = 4 * (vocab * epsilon) / (1 - vocab * epsilon)
    for start in range(0, positions_all.numel(), row_chunk_size):
        positions = positions_all[start:start + row_chunk_size]
        raw = flat_logits[positions].clone()
        raw[:, mask_id] = -torch.inf
        original = raw.double()
        normalizer = original.logsumexp(-1, keepdim=True)
        if not bool(torch.isfinite(normalizer).all()):
            raise ValueError('every hidden position needs a finite clean token')
        rows = original - normalizer
        # Recoverability certifies distinct FP32 scores did not collapse into
        # an artificial FP64 tie after subtracting a huge normalizer.
        ambiguous = ((rows + normalizer != original) & torch.isfinite(original)).any(-1)
        # Canonicalize signed zero: the original stable sort considers them
        # equal, whereas raw IEEE ordering would distinguish their sign bits.
        canonical = torch.where(raw == 0, torch.zeros((), dtype=raw.dtype, device=raw.device), raw)
        bits = canonical.view(torch.int32).to(torch.int64).bitwise_and(0xffffffff)
        monotonic = torch.where(bits.bitwise_and(0x80000000).ne(0),
                               bits.bitwise_xor(0xffffffff), bits.bitwise_xor(0x80000000))
        key = monotonic * (vocab + 1) + rank_tie
        rank_ids = key.topk(width, dim=-1, largest=True, sorted=True).indices
        ordered = rows.gather(-1, rank_ids)
        probability = ordered.exp()
        keep = torch.isfinite(ordered)
        if nucleus_p < 1:
            cumulative = probability.cumsum(-1)
            ambiguous |= ((cumulative - nucleus_p).abs() <= cumulative_guard).any(-1)
            keep &= cumulative <= nucleus_p
            keep[:, 0] = True
        capped = ordered.masked_fill(~keep, -torch.inf)
        mass = capped.exp().sum(-1)
        capped -= capped.logsumexp(-1, keepdim=True)
        chosen_ids = rank_ids.masked_fill(~keep, vocab)
        chosen_ids, permutation = chosen_ids.sort(dim=-1, stable=True)
        chosen_values = capped.gather(-1, permutation)
        row_sizes = keep.sum(-1)
        if bool(ambiguous.any()):
            count = int(ambiguous.sum())
            fallback_rows += count
            reference = _reference(flat_logits[positions[ambiguous]][None],
                tokens.flatten()[positions[ambiguous]][None], mask_id=mask_id,
                nucleus_p=nucleus_p, top_k=top_k, row_chunk_size=row_chunk_size)
            chosen_ids[ambiguous] = reference.candidate_ids[0]
            chosen_values[ambiguous] = reference.log_unary[0]
            row_sizes[ambiguous] = reference.diagnostics['support_sizes'][0]
            mass[ambiguous] = reference.diagnostics['retained_original_mass'][0]
        ids.reshape(-1, width)[positions] = chosen_ids
        values.reshape(-1, width)[positions] = chosen_values
        sizes.flatten()[positions] = row_sizes
        retained_mass.flatten()[positions] = mass
    return CompactCandidates(ids, values, vocab,
        _diagnostics(masked, sizes, retained_mass, fallback_rows, 'lexicographic_fp32_topk'))


extract_fast_candidates = extract_compact_candidates

