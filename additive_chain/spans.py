"""Exact reveal-target pruning for a nearest-neighbor chain with visible clamps.

Visible positions separate the remaining masked nodes into independent spans.
A span with no requested reveal contributes only a scalar to the selected-token
law. Replace that entire span with dummy singleton constraints before candidate
extraction to avoid its sorting and inference work. Never prune a hidden node
inside a span that contains a requested reveal.

Use the returned context ONLY in CRF support preparation/inference. The denoiser
forward still receives the original tokens, and the final update still uses
the original reveal mask/tokens. This is not valid for extra nonlocal edges or
an unclamped visible node. The count potential's strictly positive smoothing
makes every dummy assignment legal.
"""

import torch


@torch.no_grad()
def relevant_span_context(tokens, reveal_mask, *, mask_id, dummy_token_id=0):
    """Return [B,L] tokens with irrelevant masked components clamped.

    O(B*L) device work, no topology transfer or device-dependent Python loop.
    The caller must supply valid token IDs and a valid clean dummy ID for its
    vocabulary. Reveals at already visible positions do not activate a span.
    """
    if (tokens.ndim != 2 or min(tokens.shape) < 1 or tokens.dtype != torch.long
            or reveal_mask.shape != tokens.shape or reveal_mask.dtype != torch.bool
            or reveal_mask.device != tokens.device):
        raise ValueError('Expected nonempty int64 tokens and matching bool reveal mask [B,L]')
    if (type(mask_id) is not int or mask_id < 0 or type(dummy_token_id) is not int
            or dummy_token_id < 0 or dummy_token_id == mask_id):
        raise ValueError('Mask and dummy must be distinct nonnegative token IDs')
    masked = tokens.eq(mask_id)
    # Each visible token starts a new group. Masked sites after it share that
    # group up to the next visible token; the initial boundary span has ID 0.
    groups = (~masked).long().cumsum(-1)
    active_groups = torch.zeros((tokens.shape[0], tokens.shape[1] + 1),
                                dtype=torch.int32, device=tokens.device)
    active_groups.scatter_add_(1, groups, (reveal_mask & masked).int())
    relevant = active_groups.gather(1, groups).gt(0)
    return tokens.masked_fill(masked & ~relevant, dummy_token_id)

