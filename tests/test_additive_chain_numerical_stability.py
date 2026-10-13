"""Long-chain numerical gates, independent of implementation-to-port parity.

These fixed synthetic fixtures are correctness tests, not generation studies.
The two legal states straddle output tiles when width=65; centering a tile
separately would change their relative probability and fail the oracle.
"""
import math

import pytest
import torch

from additive_chain import chain, cuda, static
from additive_chain import _additive_kernels as additive
from additive_chain import _count_inference as reference
from additive_chain import _dense_inference as dense
from additive_chain.candidates import CompactCandidates
from additive_chain.counts import SparseCountPotential


def stable_oracle(unary, pair, lengths):
    """Independent FP64 dense reference on exactly the stored FP32 inputs."""
    u, p, n = unary.cpu().double(), pair.cpu().double(), lengths.cpu()
    alpha, _ = chain._forward(u, p, n)
    node = chain.marginals(u, p, n, return_edges=False).node
    return alpha, node


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual NVIDIA CUDA gate required')
@pytest.mark.parametrize('backend', ['dense', 'count', 'additive', 'cached_contextual'])
@pytest.mark.parametrize('width', [2, 65])
@torch.no_grad()
def test_cuda_long_chain_centering_probability_and_joint_draw(backend, width):
    length, batch = 1024, 2
    offset, switch_penalty = -9.306853, 30.
    device = 'cuda'
    ids = torch.arange(width, device=device)[None, None].expand(batch, length, width).contiguous()
    unary = torch.full((batch, length, width), -torch.inf, device=device)
    legal = torch.tensor([0, width-1], device=device)
    unary[..., legal] = torch.tensor([.0001, 0.], device=device).log_softmax(-1)
    lengths = torch.tensor([length, 19], device=device)
    unary[1, 19:] = -torch.inf
    unary[1, 19:, 0] = 0.
    ids = ids.masked_fill(~torch.isfinite(unary), width)
    candidates = CompactCandidates(ids, unary, width)
    base = math.exp((offset-switch_penalty)/2)
    correction = math.exp(offset) - math.exp(offset-switch_penalty)
    potential = SparseCountPotential(
        torch.full((width,), base, dtype=torch.float64, device=device),
        torch.full((width,), base, dtype=torch.float64, device=device),
        torch.tensor([0, width*width-1], device=device),
        torch.full((2,), correction, dtype=torch.float64, device=device))
    packet = reference.prepare_packet(candidates)
    rows, starts = torch.arange(batch, device=device), torch.zeros(batch, device=device, dtype=torch.long)
    table = static.prepare_static_count_cache(potential, reserve_bytes=0)
    left = torch.zeros(batch, length, width, 16, device=device)
    right = torch.zeros_like(left)
    # A nonzero but state-common learned residual activates both fused paths
    # without changing the target law. It must not reintroduce growing offsets.
    left[..., 0], right[..., 0] = .25, 1.
    if backend == 'dense':
        factors = dense.DenseFactors(packet.candidate_ids, table)
        message, sample = dense.dense_messages, dense.dense_sample_with_noise
    elif backend == 'count':
        factors = reference._factors(packet, potential)
        message, sample = reference.nf._messages, reference.nf._sample_with_noise
    elif backend == 'additive':
        factors = cuda.prepare_additive_factors(packet, potential, left, right)
        message, sample = additive._messages, additive._sample_with_noise
    else:
        factors = additive.PairFactors(left, right, packet.candidate_ids,
            ids.new_empty(0), table.log_scores, unary.new_empty(0), unary.new_empty(0),
            3, width, 16)
        message, sample = additive._messages, additive._sample_with_noise
    alpha = message(unary, factors, rows, starts, lengths)
    beta = message(unary, factors, rows, starts, lengths, backward=True)
    pair = table.log_scores[legal[:, None], legal[None, :]]
    if backend in ('additive', 'cached_contextual'):
        pair = pair + .25
    pair = pair[None, None].expand(batch, length-1, 2, 2).contiguous()
    oracle_alpha, oracle_node = stable_oracle(unary[..., legal], pair, lengths)
    valid = torch.arange(length)[None] < lengths.cpu()[:, None]
    actual_conditional = alpha[..., legal].double().softmax(-1).cpu()
    actual_node = (alpha + beta).double().softmax(-1)[..., legal].cpu()
    assert (actual_conditional[valid] - oracle_alpha.softmax(-1)[valid]).abs().max() < 1e-4
    assert (actual_node[valid] - oracle_node[valid]).abs().max() < 2e-4
    # Stored messages retain at most a local edge offset, not length*offset.
    assert alpha[torch.isfinite(alpha)].abs().max() < 100.
    noise = torch.empty_like(alpha, dtype=torch.float64).exponential_(
        generator=torch.Generator(device=device).manual_seed(61373)).log_()
    actual = sample(alpha, factors, rows, starts, lengths, noise).cpu()
    local_noise = noise[..., legal].cpu()
    expected = torch.zeros(batch, length, dtype=torch.long)
    p = pair.cpu().double()
    for row, n in enumerate(lengths.cpu().tolist()):
        for pos in range(n-1, -1, -1):
            logits = oracle_alpha[row, pos]
            if pos < n-1:
                logits = logits + p[row, pos, :, expected[row, pos+1]]
            expected[row, pos] = (logits-local_noise[row, pos]).argmax()
    expected = torch.tensor([0, width-1])[expected]
    assert torch.equal(actual[valid], expected[valid])
    reference.clear_count_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual NVIDIA CUDA gate required')
@pytest.mark.parametrize('backend', ['dense', 'count'])
@torch.no_grad()
def test_cuda_raw_marginal_helpers_normalize_long_equal_state_chain(backend):
    length, width = 1024, 2
    ids = torch.arange(width, device='cuda')[None, None].expand(1, length, width).contiguous()
    unary = torch.full(ids.shape, -math.log(2), device='cuda')
    candidates = CompactCandidates(ids, unary, width)
    potential = SparseCountPotential(
        torch.full((width,), math.exp(-9.306853/2), dtype=torch.float64, device='cuda'),
        torch.full((width,), math.exp(-9.306853/2), dtype=torch.float64, device='cuda'),
        torch.empty(0, dtype=torch.long, device='cuda'),
        torch.empty(0, dtype=torch.float64, device='cuda'))
    if backend == 'dense':
        table = static.prepare_static_count_cache(potential, reserve_bytes=0)
        node = dense.dense_count_marginals(candidates, potential, table=table)
    else:
        node = reference.nina_count_marginals(candidates, potential)
    torch.testing.assert_close(node, torch.full_like(node, .5), atol=1e-7, rtol=0.)
    reference.clear_count_cache()
