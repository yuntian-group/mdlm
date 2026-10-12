"""Portable score-law tests and explicit optional CUDA inference checks."""
from types import SimpleNamespace
import itertools

import pytest
import torch

from additive_chain.candidates import CompactCandidates
from additive_chain.counts import SparseCountPotential
from additive_chain.heads import GlobalAdditiveHead, ContextualAdditiveHead, concatenate_scaled_factors
from additive_chain.cuda import (dense_pair_reference, prepare_additive_factors,
                                sample_additive_count, additive_count_marginals)
import additive_chain._count_inference as old
import additive_chain._additive_kernels as af
from additive_chain.chain import marginals


def potential_fixture(vocab, device='cpu'):
    left = torch.linspace(.6, .9, vocab, device=device, dtype=torch.float64)
    right = torch.linspace(.7, .4, vocab, device=device, dtype=torch.float64)
    rows = torch.tensor([0, 1, vocab-2], device=device)
    cols = torch.tensor([1, 0, vocab-1], device=device)
    values = torch.tensor([.3, .6, .2], device=device, dtype=torch.float64)
    return SparseCountPotential(left, right, rows*vocab+cols, values, log_scale=7.)


def fixture(width=3, length=5, device='cpu'):
    rng = torch.Generator(device=device).manual_seed(137)
    ids = torch.arange(width, device=device)[None, None].expand(2, length, width).clone()
    unary = torch.randn(2, length, width, dtype=torch.float64, device=device,
                        generator=rng).log_softmax(-1)
    for pos in sorted({0, length//2, length-1}):
        ids[1, pos] = width+2
        ids[1, pos, 0] = 1
        unary[1, pos] = -torch.inf
        unary[1, pos, 0] = 0.
    return CompactCandidates(ids, unary, width+2), potential_fixture(width+2, device)


def test_cpu_zero_coefficients_are_exact_count_scores():
    candidates, potential = fixture()
    left = torch.randn(2, 5, 3, 64)
    right = torch.randn_like(left)
    zero = concatenate_scaled_factors([(0., (left, right))])
    assert torch.equal(dense_pair_reference(candidates, potential),
                       dense_pair_reference(candidates, potential, *zero))


def test_cpu_global_local_concatenation_and_observed_clamps():
    candidates, potential = fixture()
    g = GlobalAdditiveHead(5, rank=64)
    l = ContextualAdditiveHead(torch.randn(5, 6), hidden_size=4, rank=64)
    with torch.no_grad():
        g.right_embedding.weight.normal_(std=.01)
        l.right_token.weight.normal_(std=.01)
    gp = g.node_factors(candidates.candidate_ids)
    lp = l.node_factors(candidates.candidate_ids, torch.randn(2, 5, 4))
    factors = concatenate_scaled_factors([(.7, gp), (.3, lp)])
    base = dense_pair_reference(candidates, potential)
    expected = base + .7*torch.einsum('blar,blcr->blac', gp[0][:, :-1], gp[1][:, 1:])
    expected += .3*torch.einsum('blar,blcr->blac', lp[0][:, :-1], lp[1][:, 1:])
    actual = dense_pair_reference(candidates, potential, *factors)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    node = marginals(candidates.log_unary, actual.double(), return_edges=False).node
    assert torch.equal(node[1, [0, 2, 4], 0], torch.ones(3, dtype=node.dtype))
    assert (node[1, [0, 2, 4], 1:] == 0).all()


def test_cpu_dense_oracle_budget_and_factor_validation():
    candidates, potential = fixture()
    with pytest.raises(ValueError, match='budget'):
        dense_pair_reference(candidates, potential, max_cells=1)
    with pytest.raises(ValueError, match='factors'):
        dense_pair_reference(candidates, potential, torch.zeros(2, 5, 3, 129), torch.zeros(2, 5, 3, 129))


cuda = pytest.mark.skipif(not torch.cuda.is_available() or af.triton is None, reason='Explicit additive CUDA gate required')


@cuda
@pytest.mark.parametrize('width,length', [(3, 7), (500, 4)])
def test_cuda_zero_residual_exact_count_draws_and_messages(width, length):
    old.clear_count_cache()
    candidates, potential = fixture(width, length, 'cuda')
    left = torch.zeros(2, length, width, 128, device='cuda')
    right = torch.ones_like(left)
    expected = old.sample_nina_count(candidates, potential,
        torch.Generator(device='cuda').manual_seed(147), segment_batch_size=3)
    actual = sample_additive_count(candidates, potential, left, right,
        torch.Generator(device='cuda').manual_seed(147), segment_batch_size=3)
    assert torch.equal(actual, expected)
    packet = old.prepare_packet(candidates)
    base = old._factors(packet, potential)
    additive = prepare_additive_factors(packet, potential, left, right)
    assert additive.kind == 1  # explicit bitwise count-only nesting
    for raw, new in zip(old._chunks(packet, base, 3), old._chunks(packet, additive, 3)):
        assert all(torch.equal(a, b) for a, b in zip(raw, new))
        unary, rows, starts, lengths, _, _ = raw
        a = old.nf._messages(unary, base, rows, starts, lengths)
        b = af._messages(unary, additive, rows, starts, lengths)
        assert torch.equal(a, b)


@cuda
@pytest.mark.parametrize('width,length', [(3, 7), (500, 4), (16, 1024)])
def test_cuda_additive_scores_clamps_marginals_and_sampling(width, length):
    old.clear_count_cache()
    candidates, potential = fixture(width, length, 'cuda')
    rng = torch.Generator(device='cuda').manual_seed(337)
    left = torch.randn(2, length, width, 128, device='cuda', generator=rng) * .02
    right = torch.randn(2, length, width, 128, device='cuda', generator=rng) * .02
    pair = dense_pair_reference(candidates, potential, left, right)
    # Match input FP32 conversion; use stable FP64 reference recurrence.
    expected = marginals(candidates.log_unary.float().double(), pair.double(), return_edges=False).node
    actual = additive_count_marginals(candidates, potential, left, right, segment_batch_size=3).double()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual.sum(-1)), atol=1e-5, rtol=0.)
    first = sample_additive_count(candidates, potential, left, right,
        torch.Generator(device='cuda').manual_seed(377), segment_batch_size=3)
    second = sample_additive_count(candidates, potential, left, right,
        torch.Generator(device='cuda').manual_seed(377), segment_batch_size=3)
    assert torch.equal(first, second)
    assert (candidates.candidate_ids == first[..., None]).any(-1).all()
    assert (first[1, sorted({0, length//2, length-1})] == 1).all()


@cuda
def test_cuda_additive_joint_law_by_exhaustive_tiny_distribution():
    old.clear_count_cache()
    potential = potential_fixture(5, 'cuda')
    ids = torch.tensor([[[0, 1], [0, 1], [0, 1]]], device='cuda')
    unary = torch.tensor([[[-.3, -.8], [-.4, -.6], [-.9, -.2]]], device='cuda', dtype=torch.float64)
    left = torch.zeros(1, 3, 2, 64, device='cuda')
    right = torch.zeros_like(left)
    left[..., 0, 0], left[..., 1, 0] = 1., -1.
    right[..., 0, 0], right[..., 1, 0] = -1.3, 1.3
    base = CompactCandidates(ids, unary, 5)
    pair = dense_pair_reference(base, potential, left, right).double()[0]
    paths = torch.tensor(list(itertools.product(range(2), repeat=3)), device='cuda')
    scores = unary[0, torch.arange(3, device='cuda'), paths].sum(-1)
    scores += pair[0, paths[:, 0], paths[:, 1]] + pair[1, paths[:, 1], paths[:, 2]]
    expected = scores.softmax(0)
    samples = 8192
    candidates = CompactCandidates(ids.expand(samples, -1, -1).contiguous(),
                                    unary.expand(samples, -1, -1).contiguous(), 5)
    draws = sample_additive_count(candidates, potential,
        left.expand(samples, -1, -1, -1).contiguous(), right.expand(samples, -1, -1, -1).contiguous(),
        torch.Generator(device='cuda').manual_seed(741), segment_batch_size=samples)
    observed = torch.bincount((draws * torch.tensor([4, 2, 1], device='cuda')).sum(-1), minlength=8).double()/samples
    assert (observed-expected).abs().max() < .02


@cuda
@pytest.mark.parametrize('width',[500,1000,2000])
def test_cuda_large_candidate_caps_rank128_two_site_oracle(width):
    """Declared large caps need their own CUDA checks, not extrapolation."""
    old.clear_count_cache()
    candidates,potential=fixture(width,2,'cuda')
    candidates=CompactCandidates(candidates.candidate_ids[:1],candidates.log_unary[:1],width+2)
    rng=torch.Generator(device='cuda').manual_seed(45)
    left=torch.randn(1,2,width,128,device='cuda',generator=rng)*.03
    right=torch.randn(1,2,width,128,device='cuda',generator=rng)*.03
    pair=dense_pair_reference(candidates,potential,left,right)
    exact=marginals(candidates.log_unary.float().double(),pair.double(),return_edges=False).node
    actual=additive_count_marginals(candidates,potential,left,right).double()
    torch.testing.assert_close(actual,exact,atol=2e-5,rtol=3e-4)
    torch.testing.assert_close(actual.sum(-1),torch.ones_like(actual.sum(-1)),atol=1e-5,rtol=0.)

