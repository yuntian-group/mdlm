"""Count-frequency controls preserve support and the exact PMI decomposition."""
import itertools

import pytest
import torch

from additive_chain.count_controls import add_count_unigram, smoothed_unigram_log_probs
from additive_chain.counts import SparseCountPotential


def counts_and_potential(mode, strength=1.0):
    counts = torch.tensor([[0., 7., 0., 2.], [3., 0., 4., 0.],
                           [0., 2., 0., 0.], [1., 0., 0., 0.]], dtype=torch.float64)
    keys = counts.flatten().nonzero().flatten()
    potential = SparseCountPotential.from_counts(
        counts.sum(1), counts.sum(0), keys, counts.flatten()[keys],
        mode=mode, strength=strength, smoothing=0.1)
    return counts, potential


def test_actual_marginal_not_just_backoff():
    counts, _ = counts_and_potential('conditional')
    n = counts.sum()
    left = (counts.sum(1) + 1) / (n + 4)
    right = (counts.sum(0) + 1) / (n + 4)
    joint = 0.9 * counts / n + 0.1 * left[:, None] * right[None, :]
    actual = smoothed_unigram_log_probs(counts.sum(0))
    torch.testing.assert_close(actual.exp(), joint.sum(0), atol=1e-15, rtol=1e-15)
    assert not torch.allclose(actual.exp(), right)
    torch.testing.assert_close(actual.logsumexp(0), actual.new_tensor(0.), atol=1e-15, rtol=0)


@pytest.mark.parametrize('mode', ['conditional', 'pmi'])
def test_fractional_counts_below_one_preserve_joint_and_marginal_normalization(mode):
    counts = torch.tensor([[.1, .2], [.05, .15]], dtype=torch.float64)
    n = counts.sum()
    left, right = counts.sum(1), counts.sum(0)
    joint = .9 * counts / n + .1 * ((left + 1) / (n + 2))[:, None] * ((right + 1) / (n + 2))[None, :]
    log_right = smoothed_unigram_log_probs(right)
    torch.testing.assert_close(log_right.exp(), joint.sum(0), atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(log_right.logsumexp(0), log_right.new_tensor(0.), atol=1e-15, rtol=0)
    expected = joint / joint.sum(1)[:, None]
    if mode == 'pmi':
        expected = expected / joint.sum(0)[None, :]
    potential = SparseCountPotential.from_counts(
        left, right, torch.arange(4), counts.flatten(), mode=mode)
    actual = potential.log_scores(torch.arange(2)[:, None], torch.arange(2)[None, :]).exp()
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=3e-15)


@pytest.mark.parametrize('strength', [0., .005, .1, 1., 2.])
def test_composed_full_joint_equals_conditional_not_just_edge_scores(strength):
    counts, conditional = counts_and_potential('conditional', strength)
    _, pmi = counts_and_potential('pmi', strength)
    log_marginal = smoothed_unigram_log_probs(counts.sum(0))
    unary = torch.tensor([[[.1, -.8, -.2, -.4], [-.1, -.5, .2, -.4],
                           [.3, -.1, -.8, -.6]]], dtype=torch.float64)
    ids = torch.arange(4)[None, None].expand(1, 3, 4)
    corrected = add_count_unigram(unary, ids, log_marginal,
                                   torch.tensor([[False, True, True]]), strength=strength)
    paths = torch.tensor(list(itertools.product(range(4), repeat=3)))
    def scores(unaries, pot):
        return (unaries[0, torch.arange(3)[None, :], paths].sum(-1)
                + pot.log_scores(paths[:, :-1], paths[:, 1:]).sum(-1))
    original = scores(unary, conditional)
    decomposed = scores(corrected, pmi)
    torch.testing.assert_close(original, decomposed, atol=8e-15, rtol=3e-15)
    torch.testing.assert_close(original.softmax(0), decomposed.softmax(0), atol=2e-15, rtol=3e-15)


def test_frequency_control_is_factorized_and_does_not_change_candidates():
    counts, _ = counts_and_potential('conditional')
    ids = torch.tensor([[[0, 2], [1, 3], [0, 1]]])
    before = ids.clone()
    unary = torch.tensor([[[.1, -.8], [-.1, -.5], [.3, -.1]]], dtype=torch.float64)
    corrected = add_count_unigram(unary, ids, smoothed_unigram_log_probs(counts.sum(0)),
                                   torch.tensor([[False, True, True]]), strength=.1)
    paths = torch.tensor(list(itertools.product(range(2), repeat=3)))
    flat = corrected[0, torch.arange(3)[None, :], paths].sum(-1).softmax(0)
    product = corrected.softmax(-1)[0, torch.arange(3)[None, :], paths].prod(-1)
    torch.testing.assert_close(flat, product, atol=1e-15, rtol=1e-15)
    assert torch.equal(ids, before)
    assert torch.equal(corrected[:, 0], unary[:, 0])


def test_incoming_boundary_padding_and_zero_identity():
    marginal = smoothed_unigram_log_probs(torch.tensor([8, 0, 2]))
    ids = torch.tensor([[[1, -1, 3], [2, 0, -1]]])
    unary = torch.tensor([[[0., -torch.inf, -torch.inf], [0., -.3, -torch.inf]]])
    # First local position can have a visible predecessor outside this component.
    edges = torch.tensor([[True, True]])
    assert add_count_unigram(unary, ids, marginal, edges, strength=0.) is unary
    actual = add_count_unigram(unary, ids, marginal, edges, strength=.1)
    assert actual.dtype == unary.dtype
    torch.testing.assert_close(actual[0, 0, 0], (.1 * marginal[1]).float())
    assert torch.equal(torch.isneginf(actual), torch.isneginf(unary))
    assert torch.equal(unary[0, 1, :2], torch.tensor([0., -.3]))


@pytest.mark.parametrize('smoothing', [.1, 1.])
def test_empty_counts_uniform_and_unseen_entries_remain_positive(smoothing):
    torch.testing.assert_close(smoothed_unigram_log_probs(torch.zeros(4), smoothing=smoothing).exp(),
                               torch.full((4,), .25, dtype=torch.float64))
    logp = smoothed_unigram_log_probs(torch.tensor([0., 2., 0., 7.]), smoothing=smoothing)
    assert torch.isfinite(logp).all()


@pytest.mark.parametrize('counts', [torch.tensor([-1., 2.]), torch.tensor([float('nan'), 1.]),
                                  torch.tensor([float('inf'), 1.]), torch.empty(0),
                                  torch.ones(2, 2), torch.tensor([True, False])])
def test_reject_invalid_counts(counts):
    with pytest.raises(ValueError):
        smoothed_unigram_log_probs(counts)


@pytest.mark.parametrize('value', [0., -1., 1.1, float('nan'), True])
def test_reject_invalid_smoothing(value):
    with pytest.raises(ValueError):
        smoothed_unigram_log_probs(torch.ones(3), smoothing=value)


def test_reject_wrong_mask_invalid_ids_and_unnormalized_scores():
    ids = torch.tensor([[[0, 1], [2, -1]]])
    unary = torch.zeros_like(ids, dtype=torch.float32)
    marginal = smoothed_unigram_log_probs(torch.tensor([1., 2., 3.]))
    edges = torch.tensor([[False, True]])
    with pytest.raises(ValueError, match='incoming-edge'):
        add_count_unigram(unary, ids, marginal, edges.long())
    with pytest.raises(ValueError, match='normalized'):
        add_count_unigram(unary, ids, marginal + 1., edges)
    with pytest.raises(ValueError, match='Candidate IDs'):
        add_count_unigram(unary, ids + 4, marginal, edges)
    with pytest.raises(ValueError, match='Strength'):
        add_count_unigram(unary, ids, marginal, edges, strength=-1.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual CUDA parity requires a GPU')
def test_cuda_matches_cpu_on_same_candidate_support():
    ids = torch.tensor([[[0, 2], [2, 1]]])
    unary = torch.tensor([[[0., -.3], [.1, -.5]]])
    edges = torch.tensor([[False, True]])
    marginal = smoothed_unigram_log_probs(torch.tensor([2., 4., 9.]))
    expected = add_count_unigram(unary, ids, marginal, edges, strength=.1)
    actual = add_count_unigram(unary.cuda(), ids.cuda(), marginal.cuda(), edges.cuda(), strength=.1)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-7, rtol=2e-7)
