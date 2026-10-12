"""CPU exact-enumeration, gradient, and joint-sampling tests."""
import itertools

import pytest
import torch

from additive_chain.chain import log_partition, marginals, joint_nll, path_score, sample_ffbs


def fixture():
    unary = torch.tensor([[[.2, -.5, -.8], [-.2, .8, -.7], [.3, .4, -.2]],
                          [[-.1, .7, -.4], [-.6, .1, -.3], [-torch.inf]*3]], dtype=torch.float64)
    pair = torch.arange(36, dtype=torch.float64).reshape(2, 2, 3, 3).sin() * .6
    lengths = torch.tensor([3, 2])
    pair[1, 1] = -torch.inf
    return unary, pair, lengths


def enumerate_one(unary, pair, length):
    paths = torch.tensor(list(itertools.product(range(unary.shape[-1]), repeat=length)))
    scores = unary[torch.arange(length), paths].sum(-1)
    for pos in range(length-1):
        scores = scores + pair[pos, paths[:, pos], paths[:, pos+1]]
    probability = scores.softmax(0)
    node = torch.stack([torch.stack([probability[paths[:, pos] == state].sum()
        for state in range(unary.shape[-1])]) for pos in range(length)])
    edge = unary.new_zeros((length-1, unary.shape[-1], unary.shape[-1]))
    for pos in range(length-1):
        for a in range(unary.shape[-1]):
            for b in range(unary.shape[-1]):
                edge[pos, a, b] = probability[(paths[:, pos] == a) & (paths[:, pos+1] == b)].sum()
    return scores.logsumexp(0), node, edge, paths, probability


@pytest.mark.parametrize('constraint', ['none', 'singleton', 'hard_edges'])
def test_enumeration_left_right_orientation_lengths_and_support(constraint):
    unary, pair, lengths = fixture()
    if constraint == 'singleton':
        unary[0, 1] = -torch.inf
        unary[0, 1, 2] = .4
        unary[1, 0, 1:] = -torch.inf
    if constraint == 'hard_edges':
        pair[0, 0, :, 2] = -torch.inf  # unreachable destination, legal other paths
        pair[0, 1, 1, :] = -torch.inf
    actual = marginals(unary, pair, lengths)
    for row, length in enumerate(lengths.tolist()):
        z, node, edge, _, _ = enumerate_one(unary[row], pair[row], length)
        torch.testing.assert_close(actual.log_partition[row], z, atol=2e-14, rtol=2e-14)
        torch.testing.assert_close(actual.node[row, :length], node, atol=2e-14, rtol=2e-14)
        torch.testing.assert_close(actual.edge[row, :length-1], edge, atol=2e-14, rtol=2e-14)
        assert actual.node[row, length:].eq(0).all()
        assert actual.edge[row, length-1:].eq(0).all()
    torch.testing.assert_close(log_partition(unary, pair, lengths), actual.log_partition)
    assert marginals(unary, pair, lengths, return_edges=False).edge is None


def test_custom_backward_matches_enumeration_and_normal_autograd():
    unary, pair, lengths = fixture()
    unary.requires_grad_(); pair.requires_grad_()
    weights = torch.tensor([.3, -1.4], dtype=torch.float64)
    brute = torch.stack([enumerate_one(unary[i], pair[i], n)[0]
                         for i, n in enumerate(lengths.tolist())])
    expected = torch.autograd.grad((brute*weights).sum(), (unary, pair))
    for memory_efficient in (True, False):
        z = log_partition(unary, pair, lengths, memory_efficient=memory_efficient)
        actual = torch.autograd.grad((z*weights).sum(), (unary, pair))
        for a, b in zip(actual, expected):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, b, atol=2e-14, rtol=2e-14)


def test_gradcheck_first_and_second_order_and_differentiable_marginals():
    unary = torch.tensor([[[.2, -.4], [.8, .1]]], dtype=torch.float64, requires_grad=True)
    pair = torch.tensor([[[[.1, .7], [-.5, .3]]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(log_partition, (unary, pair))
    function = lambda u, p: log_partition(u, p, memory_efficient=False)
    assert torch.autograd.gradgradcheck(function, (unary, pair))
    assert torch.autograd.gradcheck(lambda u, p: marginals(u, p).node, (unary, pair))


def test_unreachable_states_have_zero_finite_gradients():
    unary = torch.tensor([[[.1, -.2], [-torch.inf, .3], [-.4, -.6]]], dtype=torch.float64)
    pair = torch.tensor([[[[.2, .4], [.3, .8]], [[-torch.inf, .1], [.9, -.2]]]], dtype=torch.float64)
    unary.requires_grad_(); pair.requires_grad_()
    for memory_efficient in (True, False):
        gradients = torch.autograd.grad(log_partition(unary, pair, memory_efficient=memory_efficient).sum(),
                                        (unary, pair))
        assert all(torch.isfinite(x).all() for x in gradients)
        assert gradients[0][0, 1, 0] == 0
        assert gradients[1][0, 0, :, 0].eq(0).all()
    gradient = torch.autograd.grad(marginals(unary, pair).node.square().sum(), (unary, pair))
    assert all(torch.isfinite(x).all() for x in gradient)


def test_additive_zero_residual_recovery_and_joint_nll_gradient():
    unary, base, lengths = fixture()
    residual = torch.zeros_like(base, requires_grad=True)
    states = torch.tensor([[0, 1, 2], [2, 0, -1]])
    score = path_score(unary, base, states, lengths)
    nll = joint_nll(unary, base+residual, states, lengths)
    assert torch.equal(nll, log_partition(unary, base, lengths)-score)
    grad, = torch.autograd.grad(nll.sum(), (residual,))
    expected = marginals(unary, base, lengths).edge
    for row, length in enumerate(lengths.tolist()):
        for pos in range(length-1):
            expected[row, pos, states[row, pos], states[row, pos+1]] -= 1.
    torch.testing.assert_close(grad, expected, atol=2e-14, rtol=2e-14)
    assert torch.equal(joint_nll(unary, base, states, lengths, reduction='sum'), nll.sum())


def test_ffbs_joint_empirical_law_not_product_of_marginals():
    # A strong agreement edge gives a very different joint from own marginals.
    unary = torch.tensor([[[.1, -.2], [-.3, .4], [.2, -.1]]], dtype=torch.float64)
    pair = torch.tensor([[[[1.6, -1.3], [-.9, 1.2]], [[1.1, -.8], [-1.2, 1.5]]]], dtype=torch.float64)
    _, node, _, paths, expected = enumerate_one(unary[0], pair[0], 3)
    samples = sample_ffbs(unary, pair, num_samples=24000, generator=torch.Generator().manual_seed(193))
    index = (samples[:, 0] * torch.tensor([4, 2, 1])).sum(-1)
    observed = torch.bincount(index, minlength=8).double() / samples.shape[0]
    assert (observed-expected).abs().max() < .012
    independent = node[torch.arange(3), paths].prod(-1)
    assert (observed-independent).abs().max() > .1


def test_singleton_sampling_padding_one_node_and_missing_gold():
    unary, pair, lengths = fixture()
    unary[0, 1] = -torch.inf; unary[0, 1, 2] = 0.
    draws = sample_ffbs(unary, pair, lengths, num_samples=19, generator=torch.Generator().manual_seed(5))
    assert draws[:, 0, 1].eq(2).all() and draws[:, 1, 2].eq(-1).all()
    gold = draws[0].clone(); gold[0, 1] = 0
    assert joint_nll(unary, pair, gold, lengths)[0].isposinf()
    single, no_edges = unary[:1, :1].clone().requires_grad_(), pair[:1, :0].clone().requires_grad_()
    m = marginals(single, no_edges)
    assert m.edge.shape == (1, 0, 3, 3)
    torch.testing.assert_close(m.node, single.softmax(-1))
    gradients = torch.autograd.grad(log_partition(single, no_edges).sum(), (single, no_edges))
    assert gradients[1].shape == no_edges.shape


def test_long_chain_stability_and_saved_tensor_memory():
    unary = torch.tensor([.1, -.3, .4], dtype=torch.float64).repeat(1, 1024, 1).requires_grad_()
    pair = torch.full((1, 1023, 3, 3), -130., dtype=torch.float64, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.shape) or t, lambda t: t):
        z = log_partition(unary, pair)
    assert saved == [unary.shape, pair.shape, torch.Size([1]), unary.shape]
    expected = unary.logsumexp(-1).sum(-1) - 130*1023
    torch.testing.assert_close(z, expected, atol=2e-8, rtol=0)
    m = marginals(unary.detach(), pair.detach())
    torch.testing.assert_close(m.node.sum(-1), torch.ones(1, 1024, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(m.node, unary.detach().softmax(-1), atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize('bad', ['no_path', 'nan', 'length_zero', 'dtype', 'shape'])
def test_invalid_inputs_rejected(bad):
    unary, pair, lengths = fixture()
    if bad == 'no_path': pair[0, 0] = -torch.inf
    if bad == 'nan': unary[0, 0, 0] = torch.nan
    if bad == 'length_zero': lengths[0] = 0
    if bad == 'dtype': pair = pair.float()
    if bad == 'shape': pair = pair[:, :, :2]
    with pytest.raises(ValueError):
        log_partition(unary, pair, lengths)
