"""Contextual embedding gathers preserve scores and gradients with padding."""
import copy

import pytest
import torch

from additive_chain.heads import ContextualAdditiveHead


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA gradient parity requires a GPU'))])
@pytest.mark.parametrize('width', [3, 500])
@pytest.mark.parametrize('objective', ['factor_gradients', 'adjacent_pairs'])
def test_contextual_embedding_matches_advanced_indexing_with_padding(
        monkeypatch, device, width, objective):
    generator = torch.Generator().manual_seed(337)
    vocabulary, length = 11, 7
    embeddings = torch.randn(vocabulary, 6, generator=generator, requires_grad=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        current = ContextualAdditiveHead(embeddings, hidden_size=4, rank=64, context_size=8)
    with torch.no_grad():
        current.right_token.weight.copy_(torch.randn(
            current.right_token.weight.shape, generator=generator) * .1)
        for shift in (current.left_shift, current.right_shift):
            shift.weight.copy_(torch.randn(shift.weight.shape, generator=generator) * .1)
            shift.bias.copy_(torch.randn(shift.bias.shape, generator=generator) * .1)
    current = current.to(device)
    reference = copy.deepcopy(current)
    ids = torch.full((2, length, width), vocabulary, dtype=torch.long, device=device)
    ids[1] = -1
    ids[..., :3] = torch.tensor([0, 3, 5], device=device)
    # Both supported padding sentinels, singleton visible boundaries, and a
    # visible middle separator. The last real row remains an ordinary token.
    for row, positions, sentinel in ((0, (0, 3, 6), vocabulary), (1, (0, 6), -1)):
        for position in positions:
            ids[row, position] = sentinel
            ids[row, position, 0] = 4
    ids[0, 1, 2] = vocabulary - 1
    hidden = torch.randn(2, length, 4, generator=generator).to(device).requires_grad_(True)
    actual = current(ids, hidden)
    calls = []

    def advanced_indexing(indices, table):
        calls.append(indices)
        return table[indices]

    # Restore only the former gather operation. All projections, masks,
    # gates, shifts, and autograd operations outside gathering stay identical.
    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.functional, 'embedding', advanced_indexing)
        expected = reference(ids, hidden)
    assert len(calls) == 2
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    valid = ids.ge(0) & ids.lt(vocabulary)
    for factors in (actual, expected):
        assert all(torch.count_nonzero(value[~valid]) == 0 for value in factors)
        assert all(value.dtype == torch.float32 and value.is_contiguous() for value in factors)
    if width == 500:
        assert (~valid).float().mean() > .99
    if objective == 'factor_gradients':
        upstream = [torch.randn(value.shape, generator=generator).to(device) for value in actual]
        torch.autograd.backward(actual, upstream)
        torch.autograd.backward(expected, upstream)
    else:
        # A separably weighted sum over adjacent pair scores includes the
        # singleton boundary edges without allocating a dense K-by-K tensor.
        weights = [torch.randn(2, length - 1, width, 1, generator=generator).to(device)
                   for _ in range(2)]
        def pair_loss(factors):
            left = (factors[0][:, :-1] * weights[0]).sum(-2)
            right = (factors[1][:, 1:] * weights[1]).sum(-2)
            return (left * right).sum()
        assert torch.equal(pair_loss(actual), pair_loss(expected))
        pair_loss(actual).backward()
        pair_loss(expected).backward()
    for (name, parameter), (other_name, other) in zip(
            current.named_parameters(), reference.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, other.grad, atol=3e-5, rtol=3e-4)
    assert hidden.grad is None and embeddings.grad is None
    assert not current.token_embeddings.requires_grad
