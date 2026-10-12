"""CPU construction, probability, and gradient tests; no checkpoint or network access."""
import itertools

import pytest
import torch

from additive_chain.heads import (GlobalAdditiveHead, ContextualAdditiveHead,
                            concatenate_scaled_factors)


def pair_scores(factors):
    left, right = factors
    return torch.einsum('blik,bljk->blij', left[:, :-1], right[:, 1:])


def tiny_joint_loss(head, ids, hidden=None):
    pair = pair_scores(head.node_factors(ids, hidden))[0]
    paths = torch.tensor(list(itertools.product(range(ids.shape[-1]), repeat=ids.shape[1])))
    score = pair[torch.arange(ids.shape[1] - 1), paths[:, :-1], paths[:, 1:]].sum(-1)
    return score.logsumexp(0) - score[0]


def test_global_initialization_shape_zero_pairs_and_singleton_padding():
    head = GlobalAdditiveHead(5, rank=7)
    ids = torch.tensor([[[0, 1, 5], [2, -1, 5], [1, 3, 4]]])
    left, right = head.node_factors(ids)
    assert left.shape == right.shape == (1, 3, 3, 7)
    assert left.dtype == right.dtype == torch.float32
    assert left.is_contiguous() and right.is_contiguous()
    assert torch.count_nonzero(right) == 0
    assert torch.count_nonzero(pair_scores((left, right))) == 0
    assert torch.count_nonzero(left[ids.eq(-1) | ids.eq(5)]) == 0
    assert head.rank == 7


def test_global_joint_objective_has_nonzero_first_right_gradient():
    head = GlobalAdditiveHead(4, rank=8)
    ids = torch.tensor([[[0, 1], [1, 2], [2, 3]]])
    tiny_joint_loss(head, ids).backward()
    assert head.right_embedding.weight.grad.abs().sum() > 0
    assert head.left_embedding.weight.grad is not None
    assert head.left_embedding.weight.grad.abs().sum() == 0
    before = pair_scores(head.node_factors(ids)).detach()
    torch.optim.SGD(head.parameters(), lr=.05).step()
    assert not torch.equal(before, pair_scores(head.node_factors(ids)))


def test_global_ignores_context_and_has_no_score_rescaling():
    head = GlobalAdditiveHead(4, rank=3)
    with torch.no_grad():
        head.left_embedding.weight.copy_(torch.arange(12).reshape(4, 3))
        head.right_embedding.weight.copy_(torch.arange(12).reshape(4, 3) / 10.)
    ids = torch.tensor([[[0, 1], [2, 3]]])
    factors = head.node_factors(ids, torch.randn(1, 2, 99))
    assert torch.equal(pair_scores(factors)[0, 0],
                       head.left_embedding.weight[:2] @ head.right_embedding.weight[2:].T)
    assert all(torch.equal(a, b) for a, b in zip(factors, head.node_factors(ids)))


def test_context_initial_zero_nested_global_count_and_padding():
    head = ContextualAdditiveHead(torch.randn(5, 6), hidden_size=4, rank=7, context_size=9)
    ids = torch.tensor([[[0, 1, 5], [2, -1, 5], [1, 3, 4]]])
    factors = head.node_factors(ids, torch.randn(1, 3, 4))
    assert factors[0].shape == factors[1].shape == (1, 3, 3, 7)
    assert torch.count_nonzero(pair_scores(factors)) == 0
    assert torch.count_nonzero(factors[0][ids.eq(-1) | ids.eq(5)]) == 0
    count = torch.randn(1, 2, 3, 3)
    global_score = torch.randn_like(count)
    assert torch.equal(count + global_score + pair_scores(factors), count + global_score)


def test_context_first_step_gradient_and_frozen_inputs():
    source = torch.randn(5, 6, requires_grad=True)
    head = ContextualAdditiveHead(source, hidden_size=4, rank=7, context_size=9)
    saved = head.token_embeddings.clone()
    with torch.no_grad():
        source.add_(1.)
    assert torch.equal(head.token_embeddings, saved)
    assert not head.token_embeddings.requires_grad
    assert 'token_embeddings' not in dict(head.named_parameters())
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    ids = torch.tensor([[[0, 1], [1, 2], [2, 3]]])
    tiny_joint_loss(head, ids, hidden).backward()
    assert head.right_token.weight.grad.abs().sum() > 0
    assert head.right_shift.weight.grad.abs().sum() > 0
    assert head.left_token.weight.grad is not None and head.left_token.weight.grad.abs().sum() == 0
    assert source.grad is None and hidden.grad is None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


def test_local_context_changes_scores_after_one_update():
    head = ContextualAdditiveHead(torch.randn(5, 6), hidden_size=4, rank=7, context_size=9)
    ids = torch.tensor([[[0, 1], [1, 2], [2, 3]]])
    first = torch.randn(1, 3, 4)
    second = torch.randn(1, 3, 4)
    tiny_joint_loss(head, ids, first).backward()
    torch.optim.SGD(head.parameters(), lr=.1).step()
    a, b = pair_scores(head.node_factors(ids, first)), pair_scores(head.node_factors(ids, second))
    assert not torch.allclose(a, b, atol=1e-7, rtol=1e-7)


def test_external_coefficients_factor_concatenation_and_zero():
    a = (torch.randn(2, 4, 3, 5), torch.randn(2, 4, 3, 5))
    b = (torch.randn(2, 4, 3, 7), torch.randn(2, 4, 3, 7))
    combined = concatenate_scaled_factors([(1.25, a), (.4, b)])
    assert combined[0].shape == combined[1].shape == (2, 4, 3, 12)
    torch.testing.assert_close(pair_scores(combined), 1.25 * pair_scores(a) + .4 * pair_scores(b))
    zero = concatenate_scaled_factors([(0., a), (0., b)])
    assert torch.count_nonzero(pair_scores(zero)) == 0


def test_context_parameters_are_separate_from_frozen_global():
    global_head = GlobalAdditiveHead(5).requires_grad_(False)
    local_head = ContextualAdditiveHead(torch.randn(5, 6), rank=64)
    assert not set(map(id, global_head.parameters())) & set(map(id, local_head.parameters()))
    assert all(not p.requires_grad for p in global_head.parameters())
    assert all(p.requires_grad for p in local_head.parameters())
    assert global_head.rank == local_head.rank == 64


def test_input_and_coefficient_validation():
    head = GlobalAdditiveHead(4)
    with pytest.raises(ValueError):
        head.node_factors(torch.tensor([[[5]]]))
    with pytest.raises(ValueError):
        head.node_factors(torch.tensor([[[0.]]]))
    local = ContextualAdditiveHead(torch.randn(4, 6), hidden_size=4)
    with pytest.raises(ValueError):
        local.node_factors(torch.tensor([[[0]]]), torch.randn(1, 1, 6))
    factor = (torch.randn(1, 2, 2, 3), torch.randn(1, 2, 2, 3))
    for coefficient in (-1., float('nan'), True):
        with pytest.raises(ValueError):
            concatenate_scaled_factors([(coefficient, factor)])


def test_context_fp32_factors_under_cpu_backbone_autocast():
    head = ContextualAdditiveHead(torch.randn(4, 6), hidden_size=4, rank=3)
    ids, hidden = torch.tensor([[[0, 1], [2, 3]]]), torch.randn(1, 2, 4)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        left, right = head.node_factors(ids, hidden)
    assert left.dtype == right.dtype == torch.float32


def test_frozen_embedding_projection_precedes_candidate_gather():
    head = ContextualAdditiveHead(torch.randn(4, 6), hidden_size=4, rank=3)
    seen = []
    hook = head.left_token.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].shape))
    try:
        head.node_factors(torch.zeros(2, 9, 7, dtype=torch.long), torch.randn(2, 9, 4))
    finally:
        hook.remove()
    assert seen == [torch.Size([4, 6])]
