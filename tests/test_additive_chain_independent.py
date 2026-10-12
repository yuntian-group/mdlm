import pytest
import torch

from additive_chain.independent import IndependentAdditiveHead, add_independent_residual, choose_width, parameter_count
from additive_chain.chain import log_partition, marginals
from additive_chain.training import block_reveal_nll


@pytest.mark.parametrize('embedded', [False, True])
def test_zero_recovery_parameter_count_and_detached_backbone(embedded):
    embedding = torch.randn(7, 5, requires_grad=True) if embedded else None
    head = IndependentAdditiveHead(7, 4, 3, frozen_token_embeddings=embedding)
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    ids = torch.tensor([[[0, 1, 2], [2, 3, 7], [4, 5, -1]]])
    scores = head(ids, hidden)
    assert torch.equal(scores, torch.zeros_like(scores))
    scores[0,0,0].backward()
    assert head.output.weight.grad.abs().sum() > 0
    assert hidden.grad is None
    assert embedding is None or embedding.grad is None
    assert head.active_parameter_count() == parameter_count(7,4,3,embedding_size=5 if embedded else None)


@pytest.mark.parametrize('embedded', [False, True])
def test_gather_matches_full_vocab_and_padding(embedded):
    head = IndependentAdditiveHead(7, 4, 3, frozen_token_embeddings=torch.randn(7,5) if embedded else None)
    with torch.no_grad():
        head.output.weight.normal_()
        if head.output.bias is not None:
            head.output.bias.normal_()
    h = torch.randn(2, 3, 4)
    ids = torch.tensor([[[0,1,7],[2,3,-1],[4,5,6]]]).expand(2,-1,-1)
    actual = head(ids,h)
    expected = head.full_scores(h).gather(-1,ids.clamp(0,6))
    valid = (ids>=0)&(ids<7)
    torch.testing.assert_close(actual[valid],expected[valid])
    assert torch.equal(actual[~valid],torch.zeros_like(actual[~valid]))


def test_global_capacity_match_is_nearest_and_active():
    target = 2*50258*64
    choice = choose_width(target,50258,768)
    assert choice['width'] == 125
    assert abs(choice['relative_parameter_difference']) < .001
    for w in (124,126):
        assert abs(parameter_count(50258,768,w)-target) >= abs(choice['actual_parameters']-target)


def test_selected_block_training_changes_only_unaries():
    head = IndependentAdditiveHead(3,4,5)
    hidden = torch.randn(1,3,4)
    ids = torch.arange(3).expand(1,3,3)
    unary = torch.randn(1,3,3)
    pair = torch.randn(1,2,3,3)
    saved_pair = pair.clone()
    selected = torch.tensor([[True,False,True]])
    slots = torch.tensor([[1,-1,2]])
    original = block_reveal_nll(unary,pair,slots,selected).total_block_nll
    updated = add_independent_residual(unary,ids,hidden,head)
    torch.testing.assert_close(updated,unary,rtol=0,atol=0)
    loss = block_reveal_nll(updated,pair,slots,selected).total_block_nll
    torch.testing.assert_close(loss,original,rtol=0,atol=0)
    optimizer = torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=0)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    after = block_reveal_nll(add_independent_residual(unary,ids,hidden,head),pair,slots,selected).total_block_nll
    assert after < original
    assert torch.equal(pair,saved_pair)


def test_no_pair_scores_means_factorized_distribution():
    head = IndependentAdditiveHead(3,4,5)
    with torch.no_grad():
        head.output.weight.normal_()
    ids = torch.arange(3).expand(1,4,3)
    h = torch.randn(1,4,4)
    unary = add_independent_residual(torch.randn(1,4,3),ids,h,head)
    pair = torch.zeros(1,3,3,3)
    torch.testing.assert_close(log_partition(unary,pair),unary.logsumexp(-1).sum(-1))
    # Increasing head capacity cannot introduce pair dependence by itself.
    probabilities = marginals(unary,pair,return_edges=False).node
    torch.testing.assert_close(probabilities,unary.softmax(-1))


def test_strength_zero_bypasses_head_and_retains_constraints():
    ids = torch.tensor([[[0,1,3]]])
    unary = torch.tensor([[[0.,-1.,-torch.inf]]])
    assert add_independent_residual(unary,ids,None,None,0) is unary
    head = IndependentAdditiveHead(3,4,5)
    out = add_independent_residual(unary,ids,torch.randn(1,1,4),head,1)
    assert torch.isneginf(out[0,0,2])


@pytest.mark.parametrize('bad',[-1,float('nan'),True])
def test_bad_strength_rejected(bad):
    with pytest.raises(ValueError):
        add_independent_residual(torch.zeros(1,1,1),torch.zeros(1,1,1,dtype=torch.long),None,None,bad)
