import itertools
from types import SimpleNamespace

import pytest
import torch

from additive_chain.supported_training import prepare_supported_targets, supported_block_loss


def fixture():
    ids = torch.tensor([[[0, 1], [1, 2], [0, 2]]])
    unary = torch.tensor([[[-.2, -.7], [-.6, -.4], [-.8, -.1]]], dtype=torch.float64)
    candidates = SimpleNamespace(candidate_ids=ids, log_unary=unary, vocab_size=4)
    gold = torch.tensor([[0, 3, 2]])
    selected = torch.ones((1, 3), dtype=torch.bool)
    visible = torch.zeros_like(selected)
    pair = torch.tensor([[[[.3, -.2], [.1, .5]], [[-.2, .7], [.8, -.4]]]],
                        dtype=torch.float64, requires_grad=True)
    return candidates, gold, selected, visible, pair


def prepare(candidates, gold, selected, visible):
    return prepare_supported_targets(candidates, gold_tokens=gold,
        reveal_mask=selected, visible_mask=visible)


def test_original_support_and_missing_target_stays_latent():
    c, g, r, v, p = fixture()
    ids, unary = c.candidate_ids.clone(), c.log_unary.clone()
    target = prepare(c, g, r, v)
    assert target.support.candidate_ids is c.candidate_ids
    assert target.support.log_unary is c.log_unary
    assert torch.equal(ids, c.candidate_ids) and torch.equal(unary, c.log_unary)
    assert target.support.reveal_mask.tolist() == [[True, False, True]]
    assert target.missing_mask.tolist() == [[False, True, False]]
    assert target.diagnostics['gold_injected_tokens'] == 0
    assert not target.diagnostics['is_full_revealed_block_likelihood']
    # A different absent label at the middle position cannot affect the law.
    changed = g.clone(); changed[0, 1] = 0
    other = prepare(c, changed, r, v)
    assert torch.equal(supported_block_loss(target, p)['total_nll'],
                       supported_block_loss(other, p)['total_nll'])


@pytest.mark.parametrize('efficient', [False, True])
def test_exhaustive_joint_probability_and_gradient(efficient):
    c, g, r, v, p = fixture()
    target = prepare(c, g, r, v)
    paths = torch.tensor(list(itertools.product(range(2), repeat=3)))
    score = c.log_unary[0, torch.arange(3), paths].sum(-1)
    score = score + p[0, 0, paths[:, 0], paths[:, 1]] + p[0, 1, paths[:, 1], paths[:, 2]]
    matching = (paths[:, 0] == 0) & (paths[:, 2] == 1)
    expected = score.logsumexp(0) - score[matching].logsumexp(0)
    actual = supported_block_loss(target, p, memory_efficient=efficient)
    torch.testing.assert_close(actual['total_nll'], expected, atol=1e-14, rtol=1e-14)
    assert actual['should_update'] and actual['selected_targets'] == 3 and actual['supervised_targets'] == 2
    assert actual['per_original_selected_token_nll'] == actual['total_nll'] / 3
    assert actual['per_supported_target_nll'] == actual['total_nll'] / 2
    one, = torch.autograd.grad(actual['total_nll'], p, retain_graph=True)
    two, = torch.autograd.grad(expected, p)
    torch.testing.assert_close(one, two, atol=1e-14, rtol=1e-14)


def test_empty_supervision_skips_momentum_and_weight_decay():
    c, g, r, v, p = fixture()
    optimizer = torch.optim.AdamW([p], lr=.01, weight_decay=.1)
    supported_block_loss(prepare(c, g, r, v), p)['total_nll'].backward()
    optimizer.step()  # Establish nonzero optimizer momentum before the empty case.
    optimizer.zero_grad(set_to_none=True)
    before = p.detach().clone()
    state = {key: value.clone() for key, value in optimizer.state[p].items()}
    g[:] = 3
    result = supported_block_loss(prepare(c, g, r, v), p)
    assert not result['should_update'] and result['total_nll'] == 0
    gradient, = torch.autograd.grad(result['total_nll'], p)
    assert bool(gradient.eq(0).all())
    if result['should_update']:
        result['total_nll'].backward()
        optimizer.step()
    assert torch.equal(before, p)
    assert all(torch.equal(value, optimizer.state[p][key]) for key, value in state.items())


def test_invalid_selected_gold_and_visible_targets_raise():
    c, g, r, v, _ = fixture()
    g[0, 1] = 4
    with pytest.raises(ValueError, match='originally selected'):
        prepare(c, g, r, v)
    g[0, 1] = 3; v[0, 0] = True
    with pytest.raises(ValueError, match='Visible tokens'):
        prepare(c, g, r, v)
    with pytest.raises(ValueError, match='explicit visible'):
        prepare(c, g, r, None)


def test_unselected_gold_is_not_exposed():
    c, g, r, v, p = fixture()
    r[0, 1] = False; g[0, 1] = -999
    target = prepare(c, g, r, v)
    assert not bool(target.missing_mask.any())
    assert target.support.gold_slots.tolist() == [[0, -1, 1]]
    assert supported_block_loss(target, p)['supervised_targets'] == 2


def test_batch_keeps_original_reveal_denominator_including_empty_example():
    c, g, r, v, p = fixture()
    single = supported_block_loss(prepare(c, g, r, v), p)
    c.candidate_ids = c.candidate_ids.repeat(2, 1, 1)
    c.log_unary = c.log_unary.repeat(2, 1, 1)
    g = torch.cat([g, torch.full_like(g, 3)])
    target = prepare(c, g, r.repeat(2, 1), v.repeat(2, 1))
    batch = supported_block_loss(target, p.repeat(2, 1, 1, 1))
    torch.testing.assert_close(batch['total_nll'], single['total_nll'])
    assert batch['selected_targets'] == 6 and batch['supervised_targets'] == 2
    assert batch['per_original_selected_token_nll'] == batch['total_nll'] / 6


def test_padding_is_validated_even_when_no_target_can_be_supervised():
    c, g, r, v, _ = fixture()
    g[:] = 3
    c.log_unary[0, 0, 0] = -torch.inf  # ID still valid: invalid padding contract.
    with pytest.raises(ValueError, match='padding'):
        prepare(c, g, r, v)
