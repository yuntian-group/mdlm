"""End-to-end nesting, training, and joint sampling through the public API."""
import itertools
from pathlib import Path
import re

import pytest
import torch

from additive_chain import (ContextualAdditiveHead, GlobalAdditiveHead,
    block_reveal_nll, compose_pair_scores, factors_to_pair_scores,
    log_partition, marginals, prepare_training_support, sample_tokens)


def fixture():
    ids = torch.tensor([[[0, 4], [0, 1], [1, 2], [2, 4]]])
    unary = torch.tensor([[[0., -torch.inf], [.1, -.2], [-.3, .2], [0., -torch.inf]]])
    table = torch.tensor([[.6, .2, .1, .1], [.1, .6, .2, .1],
                          [.2, .1, .6, .1], [.1, .2, .1, .6]]).log()
    safe = ids.clamp_max(3)
    count = table[safe[:, :-1, :, None], safe[:, 1:, None, :]]
    gold, selected = torch.tensor([[999, 1, 2, -100]]), torch.tensor([[False, True, True, False]])
    support = prepare_training_support(ids, unary, vocab_size=4, gold_tokens=gold,
                                       reveal_mask=selected)
    return support, count


def enumerate_distribution(unary, pair):
    paths = torch.tensor(list(itertools.product(range(unary.shape[-1]), repeat=unary.shape[0])))
    scores = unary[torch.arange(len(unary)), paths].sum(-1)
    for i in range(len(pair)):
        scores += pair[i, paths[:, i], paths[:, i+1]]
    return paths, scores.logsumexp(0), scores.softmax(0)


def test_zero_initialized_nested_models_preserve_scores_objective_and_joint_draws():
    support, count = fixture()
    global_head = GlobalAdditiveHead(4, rank=3)
    local = ContextualAdditiveHead(torch.randn(4, 5), hidden_size=6, rank=2)
    hidden = torch.randn(1, 4, 6)
    base = compose_pair_scores(count, count_strength=.3)
    plus_global = compose_pair_scores(count, count_strength=.3,
        global_factors=global_head(support.candidate_ids), global_strength=1.)
    plus_local = compose_pair_scores(count, count_strength=.3,
        global_factors=global_head(support.candidate_ids), global_strength=1.,
        contextual_factors=local(support.candidate_ids, hidden), contextual_strength=1.)
    expected_loss = block_reveal_nll(support.log_unary, base, support.gold_slots, support.reveal_mask)
    expected_draws = sample_tokens(support.candidate_ids, support.log_unary, base,
        vocab_size=4, num_samples=32, generator=torch.Generator().manual_seed(6))
    for scores in (plus_global, plus_local):
        assert torch.equal(scores, base)
        result = block_reveal_nll(support.log_unary, scores, support.gold_slots, support.reveal_mask)
        assert torch.equal(result.per_sequence, expected_loss.per_sequence)
        draws = sample_tokens(support.candidate_ids, support.log_unary, scores,
            vocab_size=4, num_samples=32, generator=torch.Generator().manual_seed(6))
        assert torch.equal(draws, expected_draws)
    assert expected_draws[:, 0, 0].eq(0).all() and expected_draws[:, 0, -1].eq(2).all()


def test_two_stages_train_as_residuals_and_freeze_preceding_parameters():
    support, count = fixture()
    head = GlobalAdditiveHead(4, rank=5)
    optimizer = torch.optim.SGD(head.parameters(), lr=.05)
    pair = compose_pair_scores(count, count_strength=.2,
        global_factors=head(support.candidate_ids), global_strength=1.)
    loss = block_reveal_nll(support.log_unary, pair, support.gold_slots, support.reveal_mask)
    loss.total_block_nll.backward()
    assert head.right_embedding.weight.grad.abs().sum() > 0
    optimizer.step()
    head.requires_grad_(False)
    head.zero_grad(set_to_none=True)
    old_state = {key: value.clone() for key, value in head.state_dict().items()}
    inherited = compose_pair_scores(count, count_strength=.2,
        global_factors=head(support.candidate_ids), global_strength=1.)
    embeddings = torch.randn(4, 5, requires_grad=True)
    hidden = torch.randn(1, 4, 6, requires_grad=True)
    local = ContextualAdditiveHead(embeddings, hidden_size=6, rank=3, context_size=8)
    expanded = compose_pair_scores(count, count_strength=.2,
        global_factors=head(support.candidate_ids), global_strength=1.,
        contextual_factors=local(support.candidate_ids, hidden), contextual_strength=1.)
    assert torch.equal(expanded, inherited)
    objective = block_reveal_nll(support.log_unary, expanded, support.gold_slots, support.reveal_mask)
    objective.total_block_nll.backward()
    assert local.right_token.weight.grad.abs().sum() > 0
    assert embeddings.grad is None and hidden.grad is None
    assert all(p.grad is None for p in head.parameters())
    torch.optim.SGD(local.parameters(), lr=.05).step()
    updated = compose_pair_scores(count, count_strength=.2,
        global_factors=head(support.candidate_ids), global_strength=1.,
        contextual_factors=local(support.candidate_ids, hidden), contextual_strength=1.)
    assert not torch.equal(updated, inherited)
    assert all(torch.equal(head.state_dict()[key], value) for key, value in old_state.items())


def test_nonzero_full_composition_objective_gradients_and_sampler_match_enumeration():
    support, count = fixture()
    count = count.double()
    unary = support.log_unary.double()
    left = torch.arange(24, dtype=torch.float64).reshape(1, 4, 2, 3).sin().requires_grad_()
    right = torch.arange(24, dtype=torch.float64).reshape(1, 4, 2, 3).cos().requires_grad_()
    local_left, local_right = left.square(), right.sin()
    pair = compose_pair_scores(count, count_strength=.4,
        global_factors=(left, right), global_strength=.7,
        contextual_factors=(local_left, local_right), contextual_strength=.2)
    manual = .4*count + .7*(left[:, :-1, :, None]*right[:, 1:, None]).sum(-1)
    manual += .2*(local_left[:, :-1, :, None]*local_right[:, 1:, None]).sum(-1)
    torch.testing.assert_close(pair, manual, atol=2e-15, rtol=2e-15)
    paths, z, probabilities = enumerate_distribution(unary[0], pair[0])
    torch.testing.assert_close(log_partition(unary, pair)[0], z, atol=2e-14, rtol=2e-14)
    chosen = (paths[:, support.reveal_mask[0]] == support.gold_slots[0, support.reveal_mask[0]]).all(-1)
    expected_loss = -probabilities[chosen].sum().log()
    objective = block_reveal_nll(unary, pair, support.gold_slots, support.reveal_mask)
    torch.testing.assert_close(objective.total_block_nll, expected_loss, atol=2e-14, rtol=2e-14)
    expected_gradient = torch.autograd.grad(expected_loss, (left, right), retain_graph=True)
    actual_gradient = torch.autograd.grad(objective.total_block_nll, (left, right))
    for actual, expected in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    draws = sample_tokens(support.candidate_ids, unary, pair, vocab_size=4,
                         num_samples=16000, generator=torch.Generator().manual_seed(71))
    # The two non-singleton sites have IDs [0,1] and [1,2].
    histogram = torch.bincount(2*draws[:, 0, 1] + draws[:, 0, 2]-1, minlength=4).double()/len(draws)
    expected = torch.stack([probabilities[(paths[:,1] == a) & (paths[:,2] == b)].sum()
                            for a,b in itertools.product(range(2), repeat=2)])
    assert (histogram-expected).abs().max() < .015


def test_zero_coefficients_recover_independent_law_and_single_position_works():
    support, count = fixture()
    factors = (torch.randn(1, 4, 2, 3), torch.randn(1, 4, 2, 3))
    pair = compose_pair_scores(count, count_strength=0., global_factors=factors, global_strength=0.)
    assert pair.eq(0).all()
    torch.testing.assert_close(marginals(support.log_unary, pair).node, support.log_unary.softmax(-1))
    single = factors_to_pair_scores((factors[0][:,:1], factors[1][:,:1]))
    assert single.shape == (1, 0, 2, 2)
    assert compose_pair_scores(single).shape == single.shape


def test_token_sampling_sequence_padding_and_invalid_ids():
    ids = torch.tensor([[[0, 1], [1, 2], [2, 2]]])
    unary = torch.tensor([[[.1, .2], [0., -torch.inf], [-torch.inf, -torch.inf]]])
    pair = torch.zeros(1, 2, 2, 2)
    draws = sample_tokens(ids, unary, pair, vocab_size=2, lengths=torch.tensor([2]), num_samples=7)
    assert draws[:,0,1].eq(1).all() and draws[:,0,2].eq(-1).all()
    broken = ids.clone(); broken[0,0,1] = 0
    with pytest.raises(ValueError, match='unique'):
        sample_tokens(broken, unary, pair, vocab_size=2, lengths=torch.tensor([2]))


def test_composition_validation_and_fp32_under_autocast():
    _, count = fixture()
    factors = (torch.randn(1, 4, 2, 3), torch.randn(1, 4, 2, 3))
    for kwargs in ({'count_strength':-.1}, {'global_strength':1.},
                   {'global_strength':True}, {'contextual_strength':float('nan')}):
        with pytest.raises(ValueError):
            compose_pair_scores(count, **kwargs)
    with pytest.raises(ValueError, match='layout'):
        compose_pair_scores(count, global_factors=(factors[0][:,:2],factors[1][:,:2]))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        value = compose_pair_scores(count, global_factors=factors, global_strength=.3)
    assert value.dtype == torch.float32


def test_documented_example_runs_without_private_imports_or_assets():
    text = (Path(__file__).resolve().parents[1]/'additive_chain/README.md').read_text()
    code, = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    namespace = {}
    exec(compile(code, 'additive_chain/README.md', 'exec'), namespace)
    assert namespace['tokens'].shape == (4, 1, 3)
    assert torch.isfinite(namespace['loss'].total_block_nll)
