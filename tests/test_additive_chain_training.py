import itertools
from types import SimpleNamespace

import pytest
import torch

from additive_chain.chain import log_partition
from additive_chain.training import prepare_training_support, block_reveal_nll


def enumerate_block(unary, pair, gold, reveal):
    paths = torch.tensor(list(itertools.product(range(unary.shape[-1]), repeat=unary.shape[0])))
    scores = unary[torch.arange(unary.shape[0]), paths].sum(-1)
    for pos in range(pair.shape[0]):
        scores = scores + pair[pos, paths[:, pos], paths[:, pos+1]]
    matching = (paths[:, reveal] == gold[reveal]).all(-1)
    return scores.logsumexp(0) - scores[matching].logsumexp(0)


def fixture():
    unary = torch.tensor([[[.1, -.2], [-.3, .4], [.2, -.1]]], dtype=torch.float64)
    pair = torch.tensor([[[[1.6, -1.3], [-.9, 1.2]], [[1.1, -.8], [-1.2, 1.5]]]], dtype=torch.float64)
    slots = torch.tensor([[0, -1, 1]])
    reveal = torch.tensor([[True, False, True]])
    return unary, pair, slots, reveal


def test_two_partition_block_probability_preserves_hidden_bridge_and_gradients():
    unary, pair, gold, reveal = fixture()
    unary.requires_grad_(); pair.requires_grad_()
    expected = enumerate_block(unary[0], pair[0], gold[0], reveal[0])
    expected_grad = torch.autograd.grad(expected, (unary, pair), retain_graph=True)
    for efficient in (True, False):
        actual = block_reveal_nll(unary, pair, gold, reveal, memory_efficient=efficient)
        torch.testing.assert_close(actual.total_block_nll, expected, atol=2e-14, rtol=2e-14)
        assert actual.revealed_tokens == 2
        assert actual.per_revealed_token_nll == actual.total_block_nll / 2
        gradients = torch.autograd.grad(actual.total_block_nll, (unary, pair))
        for a, b in zip(gradients, expected_grad):
            torch.testing.assert_close(a, b, atol=2e-14, rtol=2e-14)
    # Hidden middle is integrated, not clamped to a hidden label.
    other_gold = gold.clone(); other_gold[0, 1] = 987654321
    assert torch.equal(block_reveal_nll(unary, pair, other_gold, reveal).per_sequence,
                       block_reveal_nll(unary, pair, gold, reveal).per_sequence)


def test_zero_additive_residual_and_empty_reveal_are_exact():
    unary, pair, gold, reveal = fixture()
    residual = torch.zeros_like(pair, requires_grad=True)
    original = block_reveal_nll(unary, pair, gold, reveal)
    adjusted = block_reveal_nll(unary, pair+residual, gold, reveal)
    assert torch.equal(original.per_sequence, adjusted.per_sequence)
    empty = block_reveal_nll(unary, pair+residual, gold, torch.zeros_like(reveal))
    assert empty.total_block_nll == 0 and empty.per_revealed_token_nll == 0
    gradient, = torch.autograd.grad(empty.total_block_nll, (residual,))
    assert gradient.eq(0).all()


def test_visible_singleton_and_varying_lengths_unchanged():
    unary, pair, gold, reveal = fixture()
    unary[0, 1] = torch.tensor([-torch.inf, 0.])
    result = block_reveal_nll(unary, pair, gold, reveal)
    expected = enumerate_block(unary[0], pair[0], gold[0], reveal[0])
    torch.testing.assert_close(result.total_block_nll, expected)
    padded_u = torch.cat((unary, torch.full((1, 1, 2), -torch.inf)), 1)
    padded_p = torch.cat((pair, torch.full((1, 1, 2, 2), -torch.inf)), 1)
    padded_r = torch.cat((reveal, torch.tensor([[False]])), 1)
    padded_g = torch.cat((gold, torch.tensor([[-1]])), 1)
    extended = block_reveal_nll(padded_u, padded_p, padded_g, padded_r, torch.tensor([3]))
    assert torch.equal(result.per_sequence, extended.per_sequence)


def support_fixture():
    ids = torch.tensor([[[0, 2], [1, 4], [0, 1]]])
    raw = torch.tensor([[[-.3, -.8, -.7, -1.5], [-.9, -.2, -.8, -1.3],
                         [-.4, -.6, -1., -.8]]], dtype=torch.float64)
    values = raw.gather(-1, ids.clamp_max(3))
    values[ids == 4] = -torch.inf
    values -= values.logsumexp(-1, keepdim=True)
    candidates = SimpleNamespace(candidate_ids=ids, log_unary=values, vocab_size=4)
    gold = torch.tensor([[3, 9999, 1]])
    reveal = torch.tensor([[True, False, True]])
    visible = torch.tensor([[False, True, False]])
    return candidates, raw, gold, reveal, visible


def test_strict_missing_targets_rejected_without_silently_dropping():
    candidates, raw, gold, reveal, visible = support_fixture()
    with pytest.raises(ValueError, match='1/2 selected'):
        prepare_training_support(candidates, gold_tokens=gold, reveal_mask=reveal, visible_mask=visible)
    with pytest.raises(ValueError, match='needs original full scores'):
        prepare_training_support(candidates, gold_tokens=gold, reveal_mask=reveal, missing='inject')
    with pytest.raises(ValueError, match='Observed positions'):
        prepare_training_support(candidates, gold_tokens=gold, reveal_mask=torch.ones_like(reveal),
                                 visible_mask=visible)
    unary, pair, slots, selected = fixture()
    slots[0, 0] = -1
    with pytest.raises(ValueError, match='in-support slot'):
        block_reveal_nll(unary, pair, slots, selected)


def test_training_only_injection_selected_positions_alignment_and_fraction():
    candidates, raw, gold, reveal, visible = support_fixture()
    original_ids, original_values = candidates.candidate_ids.clone(), candidates.log_unary.clone()
    result = prepare_training_support(candidates, gold_tokens=gold, reveal_mask=reveal,
                    visible_mask=visible, missing='inject', full_log_probs=raw)
    assert result.candidate_ids.shape == (1, 3, 3)
    assert result.candidate_ids[0, 0].tolist() == [0, 2, 3]
    torch.testing.assert_close(result.log_unary[0, 0], raw[0, 0, [0, 2, 3]].log_softmax(-1))
    assert torch.equal(result.candidate_ids[:, 1:, :2], original_ids[:, 1:])
    assert torch.equal(result.log_unary[:, 1:, :2], original_values[:, 1:])
    assert result.log_unary[:, 1:, 2].isneginf().all()
    assert result.gold_slots.tolist() == [[2, -1, 1]]
    assert result.diagnostics['original_out_of_support_fraction'] == .5
    assert result.diagnostics['gold_injected_tokens'] == 1
    assert result.diagnostics['retained_candidates_removed'] == 0
    assert torch.equal(candidates.candidate_ids, original_ids)
    assert torch.equal(candidates.log_unary, original_values)
    other_gold = gold.clone(); other_gold[0, 1] = -888
    other = prepare_training_support(candidates, gold_tokens=other_gold, reveal_mask=reveal,
                                      missing='inject', full_log_probs=raw)
    assert torch.equal(other.candidate_ids, result.candidate_ids)
    assert torch.equal(other.log_unary, result.log_unary)


def test_plain_input_padding_injection_same_scale_and_bad_full_scores():
    candidates, raw, gold, reveal, _ = support_fixture()
    reveal = torch.tensor([[False, True, False]])
    gold = torch.tensor([[-999, 3, -999]])
    same_scale = torch.tensor([[0., raw[0, 1, 3]-raw[0, 1, 1], 0.]], dtype=torch.float64)
    result = prepare_training_support(candidates.candidate_ids, candidates.log_unary, vocab_size=4,
        gold_tokens=gold, reveal_mask=reveal, missing='inject', gold_scores_in_unary_scale=same_scale)
    assert result.candidate_ids.shape[-1] == 2
    assert result.candidate_ids[0, 1].tolist() == [1, 3]
    torch.testing.assert_close(result.log_unary[0, 1], raw[0, 1, [1, 3]].log_softmax(-1))
    broken = raw.clone(); broken[0, 0, 2] += 1
    with pytest.raises(ValueError, match='do not match'):
        prepare_training_support(candidates, gold_tokens=torch.tensor([[3, -1, -1]]),
            reveal_mask=torch.tensor([[True, False, False]]), missing='inject', full_log_probs=broken)


def test_no_injection_preserves_tensor_identity_and_shared_pair_backward():
    candidates, raw, gold, reveal, _ = support_fixture()
    gold[0, 0] = 2
    support = prepare_training_support(candidates, gold_tokens=gold, reveal_mask=reveal)
    assert support.candidate_ids is candidates.candidate_ids and support.log_unary is candidates.log_unary
    unary, base, slots, selected = fixture()
    low_rank = torch.tensor([[.2, -.3], [.4, .1]], dtype=torch.float64, requires_grad=True)
    residual = low_rank @ low_rank.T
    pair = base + residual[None, None]
    value = block_reveal_nll(unary, pair, slots, selected).total_block_nll
    expected = enumerate_block(unary[0], pair[0], slots[0], selected[0])
    a, = torch.autograd.grad(value, (low_rank,), retain_graph=True)
    b, = torch.autograd.grad(expected, (low_rank,))
    torch.testing.assert_close(a, b, atol=2e-14, rtol=2e-14)
