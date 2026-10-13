"""Small-state, long-chain precision regressions; no model or GPU required."""
import math

import pytest
import torch

from additive_chain import chain
from additive_chain.training import block_reveal_nll


def independent(length=1024, batch=1, offset=-10.):
    unary = torch.tensor([math.log(.8), math.log(.2)], dtype=torch.float32).repeat(batch, length, 1)
    pair = torch.full((batch, length - 1, 2, 2), offset, dtype=torch.float32)
    return unary, pair


@pytest.mark.parametrize('selected', [1, 32])
@pytest.mark.parametrize('efficient', [True, False])
def test_long_block_loss_accumulates_before_subtraction(selected, efficient):
    unary, pair = independent()
    reveal = torch.zeros((1, 1024), dtype=torch.bool)
    reveal[0, torch.arange(selected) * (1024 // selected)] = True
    gold = torch.zeros((1, 1024), dtype=torch.long)
    actual = block_reveal_nll(unary, pair, gold, reveal, memory_efficient=efficient)
    expected = block_reveal_nll(unary.double(), pair.double(), gold, reveal)
    assert actual.total_block_nll.dtype == torch.float64
    # Old FP32 accumulation errs by 4.87e-4 for one selected target and
    # 2.90e-3 for 32. Casting its final result to FP64 does not fix that.
    torch.testing.assert_close(actual.total_block_nll, expected.total_block_nll, atol=2e-5, rtol=0)
    torch.testing.assert_close(actual.per_revealed_token_nll,
                               expected.per_revealed_token_nll, atol=1e-6, rtol=0)


def test_constant_offsets_and_path_scores_on_long_chain():
    unary, pair = independent()
    states = torch.zeros((1, 1024), dtype=torch.long)
    base = chain.joint_nll(unary, pair.zero_(), states)
    for offset in (-1., -10., -100.):
        pair.fill_(offset)
        actual = chain.joint_nll(unary, pair, states)
        oracle = chain.joint_nll(unary.double(), pair.double(), states)
        assert chain.path_score(unary, pair, states).dtype == torch.float64
        assert torch.equal(chain.path_score(unary, pair, states),
                           chain.path_score(unary.double(), pair.double(), states))
        # Remaining error is the FP32 local message arithmetic, not a claim
        # of arbitrary-offset exactness or a full-FP64 dynamic program.
        torch.testing.assert_close(actual, oracle, atol=1.1e-3, rtol=0)
        torch.testing.assert_close(actual, base, atol=1.1e-3, rtol=0)


@pytest.mark.parametrize('efficient', [True, False])
def test_ragged_singletons_negative_infinity_and_batch_denominator(efficient):
    unary, pair = independent(length=64, batch=2)
    lengths = torch.tensor([64, 37])
    unary[1, 37:] = -torch.inf
    pair[1, 36:] = -torch.inf
    unary[:, 9, 1] = -torch.inf
    reveal = torch.zeros((2, 64), dtype=torch.bool)
    reveal[0, [1, 31]] = True
    reveal[1, [2, 17, 35]] = True
    gold = torch.zeros((2, 64), dtype=torch.long)
    unary.requires_grad_(); pair.requires_grad_()
    actual = block_reveal_nll(unary, pair, gold, reveal, lengths, memory_efficient=efficient)
    expected = block_reveal_nll(unary.double(), pair.double(), gold, reveal, lengths)
    assert actual.revealed_tokens == 5
    assert actual.per_revealed_token_nll == actual.total_block_nll / 5
    torch.testing.assert_close(actual.per_sequence, expected.per_sequence, atol=3e-6, rtol=0)
    gradients = torch.autograd.grad(actual.per_revealed_token_nll, (unary, pair))
    assert all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in gradients)
    assert gradients[0][1, 37:].eq(0).all() and gradients[1][1, 36:].eq(0).all()
    assert gradients[0][:, 9, 1].eq(0).all()


def test_long_custom_backward_matches_float64_without_promoting_dense_gradients():
    unary, pair = independent()
    unary.requires_grad_(); pair.requires_grad_()
    reveal = torch.zeros((1, 1024), dtype=torch.bool); reveal[0, 0] = True
    gold = torch.zeros((1, 1024), dtype=torch.long)
    actual = block_reveal_nll(unary, pair, gold, reveal)
    actual_grad = torch.autograd.grad(actual.total_block_nll, (unary, pair))
    u64 = unary.detach().double().requires_grad_()
    p64 = pair.detach().double().requires_grad_()
    reference = block_reveal_nll(u64, p64, gold, reveal)
    expected_grad = torch.autograd.grad(reference.total_block_nll, (u64, p64))
    for got, expected in zip(actual_grad, expected_grad):
        assert got.dtype == torch.float32
        torch.testing.assert_close(got.double(), expected, atol=1e-7, rtol=0)


def test_accumulator_change_leaves_messages_and_sampling_rng_exact(monkeypatch):
    unary, pair = independent(length=33)
    original = chain._forward
    actual_messages, total = original(unary, pair, torch.tensor([33]))
    assert actual_messages.dtype == torch.float32 and total.dtype == torch.float64
    def legacy_forward(u, p, lengths):
        alpha, scale = chain._normalize(u[:, 0])
        messages, logz = [alpha], scale
        for pos in range(1, u.shape[1]):
            alpha, scale = chain._normalize(u[:, pos] + chain._safe_logsumexp(
                alpha[:, :, None] + p[:, pos-1], -2))
            messages.append(alpha); logz = logz + scale
        return torch.stack(messages, 1), logz
    legacy_messages, _ = legacy_forward(unary, pair, None)
    assert torch.equal(actual_messages, legacy_messages)
    generator = torch.Generator().manual_seed(31)
    state = generator.get_state()
    fixed = chain.sample_ffbs(unary, pair, generator=generator)
    final = generator.get_state()
    generator.set_state(state)
    monkeypatch.setattr(chain, '_forward', legacy_forward)
    old = chain.sample_ffbs(unary, pair, generator=generator)
    assert torch.equal(fixed, old) and torch.equal(final, generator.get_state())
