import itertools
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from additive_chain.adaptation import count_pair_scores
from additive_chain.chain import marginals
from additive_chain.counts import SparseCountPotential
from additive_chain.generation import ChainDecoder
from additive_chain.heads import ContextualAdditiveHead, GlobalAdditiveHead
from additive_chain.iwslt_residual_objective import (
    TargetCorruption, draw_target_corruption, loss_on_target_corruption, residual_objective,
)
from additive_chain.training import block_reveal_nll
from additive_chain.translation import TranslationBatch


def batch_fixture():
    # Source prefixes have lengths 2 and 3; targets include toy EOS=3.
    return TranslationBatch(
        tokens=torch.tensor([[3, 2, 3, 2, 1, 3], [3, 0, 3, 2, 3, 3]]),
        valid=torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=torch.bool),
        target=torch.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0]], dtype=torch.bool),
        starts=[2, 3], lengths=[4, 2])


def corruption_fixture(batch):
    noisy = batch.tokens.clone()
    noisy[0, [2, 4, 5]] = 4
    noisy[1, [3, 4]] = 4
    reveal = torch.zeros_like(batch.target)
    reveal[0, [2, 5]] = True
    reveal[1, 3] = True
    return TargetCorruption(noisy, reveal, torch.full((2, 1), .5),
                            torch.tensor([2, 4]), torch.tensor([1, 2]))


class TinyFrozen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([1.2, .7, -.4, -1.3, -torch.inf]))
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.calls = []
        self.eval()

    def forward(self, tokens, valid):
        self.calls.append((tokens.clone(), valid.clone()))
        # Source conditioning and mask-dependent features are intentional.
        feature = tokens.float()[..., None] + torch.arange(6)[None, None]
        feature = feature * self.scale
        return SimpleNamespace(log_probs=self.logits.expand(*tokens.shape, 5), hidden=feature)


def count_statistics():
    keys = torch.tensor([1, 7, 13, 15])
    counts = torch.tensor([3., 2., 4., 1.], dtype=torch.float64)
    left, right = torch.zeros(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64)
    left.index_add_(0, keys // 5, counts)
    right.index_add_(0, keys % 5, counts)
    return dict(left_counts=left, right_counts=right, pair_keys=keys, pair_counts=counts)


def potential():
    return SparseCountPotential.from_counts(**count_statistics(), strength=.2)


def compute(batch=None, corruption=None, **overrides):
    batch = batch_fixture() if batch is None else batch
    corruption = corruption_fixture(batch) if corruption is None else corruption
    args = dict(frozen=TinyFrozen(), count_potential=potential(),
                head=GlobalAdditiveHead(5, rank=2), mask_id=4, top_k=2)
    args.update(overrides)
    return loss_on_target_corruption(batch, corruption, **args)


def exhaustive(unary, pair, gold_slots, selected):
    paths = torch.tensor(list(itertools.product(range(unary.shape[-1]), repeat=len(unary))))
    scores = unary[torch.arange(len(unary)), paths].sum(-1)
    for i in range(len(pair)):
        scores = scores + pair[i, paths[:, i], paths[:, i + 1]]
    chosen = (paths[:, selected] == gold_slots[selected]).all(-1)
    return scores.logsumexp(0) - scores[chosen].logsumexp(0)


def test_schedule_matches_draw_order_and_never_masks_source_padding():
    batch = batch_fixture()
    g1, g2 = torch.Generator().manual_seed(31), torch.Generator().manual_seed(31)
    actual = draw_target_corruption(batch, g1, mask_id=4)
    schedules = torch.tensor([2, 4, 8])[torch.randint(3, (2,), generator=g2)]
    steps = (torch.rand((2,), generator=g2) * schedules).long() + 1
    time = (steps.float() / schedules)[:, None]
    masked = (torch.rand(batch.tokens.shape, generator=g2) < time) & batch.target
    revealed = masked & (torch.rand(batch.tokens.shape, generator=g2) < 1 / steps[:, None])
    assert torch.equal(actual.tokens, batch.tokens.masked_fill(masked, 4))
    assert torch.equal(actual.reveal_mask, revealed)
    assert torch.equal(actual.time, time)
    assert torch.equal(actual.schedules, schedules) and torch.equal(actual.steps, steps)
    assert torch.equal(g1.get_state(), g2.get_state())
    assert torch.equal(actual.tokens[~batch.target], batch.tokens[~batch.target])
    full = draw_target_corruption(batch, g1, schedule_steps=(1,), mask_id=4)
    assert torch.equal(full.reveal_mask, batch.target)
    assert full.tokens[batch.target].eq(4).all()  # Includes target EOS.
    assert not full.reveal_mask[~batch.valid].any()


def test_target_only_gold_injection_one_forward_and_latent_unselected_gold():
    batch = batch_fixture()
    corruption = corruption_fixture(batch)
    model = TinyFrozen()
    result = compute(batch, corruption, frozen=model)
    assert len(model.calls) == 1
    assert torch.equal(model.calls[0][0], corruption.tokens)
    assert torch.equal(model.calls[0][1], batch.valid)
    assert result.target_lengths.tolist() == [4, 2]
    assert result.support.candidate_ids.shape[:2] == (2, 4)
    assert result.pair_scores.shape[1] == 3  # Target edges, never prefix edges.
    assert result.support.candidate_ids[0, 0].tolist() == [0, 1, 3]
    assert result.support.candidate_ids[0, 1].tolist() == [2, 5, 5]
    assert result.support.candidate_ids[0, 2].tolist() == [0, 1, 5]
    assert result.support.candidate_ids[1, 2].tolist() == [0, 5, 5]
    assert result.metrics["gold_injected_tokens"] == 3
    assert result.selected_tokens == 3
    assert result.metrics["source_to_target_edges"] == 0
    other = replace(batch, tokens=batch.tokens.clone())
    other.tokens[0, 4] = 2  # Hidden unselected target cannot affect loss/support.
    other.tokens[1, 4] = 1
    second = compute(other, corruption)
    assert torch.equal(result.support.candidate_ids, second.support.candidate_ids)
    assert torch.equal(result.unary_scores, second.unary_scores)
    assert torch.equal(result.total_nll, second.total_nll)


def test_exact_enumeration_and_gradients_with_variable_target_lengths():
    head = GlobalAdditiveHead(5, 2)
    with torch.no_grad():
        head.left_embedding.weight.copy_(torch.arange(10).reshape(5, 2).sin())
        head.right_embedding.weight.copy_(torch.arange(10).reshape(5, 2).cos() * .2)
    model = TinyFrozen()
    result = compute(head=head, frozen=model)
    expected = sum(exhaustive(result.unary_scores[i, :n], result.pair_scores[i, :n-1],
                              result.support.gold_slots[i, :n], result.support.reveal_mask[i, :n])
                   for i, n in enumerate([4, 2]))
    torch.testing.assert_close(result.total_nll, expected.double(), atol=3e-6, rtol=2e-6)
    actual = torch.autograd.grad(result.total_nll, tuple(head.parameters()), retain_graph=True)
    reference = torch.autograd.grad(expected, tuple(head.parameters()))
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=2e-6)
    assert model.logits.grad is None and model.scale.grad is None
    torch.testing.assert_close(result.loss, result.total_nll / 3)


def test_zero_residual_recovers_count_energy_and_temperature_matches_decoder():
    result = compute(top_k=4, nucleus_p=1.)
    count = count_pair_scores(result.support, potential())
    assert torch.equal(result.pair_scores, count / .85)
    expected = block_reveal_nll(result.support.log_unary.float() / .85, count / .85,
        result.support.gold_slots, result.support.reveal_mask, result.target_lengths)
    assert torch.equal(result.total_nll, expected.total_block_nll)

    decoder = ChainDecoder(mask_id=4, count_statistics=count_statistics(),
                           count_strength=.2, top_k=4, nucleus_p=1., temperature=.85)
    batch, corruption = batch_fixture(), corruption_fixture(batch_fixture())
    raw = TinyFrozen()(corruption.tokens, batch.valid).log_probs[0:1, 2:]
    packet = decoder._candidates(raw, corruption.tokens[0:1, 2:])
    # Generation independently scales the count exponent and normalizes
    # tempered unary rows. Both changes must leave the same chain distribution.
    generated_pair = count_pair_scores(packet, decoder.potential)
    trained = marginals(result.unary_scores[0:1], result.pair_scores[0:1]).node
    generated = marginals(packet.log_unary.float(), generated_pair).node
    torch.testing.assert_close(trained, generated, atol=3e-7, rtol=3e-6)


def test_contextual_head_uses_same_corrupted_hidden_and_frozen_global():
    global_head = GlobalAdditiveHead(5, 2)
    with torch.no_grad():
        global_head.right_embedding.weight.fill_(.2)
    global_head.eval().requires_grad_(False)
    contextual = ContextualAdditiveHead(torch.randn(5, 3), hidden_size=6, rank=2, context_size=4)
    model = TinyFrozen()
    captured = []
    hook = contextual.register_forward_pre_hook(lambda _m, args: captured.append(args[1].clone()))
    result = compute(frozen=model, head=contextual, global_head=global_head, global_strength=1.)
    hook.remove()
    assert len(model.calls) == len(captured) == 1
    expected = result.corruption.tokens[0, 2:].float()[:, None] + torch.arange(6)
    assert torch.equal(captured[0][0], expected)
    assert captured[0][1, 2:].eq(0).all()
    result.total_nll.backward()
    assert contextual.right_token.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in global_head.parameters())
    assert all(p.grad is None for p in model.parameters())
    base = compute(head=global_head)
    torch.testing.assert_close(result.total_nll.detach(), base.total_nll.detach())


def test_zero_head_has_nonzero_first_step_gradient_and_no_padding_contribution():
    head = GlobalAdditiveHead(5, 2)
    result = compute(head=head)
    result.total_nll.backward()
    assert head.right_embedding.weight.grad.abs().sum() > 0
    assert head.left_embedding.weight.grad.eq(0).all()
    pair = result.pair_scores.detach().clone().requires_grad_()
    unary = result.unary_scores.detach().clone().requires_grad_()
    loss = block_reveal_nll(unary, pair, result.support.gold_slots,
                           result.support.reveal_mask, result.target_lengths)
    du, dp = torch.autograd.grad(loss.total_block_nll, (unary, pair))
    assert du[1, 2:].eq(0).all() and dp[1, 1:].eq(0).all()
    changed = pair.detach().clone()
    changed[1, 1:] = 999
    repeated = block_reveal_nll(unary, changed, result.support.gold_slots,
                               result.support.reveal_mask, result.target_lengths)
    assert torch.equal(loss.total_block_nll, repeated.total_block_nll)


def test_empty_reveal_returns_no_loss_and_skips_backbone():
    batch = batch_fixture()
    corruption = corruption_fixture(batch)
    corruption = replace(corruption, reveal_mask=torch.zeros_like(batch.target))
    model = TinyFrozen()
    result = compute(batch, corruption, frozen=model)
    assert result.loss is result.total_nll is result.support is result.pair_scores is None
    assert result.selected_tokens == 0 and not model.calls


def test_optional_count_table_matches_sparse_lookup_scores_loss_and_gradients():
    p = potential()
    ids = torch.arange(5)
    table = (p.log_scores(ids[:, None], ids[None, :]) - p.log_scale).float()
    head = GlobalAdditiveHead(5, 2)
    sparse = compute(head=head)
    dense = compute(head=head, count_score_table=table)
    assert torch.equal(sparse.pair_scores, dense.pair_scores)
    assert torch.equal(sparse.total_nll, dense.total_nll)
    grad_sparse = torch.autograd.grad(sparse.total_nll, tuple(head.parameters()))
    grad_dense = torch.autograd.grad(dense.total_nll, tuple(head.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(grad_sparse, grad_dense))
    with pytest.raises(ValueError, match="frozen FP32"):
        compute(count_score_table=table.requires_grad_())


def test_wrapper_draws_once_and_preserves_rng():
    batch = batch_fixture()
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    result = residual_objective(batch, generator=g1, frozen=TinyFrozen(),
        count_potential=potential(), head=GlobalAdditiveHead(5, 2),
        mask_id=4, top_k=2, schedule_steps=(1,))
    direct = draw_target_corruption(batch, g2, mask_id=4, schedule_steps=(1,))
    assert torch.equal(result.corruption.tokens, direct.tokens)
    assert torch.equal(g1.get_state(), g2.get_state())
    assert result.selected_tokens == 6


@pytest.mark.parametrize("bad", ["source_mask", "source_reveal", "padding_reveal", "bad_time",
                                "bad_layout", "training_backbone", "global_not_frozen"])
def test_rejects_leakage_and_inconsistent_inputs(bad):
    batch = batch_fixture()
    corruption = corruption_fixture(batch)
    overrides = {}
    if bad == "source_mask":
        corruption.tokens[0, 0] = 4
    elif bad == "source_reveal":
        corruption.reveal_mask[0, 0] = True
    elif bad == "padding_reveal":
        corruption.reveal_mask[1, 5] = True
    elif bad == "bad_time":
        corruption.time[0, 0] = .2
    elif bad == "bad_layout":
        batch.target[0, 0] = True
    elif bad == "training_backbone":
        overrides["frozen"] = TinyFrozen().train()
    else:
        overrides.update(head=ContextualAdditiveHead(torch.randn(5, 3), hidden_size=6, rank=2),
                         global_head=GlobalAdditiveHead(5, 2), global_strength=1.)
    with pytest.raises(ValueError):
        compute(batch, corruption, **overrides)
