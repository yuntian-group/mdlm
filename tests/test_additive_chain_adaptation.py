import itertools
from types import SimpleNamespace

import pytest
import torch

from additive_chain.adaptation import adaptation_step, count_pair_scores, draw_corruption, loss_on_corruption
from additive_chain.counts import SparseCountPotential
from additive_chain.heads import ContextualAdditiveHead, GlobalAdditiveHead
from additive_chain.training import block_reveal_nll


def test_union_annotations_are_postponed_for_python39_imports():
    from additive_chain.adaptation import AdaptationStep
    assert AdaptationStep.__annotations__['objective'] == 'BlockRevealLoss | None'
    assert AdaptationStep.__annotations__['pair_scores'] == 'torch.Tensor | None'


class TinyFrozen:
    mask_id = 4

    def __init__(self):
        self.calls = []
        self.logits = torch.tensor([1.2, .7, -.4, -1.3, -torch.inf], requires_grad=True)
        self.features = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6).requires_grad_()

    def forward(self, tokens, time):
        self.calls.append((tokens.clone(), time.clone()))
        return SimpleNamespace(log_probs=self.logits.expand(*tokens.shape, 5),
                               hidden=self.features.expand(tokens.shape[0], -1, -1))


def potential(strength=.1):
    keys = torch.tensor([1, 7, 13, 15])
    counts = torch.tensor([3., 2., 4., 1.], dtype=torch.float64)
    left, right = torch.zeros(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64)
    left.index_add_(0, keys // 5, counts); right.index_add_(0, keys % 5, counts)
    return SparseCountPotential.from_counts(left, right, keys, counts, strength=strength)


def fixture():
    clean = torch.tensor([[3, 2, 1, 3]])
    corrupted = torch.tensor([[4, 2, 4, 4]])
    reveal = torch.tensor([[True, False, False, True]])
    return clean, corrupted, reveal, torch.tensor([[.5]])


def enumerate_loss(unary, pair, gold, reveal):
    paths = torch.tensor(list(itertools.product(range(unary.shape[-1]), repeat=unary.shape[0])))
    scores = unary[torch.arange(len(unary)), paths].sum(-1)
    for i in range(len(pair)):
        scores = scores + pair[i, paths[:, i], paths[:, i + 1]]
    selected = (paths[:, reveal] == gold[reveal]).all(-1)
    return scores.logsumexp(0) - scores[selected].logsumexp(0)


def test_draw_matches_exact_integer_then_two_uniform_call_order_and_rng():
    clean = torch.arange(24).reshape(3, 8) % 4
    g1 = torch.Generator().manual_seed(27); g2 = torch.Generator().manual_seed(27)
    actual = draw_corruption(clean, 4, g1, schedule_steps=32)
    step = torch.randint(1, 33, (3, 1), generator=g2)
    time = step.float() / 32
    masked = torch.rand(clean.shape, generator=g2) < time
    selected = masked & (torch.rand(clean.shape, generator=g2) < 1. / step)
    expected = (torch.where(masked, 4, clean), selected, time)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(g1.get_state(), g2.get_state())
    assert not (actual[1] & actual[0].ne(4)).any()
    # A one-step schedule masks/reveals every position, including the first.
    masked, reveal, time = draw_corruption(clean, 4, g1, schedule_steps=1)
    assert masked.eq(4).all() and reveal.all() and time.eq(1).all()


def test_one_masked_forward_selected_only_augmentation_and_unrevealed_bridge():
    clean, corrupted, reveal, time = fixture()
    frozen, head = TinyFrozen(), GlobalAdditiveHead(5, rank=2)
    result = loss_on_corruption(clean, corrupted, reveal, time, frozen=frozen,
        count_potential=potential(), head=head, top_k=2, nucleus_p=.9)
    assert len(frozen.calls) == 1 and torch.equal(frozen.calls[0][0], corrupted)
    assert torch.equal(frozen.calls[0][1], time)
    assert result.support.diagnostics['gold_injected_tokens'] == 2
    assert result.support.diagnostics['original_out_of_support_fraction'] == 1
    assert result.support.candidate_ids[0, 0].tolist() == [0, 1, 3]
    assert result.support.candidate_ids[0, 1].tolist() == [2, 5, 5]
    assert result.support.candidate_ids[0, 2].tolist() == [0, 1, 5]
    assert result.support.gold_slots.tolist() == [[2, -1, -1, 2]]
    expected = enumerate_loss(result.support.log_unary[0].float(), result.pair_scores[0],
                              result.support.gold_slots[0], reveal[0])
    torch.testing.assert_close(result.loss, expected / 2, atol=2e-6, rtol=2e-6)
    other = clean.clone(); other[0, 2] = 3
    second = loss_on_corruption(other, corrupted, reveal, time, frozen=frozen,
        count_potential=potential(), head=head, top_k=2)
    assert torch.equal(second.support.candidate_ids, result.support.candidate_ids)
    assert torch.equal(second.support.log_unary, result.support.log_unary)
    assert torch.equal(second.loss, result.loss)
    result.loss.backward()
    assert head.right_embedding.weight.grad.abs().sum() > 0
    assert head.left_embedding.weight.grad.eq(0).all()
    assert frozen.logits.grad is None and frozen.features.grad is None


def test_dense_count_lookup_chunks_match_scaled_law_and_zero_head_exact_recovery():
    clean, corrupted, reveal, time = fixture()
    p = potential()
    result = loss_on_corruption(clean, corrupted, reveal, time, frozen=TinyFrozen(),
                               count_potential=p, head=GlobalAdditiveHead(5, 3), top_k=2)
    a = count_pair_scores(result.support, p, edge_chunk=1)
    b = count_pair_scores(result.support, p, edge_chunk=99)
    assert torch.equal(a, b) and torch.equal(a, result.pair_scores)
    ids = result.support.candidate_ids.masked_fill(result.support.candidate_ids.eq(5), 0)
    expected = p.log_scores(ids[:, :-1, :, None], ids[:, 1:, None, :]) - p.log_scale
    torch.testing.assert_close(a.double(), expected, atol=2e-7, rtol=2e-7)
    original = block_reveal_nll(result.support.log_unary.float(), a,
                               result.support.gold_slots, reveal)
    assert torch.equal(original.per_sequence, result.objective.per_sequence)


def test_common_count_log_scale_is_omitted_and_zero_strength_recovers_independence():
    from dataclasses import replace
    clean, corrupted, reveal, time = fixture()
    head = GlobalAdditiveHead(5, 2)
    p = potential(0.)
    result = loss_on_corruption(clean, corrupted, reveal, time, frozen=TinyFrozen(),
        count_potential=p, head=head, top_k=2)
    assert result.pair_scores.eq(0).all()
    offset = count_pair_scores(result.support, replace(p, log_scale=p.log_scale + 17.))
    assert torch.equal(offset, result.pair_scores)
    selected = result.support.log_unary.float().gather(-1,
        result.support.gold_slots.clamp_min(0)[..., None]).squeeze(-1)
    expected = -selected[reveal].sum() / reveal.sum()
    torch.testing.assert_close(result.loss, expected, atol=2e-6, rtol=2e-6)


def test_nonzero_head_gradient_matches_independent_enumeration():
    clean, corrupted, reveal, time = fixture()
    head = GlobalAdditiveHead(5, 2)
    with torch.no_grad():
        head.left_embedding.weight.copy_(torch.arange(10).reshape(5, 2).sin())
        head.right_embedding.weight.copy_(torch.arange(10).reshape(5, 2).cos() * .2)
    result = loss_on_corruption(clean, corrupted, reveal, time, frozen=TinyFrozen(),
                               count_potential=potential(), head=head, top_k=2)
    expected = enumerate_loss(result.support.log_unary[0].float(), result.pair_scores[0],
                              result.support.gold_slots[0], reveal[0]) / 2
    actual = torch.autograd.grad(result.loss, tuple(head.parameters()), retain_graph=True)
    reference = torch.autograd.grad(expected, tuple(head.parameters()))
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


def test_zero_context_preserves_frozen_predecessor_and_uses_same_masked_features():
    clean, corrupted, reveal, time = fixture()
    global_head = GlobalAdditiveHead(5, 3)
    with torch.no_grad():
        global_head.right_embedding.weight.fill_(.25)
    global_head.eval().requires_grad_(False)
    old = {k: v.clone() for k, v in global_head.state_dict().items()}
    local = ContextualAdditiveHead(torch.randn(5, 4), hidden_size=6, rank=2, context_size=3)
    frozen = TinyFrozen()
    seen = []
    hook = local.register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[1].clone()))
    result = loss_on_corruption(clean, corrupted, reveal, time, frozen=frozen,
        count_potential=potential(), head=local, global_head=global_head,
        global_strength=.25, top_k=2)
    hook.remove()
    assert len(frozen.calls) == 1 and len(seen) == 1
    assert torch.equal(seen[0], frozen.features.detach())
    left, right = global_head(result.support.candidate_ids)
    predecessor = count_pair_scores(result.support, potential()) + .25 * (
        left[:, :-1] @ right[:, 1:].transpose(-1, -2))
    assert torch.equal(result.pair_scores, predecessor)
    expected = block_reveal_nll(result.support.log_unary.float(), predecessor,
                               result.support.gold_slots, reveal)
    assert torch.equal(result.loss, expected.per_revealed_token_nll)
    result.loss.backward()
    assert local.right_token.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in global_head.parameters())
    assert all(torch.equal(old[k], v) for k, v in global_head.state_dict().items())
    assert frozen.features.grad is None


def test_empty_reveal_skips_backbone_and_does_not_supply_an_optimizer_loss():
    clean, corrupted, reveal, time = fixture()
    frozen = TinyFrozen()
    result = loss_on_corruption(clean, corrupted, torch.zeros_like(reveal), time,
        frozen=frozen, count_potential=potential(), head=GlobalAdditiveHead(5, 2))
    assert result.loss is None and result.support is None and result.pair_scores is None
    assert not frozen.calls and result.metrics['backbone_calls'] == 0


def test_batch_normalization_is_total_block_nll_over_total_revealed_tokens():
    clean, corrupted, reveal, time = fixture()
    clean, corrupted, reveal, time = [x.repeat(2, 1) for x in (clean, corrupted, reveal, time)]
    reveal[1, 0] = False
    result = loss_on_corruption(clean, corrupted, reveal, time, frozen=TinyFrozen(),
        count_potential=potential(), head=GlobalAdditiveHead(5, 2), top_k=2)
    assert result.metrics['revealed_tokens'] == 3
    assert torch.equal(result.loss, result.objective.per_sequence.sum() / 3)


def test_adaptation_step_composes_schedule_without_extra_rng_or_forward():
    clean, *_ = fixture()
    frozen = TinyFrozen()
    result = adaptation_step(clean, frozen=frozen, count_potential=potential(),
        head=GlobalAdditiveHead(5, 2), generator=torch.Generator().manual_seed(5),
        schedule_steps=1, top_k=2)
    assert len(frozen.calls) == 1 and result.corrupted_tokens.eq(4).all()
    assert result.reveal_mask.all() and result.metrics['revealed_tokens'] == 4


def test_strict_targets_and_invalid_inputs_fail_before_leaking_labels():
    clean, corrupted, reveal, time = fixture()
    kwargs = dict(frozen=TinyFrozen(), count_potential=potential(), head=GlobalAdditiveHead(5, 2), top_k=2)
    with pytest.raises(ValueError, match='outside frozen support'):
        loss_on_corruption(clean, corrupted, reveal, time, missing='error', **kwargs)
    bad = corrupted.clone(); bad[0, 1] = 0
    with pytest.raises(ValueError, match='only replace'):
        loss_on_corruption(clean, bad, reveal, time, **kwargs)
    with pytest.raises(ValueError, match='only masked'):
        loss_on_corruption(clean, corrupted, torch.ones_like(reveal), time, **kwargs)
    with pytest.raises(ValueError, match='predecessor'):
        loss_on_corruption(clean, corrupted, reveal, time, global_strength=.25, **kwargs)
    local = ContextualAdditiveHead(torch.randn(5, 4), hidden_size=6, rank=2)
    kwargs['head'] = local
    with pytest.raises(ValueError, match='frozen/eval'):
        loss_on_corruption(clean, corrupted, reveal, time, global_head=GlobalAdditiveHead(5, 2), **kwargs)


def test_documented_toy_step_runs_without_assets_or_private_imports():
    import re
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'additive_chain/ADAPTATION.md').read_text()
    # Other sections contain API fragments that take caller-supplied inputs.
    # This test executes the explicitly self-contained runnable toy section.
    toy = text.split('## Runnable toy example\n', 1)[1].split('\n## ', 1)[0]
    code, = re.findall(r'```python\n(.*?)```', toy, re.DOTALL)
    namespace = {}
    exec(compile(code, 'ADAPTATION.md', 'exec'), namespace)
    assert namespace['result'].loss is not None
    assert torch.isfinite(namespace['result'].loss)
