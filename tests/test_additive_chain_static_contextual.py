"""Cached fixed-base/dynamic-local law tests; GPU gates are explicit/skipped on CPU."""
import itertools

import pytest
import torch

from additive_chain.heads import ContextualAdditiveHead, concatenate_scaled_factors
from additive_chain.cuda import dense_pair_reference as old_pair_reference, prepare_additive_factors
from additive_chain.chain import marginals
from additive_chain import _additive_kernels as kernels
from additive_chain import _count_inference as reference
from additive_chain import static_contextual as local
from additive_chain import static
from test_additive_chain_static import fixture, candidate_fixture


def local_factors(candidates, rank=3):
    shape = (*candidates.candidate_ids.shape, rank)
    points = torch.arange(candidates.candidate_ids.numel()*rank,
                          device=candidates.log_unary.device, dtype=torch.float32).reshape(shape)
    return points.sin()*.13, points.cos()*.19


def prepared(width=3, length=5, device='cpu', rank=3):
    potential, head = fixture(vocab=width+2, rank=rank, device=device)
    candidates = candidate_fixture(vocab=width+2, width=width, length=length, device=device)
    cache = static.prepare_static_global_cache(potential, head, .25)
    left, right = local_factors(candidates, rank)
    return potential, head, candidates, cache, left, right


def combined_factors(head, candidates, left, right):
    return concatenate_scaled_factors([(.25, head.node_factors(candidates.candidate_ids)),
                                        (1., (left, right))])


def test_cpu_three_term_scores_exact_support_and_dynamic_factors():
    p, h, c, cache, left, right = prepared()
    table_before = cache.table.log_scores.clone()
    for shifted_left in (left, left+.01):
        actual = local.dense_pair_reference(c, p, cache, shifted_left, right)
        expected = old_pair_reference(c, p, *combined_factors(h, c, shifted_left, right))
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    assert torch.equal(cache.table.log_scores, table_before)
    assert not torch.equal(local.dense_pair_reference(c, p, cache, left, right),
                           local.dense_pair_reference(c, p, cache, left+.01, right))


def test_cpu_real_contextual_head_fresh_masked_features_and_external_coefficient():
    p, h, c, cache, _, _ = prepared()
    embedding = torch.arange(p.vocab_size*4, dtype=torch.float32).reshape(-1, 4).sin()
    with torch.random.fork_rng(devices=[]):
        head = ContextualAdditiveHead(embedding, rank=3, context_size=5)
    head.eval().requires_grad_(False)
    head.right_token.weight.copy_(torch.arange(12).reshape(3, 4).cos()*.2)
    head.right_shift.bias.copy_(torch.tensor([.1, -.2, .3]))
    hidden = torch.arange(2*5*4, dtype=torch.float32).reshape(2, 5, 4).cos()
    ids_before, unary_before = c.candidate_ids.clone(), c.log_unary.clone()
    table_before = cache.table.log_scores.clone()
    results = []
    for feature in (hidden, hidden+hidden.sin()):
        left, right = concatenate_scaled_factors([(.7, head.node_factors(c.candidate_ids, feature))])
        actual = local.dense_pair_reference(c, p, cache, left, right)
        expected = old_pair_reference(c, p, *combined_factors(h, c, left, right))
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        results.append(actual)
    assert not torch.equal(*results)
    assert torch.equal(c.candidate_ids, ids_before) and torch.equal(c.log_unary, unary_before)
    assert torch.equal(cache.table.log_scores, table_before)
    assert all(parameter.grad is None for parameter in head.parameters())


@pytest.mark.parametrize('invalid', ['shape', 'dtype', 'stride'])
def test_cpu_invalid_cache_layout_rejected_before_inference(invalid):
    p, _, c, cache, left, right = prepared()
    if invalid == 'shape': cache.table.log_scores = cache.table.log_scores[:-1]
    if invalid == 'dtype': cache.table.log_scores = cache.table.log_scores.double()
    if invalid == 'stride': cache.table.log_scores = cache.table.log_scores.T
    with pytest.raises(ValueError, match='Contiguous matching FP32'):
        local.prepare_contextual_factors(reference.prepare_packet(c), p, cache, left, right)


def test_cpu_public_wrappers_return_token_ids_or_probabilities(monkeypatch):
    p, _, c, cache, left, right = prepared()
    slots = c.log_unary.argmax(-1)
    captured = []
    def fake_infer(*args, **kwargs):
        captured.append((args, kwargs))
        return slots if kwargs.get('sampling', 'joint') == 'joint' else c.log_unary
    monkeypatch.setattr(local, 'infer_static_contextual', fake_infer)
    generator = torch.Generator().manual_seed(252)
    output = local.sample_static_contextual(c, p, cache, left, right, generator, segment_batch_size=2)
    assert torch.equal(output, c.candidate_ids.gather(-1, slots[..., None]).squeeze(-1))
    assert captured[-1] == ((c, p, cache, left, right, generator), dict(segment_batch_size=2))
    probabilities = local.static_contextual_marginals(c, p, cache, left, right, segment_batch_size=2)
    assert torch.equal(probabilities, c.log_unary.exp())
    assert captured[-1] == ((c, p, cache, left, right),
                           dict(segment_batch_size=2, sampling='log_marginals'))


def test_cpu_both_visible_boundaries_and_padding_use_all_three_terms():
    p, h, c, cache, left, right = prepared()
    packet = reference.prepare_packet(c)
    factors = local.prepare_contextual_factors(packet, p, cache, left, right)
    pairs = local.dense_pair_reference(c, p, cache, left, right)
    rows, positions, slots = torch.tensor([1]), torch.tensor([0]), torch.tensor([0])
    assert factors.kind == 3 and factors.rank == 3 and factors.keys.numel() == 0
    torch.testing.assert_close(factors.boundary(rows, positions, slots, left_boundary=True),
                               pairs[1, 0, 0][None])
    torch.testing.assert_close(factors.boundary(rows, positions+1, slots, left_boundary=False),
                               pairs[1, 1, :, 0][None])
    assert (factors.left[1, [0, 2, 4], 1:] == 0).all()
    assert (factors.right[1, [0, 2, 4], 1:] == 0).all()


def test_cpu_complete_joint_law_by_enumeration_includes_unrevealed_nodes():
    p, h, c, cache, left, right = prepared()
    pair = local.dense_pair_reference(c, p, cache, left, right).double()
    paths = torch.tensor(list(itertools.product(range(3), repeat=5)))
    score = c.log_unary[:, torch.arange(5), paths].sum(-1)
    score += pair[:, torch.arange(4), paths[:, :-1], paths[:, 1:]].sum(-1)
    probability = score.softmax(-1)
    expected = torch.stack([torch.stack([probability[:, paths[:, pos] == slot].sum(-1)
                    for slot in range(3)], -1) for pos in range(5)], 1)
    actual = marginals(c.log_unary, pair, return_edges=False).node
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    assert torch.equal(actual[1, [0, 2, 4], 0], torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize('sampling', ['joint', 'log_marginals'])
@pytest.mark.parametrize('empty', [True, False])
def test_cpu_zero_local_uses_exact_cached_predecessor_no_rng_or_rebuild(monkeypatch, sampling, empty):
    p, h, c, cache, left, right = prepared()
    if empty:
        left = right = None
    else:
        left.zero_()
    generator = torch.Generator().manual_seed(251)
    before = generator.get_state().clone()
    expected = object()
    seen = []
    def predecessor(candidates, potential, received_cache, rng, **kwargs):
        seen.append((candidates, potential, received_cache, rng, kwargs))
        return expected
    monkeypatch.setattr(static, 'infer_static_global', predecessor)
    monkeypatch.setattr(static, 'prepare_static_global_cache', lambda *a, **k: pytest.fail('No hidden rebuild'))
    actual = local.infer_static_contextual(c, p, cache, left, right, generator, sampling=sampling)
    assert actual is expected and torch.equal(generator.get_state(), before)
    assert seen == [(c, p, cache, generator, dict(segment_batch_size=32, sampling=sampling))]


def test_cpu_rejects_wrong_factor_shape_nonfinite_stale_or_absent_table():
    p, h, c, cache, left, right = prepared()
    packet = reference.prepare_packet(c)
    with pytest.raises(ValueError, match='factors'):
        local.prepare_contextual_factors(packet, p, cache, left[..., :2], right)
    with pytest.raises(ValueError, match='finite'):
        local.prepare_contextual_factors(packet, p, cache, left*float('nan'), right)
    with pytest.raises(ValueError, match='budget'):
        local.dense_pair_reference(c, p, cache, left, right, max_cells=1)
    with pytest.raises(ValueError, match='NVIDIA'):
        local.infer_static_contextual(c, p, cache, left, right)
    h.left_embedding.weight.add_(.01)
    with pytest.raises(ValueError, match='changed'):
        local.prepare_contextual_factors(packet, p, cache, left, right)
    zero_cache = static.prepare_static_global_cache(p, h, 0.)
    with pytest.raises(ValueError, match='prepared'):
        local.prepare_contextual_factors(packet, p, zero_cache, left, right)
    zero_cache.close()
    with pytest.raises(RuntimeError, match='closed'):
        local.prepare_contextual_factors(packet, p, zero_cache, left, right)


def test_cpu_zero_global_zero_local_dense_reference_remains_count_only():
    p, h, c, _, left, right = prepared()
    cache = static.prepare_static_global_cache(p, h, 0.)
    torch.testing.assert_close(local.dense_pair_reference(c, p, cache, left*0, right),
                               old_pair_reference(c, p), atol=0, rtol=0)


def test_cpu_positive_local_with_zero_global_is_supported_without_implicit_cache(monkeypatch):
    p, h, c, _, left, right = prepared()
    cache = static.prepare_static_global_cache(p, h, 0.)
    torch.testing.assert_close(local.dense_pair_reference(c, p, cache, left, right),
                               old_pair_reference(c, p, left, right), atol=0, rtol=0)
    calls, expected = [], object()
    generator = torch.Generator().manual_seed(273)
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return expected
    monkeypatch.setattr(local.additive, 'infer_additive_count', original)
    actual = local.infer_static_contextual(c, p, cache, left, right, generator)
    assert actual is expected and calls == [((c, p, left, right, generator),
        dict(segment_batch_size=32, sampling='joint'))]
    assert cache.table is None


def test_cpu_wrong_vocabulary_and_sampling_mode_fail_closed():
    p, _, c, cache, left, right = prepared()
    with pytest.raises(ValueError, match='sampling'):
        local.infer_static_contextual(c, p, cache, left, right, sampling='map')
    from dataclasses import replace
    wrong = replace(c, vocab_size=c.vocab_size+1)
    for function in (local.infer_static_contextual, local.dense_pair_reference):
        with pytest.raises(ValueError, match='vocabularies'):
            function(wrong, p, cache, left, right)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='Explicit cached-contextual GPU window required')


@cuda
@pytest.mark.parametrize('width,length', [(3, 7), (500, 4), (16, 1024)])
def test_cuda_messages_joint_draws_and_marginals_match_fused_law(width, length):
    p, h, c, cache, left, right = prepared(width, length, 'cuda', rank=64)
    packet = reference.prepare_packet(c)
    old = prepare_additive_factors(packet, p, *combined_factors(h, c, left, right))
    new = local.prepare_contextual_factors(packet, p, cache, left, right)
    before_chunks = list(reference._chunks(packet, old, 3))
    after_chunks = list(reference._chunks(packet, new, 3))
    assert len(before_chunks) == len(after_chunks)
    for before, after in zip(before_chunks, after_chunks):
        unary, rows, starts, lengths, positions, valid = after
        assert all(torch.equal(a, b) for a, b in zip(before[1:], after[1:]))
        torch.testing.assert_close(unary, before[0], atol=2e-6, rtol=2e-6)
        alpha = kernels._messages(unary, new, rows, starts, lengths)
        oracle = kernels._messages(before[0], old, rows, starts, lengths)
        assert ((alpha.double().softmax(-1)-oracle.double().softmax(-1)).abs().sum(-1)*.5).max() < 2e-4
        noise = torch.empty_like(alpha, dtype=torch.float64).exponential_(
            generator=torch.Generator(device='cuda').manual_seed(281)).log_()
        a = kernels._sample_with_noise(alpha, new, rows, starts, lengths, noise)
        b = kernels._sample_with_noise(oracle, old, rows, starts, lengths, noise)
        assert torch.equal(a, b)
    actual = local.static_contextual_marginals(c, p, cache, left, right, segment_batch_size=3)
    pair = local.dense_pair_reference(c, p, cache, left, right).double()
    oracle = marginals(c.log_unary.float().double(), pair, return_edges=False).node
    assert ((actual.double()-oracle).abs().sum(-1)*.5).max() < 2e-4
    assert (actual.sum(-1)-1).abs().max() < 2e-6


@cuda
def test_cuda_zero_local_exact_tokens_and_rng_after_predecessor():
    p, h, c, cache, left, right = prepared(3, 7, 'cuda', rank=64)
    g1 = torch.Generator(device='cuda').manual_seed(291)
    g2 = torch.Generator(device='cuda').manual_seed(291)
    before = static.sample_static_global(c, p, cache, g1)
    after = local.sample_static_contextual(c, p, cache, left*0, right, g2)
    assert torch.equal(before, after) and torch.equal(g1.get_state(), g2.get_state())
