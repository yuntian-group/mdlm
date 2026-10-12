"""Static score/law CPU tests and explicitly optional CUDA parity tests."""
import itertools

import pytest
import torch

from additive_chain.candidates import CompactCandidates
from additive_chain.counts import SparseCountPotential
from additive_chain.heads import GlobalAdditiveHead, concatenate_scaled_factors
from additive_chain.cuda import dense_pair_reference, prepare_additive_factors
from additive_chain import _additive_kernels as af
from additive_chain import _count_inference as reference
from additive_chain import _dense_inference as dense
from additive_chain.chain import marginals
from additive_chain import static


def fixture(vocab=7, rank=3, device='cpu'):
    left = torch.linspace(.6, .9, vocab, device=device, dtype=torch.float64)
    right = torch.linspace(.7, .4, vocab, device=device, dtype=torch.float64)
    rows = torch.tensor([0, 1, vocab-2], device=device)
    cols = torch.tensor([1, 0, vocab-1], device=device)
    values = torch.tensor([.3, .6, .2], device=device, dtype=torch.float64)
    potential = SparseCountPotential(left, right, rows*vocab+cols, values, log_scale=7.)
    head = GlobalAdditiveHead(vocab, rank=rank).to(device).requires_grad_(False).eval()
    with torch.no_grad():
        points = torch.arange(vocab * rank, device=device, dtype=torch.float32).reshape(vocab, rank)
        head.left_embedding.weight.copy_(points.sin() * .3)
        head.right_embedding.weight.copy_(points.cos() * .2)
    return potential, head


def candidate_fixture(vocab=7, width=3, length=5, device='cpu'):
    ids = torch.arange(width, device=device)[None, None].expand(2, length, width).clone()
    values = torch.arange(2 * length * width, device=device, dtype=torch.float64)
    unary = values.sin().reshape(2, length, width).log_softmax(-1)
    for pos in sorted({0, length//2, length-1}):
        ids[1, pos] = vocab
        ids[1, pos, 0] = 1
        unary[1, pos] = -torch.inf
        unary[1, pos, 0] = 0.
    return CompactCandidates(ids, unary, vocab)


def table_pairs(candidates, cache):
    packet = reference.prepare_packet(candidates)
    return dense.dense_pair_scores(cache.table, packet.candidate_ids[:, :-1, :, None],
                                  packet.candidate_ids[:, 1:, None, :])


def test_cpu_score_law_same_coefficient_order_no_factor_rescaling():
    potential, head = fixture()
    cache = static.prepare_static_global_cache(potential, head, .25, max_temporary_bytes=168)
    all_ids = torch.arange(7)[None, None].expand(1, 2, 7).clone()
    candidates = CompactCandidates(all_ids, torch.zeros_like(all_ids, dtype=torch.float64), 7)
    factors = concatenate_scaled_factors([(.25, head.node_factors(all_ids))])
    expected = dense_pair_reference(candidates, potential, *factors)[0, 0]
    torch.testing.assert_close(cache.table.log_scores, expected, atol=1e-7, rtol=1e-7)
    keys, log_weights, left, right = reference.count_log_components(potential)
    direct = (left[:, None] + right[None, :]).float()
    direct.view(-1)[keys] = log_weights.float()
    direct += (.25 * head.left_embedding.weight) @ head.right_embedding.weight.T
    torch.testing.assert_close(cache.table.log_scores, direct, atol=1e-7, rtol=1e-7)
    assert cache.table.omitted_log_scale == potential.log_scale
    assert cache.diagnostics['table_bytes'] == 7 * 7 * 4
    assert cache.diagnostics['learned_chunk_tensor_bytes'] <= 168
    assert cache.diagnostics['persistent_files_written'] is False
    assert cache.diagnostics['gpu_law_gate_passed'] is False
    assert cache.diagnostics['speedup_measured'] is False


def test_cpu_observed_boundaries_padding_and_exhaustive_joint_law():
    potential, head = fixture()
    candidates = candidate_fixture()
    cache = static.prepare_static_global_cache(potential, head, .7)
    packet = reference.prepare_packet(candidates)
    factors = cache.prepared_factors(packet)
    pair = table_pairs(candidates, cache).double()
    old = dense_pair_reference(candidates, potential,
        *concatenate_scaled_factors([(.7, head.node_factors(candidates.candidate_ids))])).double()
    torch.testing.assert_close(pair, old, atol=2e-7, rtol=2e-7)
    rows, positions, states = torch.tensor([1]), torch.tensor([0]), torch.tensor([0])
    torch.testing.assert_close(factors.boundary(rows, positions, states, left_boundary=True),
                               cache.table.log_scores[1, :3][None])
    torch.testing.assert_close(factors.boundary(rows, positions+1, states, left_boundary=False),
                               cache.table.log_scores[:3, 1][None])
    assert dense.dense_pair_scores(cache.table, torch.tensor([-1, 7]), torch.tensor([2, 0])).tolist() == [0., 0.]
    paths = torch.tensor(list(itertools.product(range(3), repeat=5)))
    scores = candidates.log_unary[:, torch.arange(5), paths].sum(-1)
    scores += pair[:, torch.arange(4), paths[:, :-1], paths[:, 1:]].sum(-1)
    probability = scores.softmax(-1)
    enumerated = torch.stack([torch.stack([probability[:, paths[:, pos] == slot].sum(-1)
                              for slot in range(3)], -1) for pos in range(5)], 1)
    actual = marginals(candidates.log_unary, pair, return_edges=False).node
    torch.testing.assert_close(actual, enumerated, atol=1e-14, rtol=1e-14)
    assert torch.equal(actual[1, [0, 2, 4], 0], torch.ones(3, dtype=torch.float64))
    assert (actual[1, [0, 2, 4], 1:] == 0).all()


@pytest.mark.parametrize('coefficient,zero_right', [(0., False), (.7, True)])
def test_cpu_exact_zero_delegates_without_allocating_or_changing_rng(monkeypatch, coefficient, zero_right):
    potential, head = fixture()
    if zero_right:
        head.right_embedding.weight.zero_()
    monkeypatch.setattr(dense, 'build_dense_count_table', lambda *a, **k: pytest.fail('Zero allocated a table'))
    cache = static.prepare_static_global_cache(potential, head, coefficient, max_table_bytes=0)
    candidates = candidate_fixture()
    generator = torch.Generator().manual_seed(137)
    seen = []
    def original(c, p, **kwargs):
        seen.append((c, p, kwargs))
        return torch.zeros(c.candidate_ids.shape[:2], dtype=torch.long)
    monkeypatch.setattr(static.additive, 'infer_additive_count', original)
    actual = static.sample_static_global(candidates, potential, cache, generator)
    assert torch.equal(actual, candidates.candidate_ids[..., 0])
    assert seen[0][0] is candidates and seen[0][1] is potential
    assert seen[0][2]['generator'] is generator
    assert cache.table is None and cache.diagnostics['zero_dispatch_bitwise']


def test_cpu_frozen_source_invalidation_and_explicit_close():
    potential, head = fixture()
    cache = static.prepare_static_global_cache(potential, head, .25)
    cache.assert_current(potential)
    head.right_embedding.weight.add_(.01)
    with pytest.raises(ValueError, match='changed'):
        cache.assert_current(potential)
    cache = static.prepare_static_global_cache(potential, head, .25)
    potential.row_values.add_(.01)
    with pytest.raises(ValueError, match='changed'):
        cache.assert_current(potential)
    cache = static.prepare_static_global_cache(potential, head, .25)
    cache.close()
    assert cache.table is None
    with pytest.raises(RuntimeError, match='closed'):
        cache.assert_current(potential)


def test_cpu_explicit_budget_type_finite_and_training_guards():
    potential, head = fixture()
    for coefficient in (-1., True, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='coefficient'):
            static.prepare_static_global_cache(potential, head, coefficient)
    with pytest.raises(MemoryError, match='budget'):
        static.prepare_static_global_cache(potential, head, .25, max_table_bytes=195)
    with pytest.raises(MemoryError, match='Temporary'):
        static.prepare_static_global_cache(potential, head, .25, max_temporary_bytes=1)
    with pytest.raises(ValueError, match='budgets'):
        static.prepare_static_global_cache(potential, head, .25, reserve_bytes=-1)
    head.requires_grad_(True)
    with pytest.raises(ValueError, match='Frozen'):
        static.prepare_static_global_cache(potential, head, .25)
    head.requires_grad_(False)
    head.left_embedding.weight[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        static.prepare_static_global_cache(potential, head, .25)


def test_cpu_fp64_long_chain_oracle_preserves_visible_constraints():
    potential, head = fixture()
    candidates = candidate_fixture(length=1024)
    cache = static.prepare_static_global_cache(potential, head, .25)
    pair = table_pairs(candidates, cache).double()
    expected = dense_pair_reference(candidates, potential,
        *concatenate_scaled_factors([(.25, head.node_factors(candidates.candidate_ids))])).double()
    actual = marginals(candidates.log_unary, pair, return_edges=False).node
    oracle = marginals(candidates.log_unary, expected, return_edges=False).node
    assert ((actual - oracle).abs().sum(-1) * .5).max() < 1e-7
    assert (actual.sum(-1) - 1).abs().max() < 1e-12
    assert (actual[1, [0, 512, 1023], 0] == 1).all()


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='Explicit static-global CUDA window required')


@cuda
@pytest.mark.parametrize('width,length', [(3, 7), (500, 5), (16, 1024)])
def test_cuda_table_messages_and_same_noise_draws_against_additive(width, length):
    potential, head = fixture(vocab=width+2, rank=64, device='cuda')
    candidates = candidate_fixture(vocab=width+2, width=width, length=length, device='cuda')
    cache = static.prepare_static_global_cache(potential, head, .25, reserve_bytes=64*1024**2)
    left, right = concatenate_scaled_factors([(.25, head.node_factors(candidates.candidate_ids))])
    packet = reference.prepare_packet(candidates)
    old_factors = prepare_additive_factors(packet, potential, left, right)
    new_factors = cache.prepared_factors(packet)
    old_chunks = list(reference._chunks(packet, old_factors, 3))
    new_chunks = list(reference._chunks(packet, new_factors, 3))
    assert len(old_chunks) == len(new_chunks)
    for old, new in zip(old_chunks, new_chunks):
        old_unary, rows, starts, lengths, positions, valid = old
        unary = new[0]
        assert all(torch.equal(a, b) for a, b in zip(old[1:], new[1:]))
        torch.testing.assert_close(unary, old_unary, atol=2e-6, rtol=2e-6)
        alpha = af._messages(old_unary, old_factors, rows, starts, lengths)
        actual = dense.dense_messages(unary, new_factors, rows, starts, lengths)
        # Compare normalized conditionals, not arbitrary whole-chain offsets.
        old_p, new_p = alpha.double().softmax(-1), actual.double().softmax(-1)
        assert ((old_p-new_p).abs().sum(-1)*.5).max() <= 2e-4
        noise = torch.empty_like(alpha, dtype=torch.float64).exponential_(
            generator=torch.Generator(device='cuda').manual_seed(173)).log_()
        before = af._sample_with_noise(alpha, old_factors, rows, starts, lengths, noise)
        after = dense.dense_sample_with_noise(actual, new_factors, rows, starts, lengths, noise)
        assert torch.equal(before, after)
    actual = static.static_global_marginals(candidates, potential, cache, segment_batch_size=3)
    oracle_pair = table_pairs(candidates, cache).double()
    # This small-K independent FP64 recurrence validates the table FFBS itself.
    if width <= 16:
        oracle = marginals(candidates.log_unary.float().double(), oracle_pair, return_edges=False).node
        assert ((actual.double()-oracle).abs().sum(-1)*.5).max() <= 2e-4
    assert (actual.sum(-1)-1).abs().max() <= 2e-6
