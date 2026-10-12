"""Independent decoding-law tests; no model downloads or empirical claims."""
from types import SimpleNamespace

import pytest
import torch

from additive_chain import generation as gen
from additive_chain.count_controls import add_count_unigram, smoothed_unigram_log_probs
from additive_chain.sampling_controls import IndependentControl


def marginal(device='cpu'):
    return smoothed_unigram_log_probs(torch.tensor([3., 5., 10., 2., 0.], device=device))


def forward(tokens, time):
    # Depend on the evolving sequence, so trajectory parity is not a constant-logit test.
    base = torch.arange(5, device=tokens.device).float()[None, None] * -.3
    context = tokens.remainder(3).float()[..., None] * torch.tensor(
        [.03, -.05, .08, .01, 0.], device=tokens.device)
    return (base + context + time * .01).expand(*tokens.shape, 5).clone()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Actual CUDA required'))])
@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize('steps', [1, 7, 32])
def test_zero_and_unit_temperature_match_baseline_tokens_and_both_rng_states(device, fast, steps):
    tokens = torch.tensor([[0, 4, 4, 1, 4, 4], [4, 4, 2, 4, 4, 4]], device=device)
    traces = []
    for control in (None, IndependentControl(frequency_strength=0),
                    IndependentControl(temperature=1),
                    IndependentControl(log_marginal=marginal(device), frequency_strength=0, temperature=1)):
        decoder = gen.ChainDecoder(mask_id=4, device=device, top_k=3, fast_candidates=fast,
                                   independent_control=control)
        decoder.prepare()
        reveal = torch.Generator(device=device).manual_seed(41)
        proposal = torch.Generator(device=device).manual_seed(78)
        result = gen.sample_window(tokens, forward, decoder, steps=steps,
                                  reveal_generator=reveal, proposal_generator=proposal)
        traces.append((result.tokens, reveal.get_state(), proposal.get_state(), result.stats))
        assert decoder.cache is None and not decoder.active
        decoder.close()
    for actual in traces[1:]:
        assert all(torch.equal(a, b) for a, b in zip(actual[:3], traces[0][:3]))
        assert actual[3] == traces[0][3]


@pytest.mark.parametrize('alpha,tau', [(.1, 1.), (0., .95), (.2, .8)])
@pytest.mark.parametrize('fast', [False, True])
def test_original_support_and_exact_individual_or_composed_law(alpha, tau, fast, monkeypatch):
    from additive_chain import _count_inference, cuda
    def forbidden(*a, **kw):
        raise AssertionError('Independent control must not use pair inference/setup')
    monkeypatch.setattr(_count_inference, 'prepare_count_cache', forbidden)
    monkeypatch.setattr(cuda, 'sample_additive_count', forbidden)
    log_j = marginal()
    control = IndependentControl(frequency_strength=alpha,
                                 log_marginal=log_j if alpha else None, temperature=tau)
    decoder = gen.ChainDecoder(mask_id=4, top_k=3, nucleus_p=.9,
                               fast_candidates=fast, independent_control=control)
    decoder.prepare()
    # First selected row is ORIGINAL position2, not position0. The next batch
    # also selects position0, which alone must receive no incoming-edge term.
    tokens = torch.tensor([[0, 1, 4, 4], [4, 2, 4, 1]])
    selected = torch.tensor([[False, False, True, False], [True, False, True, False]])
    raw = forward(tokens, .5)
    expected_candidates = decoder._candidates(raw[selected][None], tokens[selected][None])
    before_ids, before_unary = expected_candidates.candidate_ids.clone(), expected_candidates.log_unary.clone()
    seen = []
    original = decoder._candidates
    def extract(values, x):
        assert torch.equal(values, raw[selected][None])
        c = original(values, x)
        seen.append(c)
        return c
    monkeypatch.setattr(decoder, '_candidates', extract)
    weights = expected_candidates.log_unary
    if alpha:
        incoming = torch.tensor([[True, False, True]])
        weights = add_count_unigram(weights, expected_candidates.candidate_ids, log_j, incoming, strength=alpha)
    if tau != 1.:
        weights = weights / tau
    probabilities = weights[0].softmax(-1)
    proposal, reference = torch.Generator().manual_seed(71), torch.Generator().manual_seed(71)
    slot = torch.multinomial(probabilities, 1, generator=reference)
    expected = tokens.clone()
    expected[selected] = before_ids[0].gather(-1, slot).squeeze(-1)
    actual = decoder.reveal(raw, tokens, selected, proposal)
    assert len(seen) == 1
    assert torch.equal(seen[0].candidate_ids, before_ids)
    assert torch.equal(seen[0].log_unary, before_unary)
    torch.testing.assert_close(control.probabilities(seen[0], selected), probabilities, rtol=0, atol=0)
    assert torch.equal(actual, expected) and torch.equal(proposal.get_state(), reference.get_state())
    assert torch.equal(actual[~selected], tokens[~selected])
    assert decoder.cache is None
    decoder.close()


def test_nucleus_is_not_recomputed_after_temperature_or_frequency():
    # Original p=.9 excludes its crossing token. Cooling BEFORE top-p would
    # produce a different support; our transformation must retain both IDs.
    raw = torch.tensor([[[.50, .30, .15, .05, 0.]]]).log().float()
    tokens = torch.tensor([[4]])
    decoder = gen.ChainDecoder(mask_id=4, top_k=4, nucleus_p=.9)
    c = decoder._candidates(raw, tokens)
    assert c.candidate_ids[0,0,:2].tolist() == [0, 1]
    cooled = IndependentControl(temperature=.1).probabilities(c, tokens.eq(4))
    assert (cooled > 0).sum() == 2
    assert cooled[0, 2:].eq(0).all()
    assert c.candidate_ids[0,0,2:].eq(5).all() and c.log_unary[0,0,2:].isneginf().all()


def test_static_marginal_copied_validated_once_and_mutation_rejected():
    log_j = marginal()
    control = IndependentControl(frequency_strength=.1, log_marginal=log_j)
    before = control.identity()
    log_j.add_(123)
    assert abs(float(control._log_marginal.logsumexp(0))) < 1e-10
    assert control.identity() == before
    c = gen.ChainDecoder(mask_id=4)._candidates(forward(torch.tensor([[4]]), .5), torch.tensor([[4]]))
    control._log_marginal.add_(1)
    with pytest.raises(ValueError, match='changed'):
        control.probabilities(c, torch.tensor([[True]]))


def test_identity_binds_exact_marginal_bytes_dtype_shape_and_cannot_be_mutated():
    import hashlib
    one, two = marginal(), marginal().flip(0)
    first = IndependentControl(frequency_strength=.1, log_marginal=one)
    second = IndependentControl(frequency_strength=.1, log_marginal=two)
    metadata = first.identity()['log_marginal']
    assert metadata == dict(sha256=hashlib.sha256(one.numpy().tobytes()).hexdigest(),
                            dtype='torch.float64', shape=[5])
    assert metadata['sha256'] != second.identity()['log_marginal']['sha256']
    metadata['shape'][0] = 999
    assert first.identity()['log_marginal']['shape'] == [5]
    assert IndependentControl().identity()['log_marginal'] is None


def test_constructor_inside_inference_mode_and_identity_settings_do_not_consume_rng():
    initial = torch.random.get_rng_state().clone()
    with torch.inference_mode():
        control = IndependentControl(frequency_strength=.1, log_marginal=marginal())
        tokens = torch.tensor([[4, 4]])
        c = gen.extract_compact_candidates(forward(tokens, .5), tokens,
                                            mask_id=4, nucleus_p=.9, top_k=3)
        probability = control.probabilities(c, tokens.eq(4))
        assert torch.isfinite(probability).all()
        torch.testing.assert_close(probability.sum(-1), torch.ones(2, dtype=torch.float64),
                                   rtol=0, atol=1e-15)
        neutral = IndependentControl(log_marginal=marginal(), frequency_strength=0, temperature=1)
        assert torch.equal(neutral.probabilities(c, tokens.eq(4)), c.log_unary[0].exp())
    assert torch.equal(initial, torch.random.get_rng_state())


@pytest.mark.parametrize('options', [dict(frequency_strength=-1), dict(frequency_strength=True),
    dict(frequency_strength=float('nan')), dict(temperature=0), dict(temperature=True),
    dict(temperature=float('inf')), dict(frequency_strength=.1),
    dict(log_marginal=torch.zeros(5)), dict(log_marginal=torch.zeros(5,dtype=torch.float64))])
def test_invalid_controls_rejected(options):
    with pytest.raises(ValueError):
        IndependentControl(**options)


def test_vocabulary_and_pair_composition_rejected_without_dropping_inputs():
    control = IndependentControl(frequency_strength=.1, log_marginal=marginal())
    with pytest.raises(ValueError, match='vocabulary'):
        gen.ChainDecoder(mask_id=5, independent_control=control)
    with pytest.raises(ValueError, match='zero pair strengths'):
        gen.ChainDecoder(mask_id=4, count_strength=.1, independent_control=control)


def test_empty_window_consumes_only_original_reveal_stream():
    decoder = gen.ChainDecoder(mask_id=4, independent_control=IndependentControl(temperature=.9))
    decoder.prepare()
    reveal, proposal = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    initial = proposal.get_state().clone()
    result = gen.sample_window(torch.tensor([[0,1,2]]),
        lambda *a: pytest.fail('No forward for visible window'), decoder, steps=4,
        reveal_generator=reveal, proposal_generator=proposal)
    assert result.stats['actual_forward_calls'] == 0
    assert torch.equal(initial, proposal.get_state())
