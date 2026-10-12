"""CPU scheduler/adapter tests, not CUDA or real-checkpoint trajectory claims."""
import hashlib
from types import SimpleNamespace

import pytest
import torch

from additive_chain import generation as gen
from additive_chain.candidates import extract_compact_candidates
from additive_chain.heads import GlobalAdditiveHead, ContextualAdditiveHead


def rng(seed):
    return torch.Generator().manual_seed(seed)


def counts(vocab=5):
    matrix = torch.arange(vocab * vocab).reshape(vocab, vocab).double() + 1
    return {"left_counts": matrix.sum(1), "right_counts": matrix.sum(0),
            "pair_keys": torch.arange(vocab * vocab), "pair_counts": matrix.flatten()}


def frozen_head(kind="global", vocab=5):
    if kind == "global":
        head = GlobalAdditiveHead(vocab, rank=2)
    else:
        head = ContextualAdditiveHead(torch.arange(vocab * 3).reshape(vocab, 3).float() / 10,
                                     hidden_size=3, rank=2, context_size=2)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.copy_(torch.arange(parameter.numel()).reshape_as(parameter).float() / 30 + .03)
    return head.eval().requires_grad_(False)


def logits(tokens, t):
    values = torch.arange(5).float()[None, None, :] * -.4
    return values.expand(*tokens.shape, 5).clone()


def legacy_baseline(tokens, forward, *, steps, reveal, proposal, top_k=3, top_p=.9):
    """Standalone literal finite-step baseline loop, with independent draws."""
    current = tokens.clone()
    sizes, times = [], []
    for index in range(steps):
        t, s = (steps - index) / steps, (steps - index - 1) / steps
        chance = 1. if index == steps - 1 else 1. - s / t
        uniform = torch.rand(tokens.shape, dtype=torch.float64, generator=reveal)
        selected = (uniform < chance) & current.eq(4)
        sizes.append(int(selected.sum()))
        if selected.any():
            raw = forward(current, t)
            c = extract_compact_candidates(raw[selected][None], current[selected][None],
                                          mask_id=4, nucleus_p=top_p, top_k=top_k)
            slot = torch.multinomial(c.log_unary[0].exp(), 1, generator=proposal)
            current[selected] = c.candidate_ids[0].gather(-1, slot).squeeze(-1)
            times.append(t)
    return current, sizes, times


@pytest.mark.parametrize("steps", [1, 2, 7, 32])
@pytest.mark.parametrize("fast", [False, True])
def test_baseline_tokens_and_both_rng_states_match_original_loop(steps, fast):
    tokens = torch.tensor([[0, 4, 4, 1, 4, 4, 4]])
    decoder = gen.ChainDecoder(mask_id=4, top_k=3, fast_candidates=fast)
    decoder.prepare()
    reveal, proposal, other_r, other_p = rng(31), rng(72), rng(31), rng(72)
    calls = []
    def forward(current, t):
        calls.append((current.clone(), t))
        return logits(current, t)
    result = gen.sample_window(tokens, forward, decoder, steps=steps,
                               reveal_generator=reveal, proposal_generator=proposal)
    expected, sizes, times = legacy_baseline(tokens, logits, steps=steps,
                                             reveal=other_r, proposal=other_p)
    assert torch.equal(result.tokens, expected)
    assert torch.equal(reveal.get_state(), other_r.get_state())
    assert torch.equal(proposal.get_state(), other_p.get_state())
    assert result.stats["reveal_counts"] == sizes
    assert [t for _, t in calls] == times
    assert result.stats["actual_forward_calls"] == len(times)
    assert torch.equal(result.tokens[tokens.ne(4)], tokens[tokens.ne(4)])


def test_empty_steps_draw_reveal_uniforms_but_never_forward_or_propose():
    decoder = gen.ChainDecoder(mask_id=4)
    decoder.prepare()
    r, p, ref_r = rng(11), rng(12), rng(11)
    original_p = p.get_state().clone()
    def forbidden(*args):
        raise AssertionError("No masked sites")
    result = gen.sample_window(torch.tensor([[0, 1, 2]]), forbidden, decoder, steps=8,
                               reveal_generator=r, proposal_generator=p)
    for _ in range(8):
        torch.rand((1, 3), dtype=torch.float64, generator=ref_r)
    assert result.stats["actual_forward_calls"] == 0
    assert torch.equal(r.get_state(), ref_r.get_state())
    assert torch.equal(p.get_state(), original_p)


def test_stream_keys_match_existing_development_and_final_namespaces():
    for prefix in ("bd3lm_parallel_count_sweep_v1:fixed-seed-1", "additive-ladder-final-evaluation-v1"):
        key = f"{prefix}:attempt-3:window-0:reveal"
        expected = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)
        assert gen.stream_seed(prefix, 3, "reveal") == expected
        assert gen.stream_seed(prefix, 3, "proposal") != expected
    with pytest.raises(ValueError):
        gen.stream_seed("", 0, "reveal")


def test_zero_all_coefficients_recover_baseline_without_touching_heads():
    class ForbiddenHead:
        def __getattr__(self, name):
            raise AssertionError("Inactive head must not execute")
    a = gen.ChainDecoder(mask_id=4, global_head=ForbiddenHead(), contextual_head=ForbiddenHead())
    b = gen.ChainDecoder(mask_id=4, sampling="own-marginal", backend="static")
    a.prepare(); b.prepare()
    tokens = torch.tensor([[0, 4, 4, 4]])
    ra, rb = rng(8), rng(8)
    pa, pb = rng(9), rng(9)
    x = gen.sample_window(tokens, logits, a, steps=3, reveal_generator=ra, proposal_generator=pa)
    y = gen.sample_window(tokens, logits, b, steps=3, reveal_generator=rb, proposal_generator=pb)
    assert torch.equal(x.tokens, y.tokens) and torch.equal(pa.get_state(), pb.get_state())
    assert a.cache is None and b.cache is None


@pytest.fixture
def mock_gpu_setup(monkeypatch):
    from additive_chain import _count_inference
    monkeypatch.setattr(_count_inference, "prepare_count_cache", lambda potential: 0)


def test_joint_composition_exact_support_same_hidden_and_left_strengths(mock_gpu_setup, monkeypatch):
    from additive_chain import cuda
    global_head, local = frozen_head(), frozen_head("contextual")
    decoder = gen.ChainDecoder(mask_id=4, count_statistics=counts(), count_strength=.1,
        global_head=global_head, global_strength=.25, contextual_head=local,
        contextual_strength=2., top_k=3, nucleus_p=1., prune_irrelevant=False)
    decoder.prepare()
    tokens = torch.tensor([[0, 4, 4, 4]])
    selected = torch.tensor([[False, False, True, False]])
    hidden = torch.arange(12).reshape(1, 4, 3).float()
    output = SimpleNamespace(log_probs=logits(tokens, .5), hidden=hidden)
    observed = {}
    def sample(c, potential, left, right, generator, **kwargs):
        observed.update(c=c, potential=potential, left=left, right=right)
        return c.candidate_ids[..., 0]
    monkeypatch.setattr(cuda, "sample_additive_count", sample)
    actual = decoder.reveal(output, tokens, selected, rng(4))
    c = observed["c"]
    # The two unrevealed latent sites still have their complete candidates.
    assert torch.isfinite(c.log_unary[0, 1]).sum() == 3
    assert torch.isfinite(c.log_unary[0, 3]).sum() == 3
    gl, gr = global_head.node_factors(c.candidate_ids, hidden)
    ll, lr = local.node_factors(c.candidate_ids, hidden)
    torch.testing.assert_close(observed["left"], torch.cat((gl * .25, ll * 2), -1))
    torch.testing.assert_close(observed["right"], torch.cat((gr, lr), -1))
    assert torch.equal(actual[~selected], tokens[~selected])
    assert actual[0, 2] != 4


def test_own_marginal_uses_same_chain_and_draws_selected_only(mock_gpu_setup, monkeypatch):
    from additive_chain import cuda, chain
    decoder = gen.ChainDecoder(mask_id=4, count_statistics=counts(), count_strength=.1,
        sampling="own-marginal", top_k=3, nucleus_p=1., prune_irrelevant=False)
    decoder.prepare()
    tokens = torch.tensor([[0, 4, 4, 4]])
    selected = torch.tensor([[False, True, False, True]])
    output = logits(tokens, .5)
    observed = {}
    def marginals(c, potential, left, right, **kwargs):
        pair = cuda.dense_pair_reference(c, potential, left, right)
        probability = chain.marginals(c.log_unary.float(), pair).node.float()
        observed.update(c=c, probability=probability)
        return probability
    monkeypatch.setattr(cuda, "additive_count_marginals", marginals)
    proposal, expected_rng = rng(17), rng(17)
    result = decoder.reveal(output, tokens, selected, proposal)
    slot = torch.multinomial(observed["probability"][selected], 1, generator=expected_rng)
    expected = tokens.clone()
    expected[selected] = observed["c"].candidate_ids[selected].gather(-1, slot).squeeze(-1)
    assert torch.equal(result, expected)
    assert torch.equal(proposal.get_state(), expected_rng.get_state())


def test_pruning_only_inference_not_frozen_context(mock_gpu_setup, monkeypatch):
    from additive_chain import cuda
    decoder = gen.ChainDecoder(mask_id=4, count_statistics=counts(), count_strength=.1, top_k=3)
    decoder.prepare()
    tokens = torch.tensor([[0, 4, 1, 4, 4]])
    selected = torch.tensor([[False, True, False, False, False]])
    observed = {}
    def sample(c, *args, **kwargs):
        observed["c"] = c
        return c.candidate_ids[..., 0]
    monkeypatch.setattr(cuda, "sample_additive_count", sample)
    result = decoder.reveal(logits(tokens, .5), tokens, selected, rng(4))
    assert observed["c"].diagnostics["masked"].tolist() == [[False, True, False, False, False]]
    assert torch.equal(result[~selected], tokens[~selected])
    assert tokens.tolist() == [[0, 4, 1, 4, 4]]


def test_static_cache_prepared_once_and_contextual_not_cached(monkeypatch):
    from additive_chain import static_contextual
    local = frozen_head("contextual")
    decoder = gen.ChainDecoder(mask_id=4, count_statistics=counts(), count_strength=.1,
        contextual_head=local, contextual_strength=.25, backend="static", top_k=3)
    setup = decoder.prepare(max_table_bytes=1024, max_temporary_bytes=4096, reserve_bytes=0)
    table = decoder.cache.table
    seen = []
    def sample(c, potential, cache, left, right, generator, **kwargs):
        assert cache.table is table
        seen.append(left.clone())
        return c.candidate_ids[..., 0]
    monkeypatch.setattr(static_contextual, "sample_static_contextual", sample)
    tokens = torch.tensor([[0, 4, 4]])
    selected = tokens.eq(4)
    for value in ([1., 2., 4.], [2., 4., 1.]):
        decoder.reveal(SimpleNamespace(log_probs=logits(tokens, .5),
            hidden=torch.tensor(value)[None, None].expand(1, 3, 3)), tokens, selected, rng(3))
    assert setup["excluded_from_generation_timing"] and setup["seconds"] >= 0
    assert not torch.equal(seen[0], seen[1])
    with pytest.raises(ValueError, match="already"):
        decoder.prepare()
    decoder.close()
    assert decoder.cache is None


def test_malformed_generators_hidden_and_unselected_logits_rejected(mock_gpu_setup):
    decoder = gen.ChainDecoder(mask_id=4, strict_unary_validation=True)
    decoder.prepare()
    tokens = torch.tensor([[0, 4, 4]])
    same = rng(3)
    with pytest.raises(ValueError, match="distinct"):
        gen.sample_window(tokens, logits, decoder, steps=1, reveal_generator=same, proposal_generator=same)
    bad = logits(tokens, .5); bad[0, 2] = -torch.inf
    with pytest.raises(ValueError, match="Every masked"):
        decoder.reveal(bad, tokens, torch.tensor([[False, True, False]]), rng(1))
    active = gen.ChainDecoder(mask_id=4, contextual_head=frozen_head("contextual"), contextual_strength=1.)
    active.prepare()
    with pytest.raises(ValueError, match="same masked"):
        active.reveal(logits(tokens, .5), tokens, tokens.eq(4), rng(4))


def test_default_validates_used_rows_without_duplicate_full_window_scan(monkeypatch):
    decoder = gen.ChainDecoder(mask_id=4)
    decoder.prepare()
    tokens = torch.tensor([[0, 4, 4]])
    selected = torch.tensor([[False, True, False]])
    bad = logits(tokens, .5); bad[0, 2] = -torch.inf
    scanned_shapes = []
    original = torch.isfinite
    def inspected(value):
        scanned_shapes.append(tuple(value.shape))
        return original(value)
    monkeypatch.setattr(torch, "isfinite", inspected)
    result = decoder.reveal(bad, tokens, selected, rng(1))
    assert result[0, 1] != 4 and result[0, 2] == 4
    assert tuple(bad.shape) not in scanned_shapes
    # The same invalid row must be rejected when it actually enters a proposal.
    with pytest.raises(ValueError, match="finite clean"):
        decoder.reveal(bad, tokens, tokens.eq(4), rng(1))


def test_latent_unselected_rows_are_still_validated_for_active_chain(mock_gpu_setup):
    decoder = gen.ChainDecoder(mask_id=4, count_statistics=counts(), count_strength=.1)
    decoder.prepare()
    tokens = torch.tensor([[0, 4, 4]])
    bad = logits(tokens, .5); bad[0, 2] = -torch.inf
    # Both hidden sites share a component, so the unselected one remains latent.
    with pytest.raises(ValueError, match="finite clean"):
        decoder.reveal(bad, tokens, torch.tensor([[False, True, False]]), rng(1))


class Tokenizer:
    bos_token_id = eos_token_id = 0
    def batch_decode(self, rows, skip_special_tokens):
        assert skip_special_tokens is False
        return [" ".join(map(str, row)) for row in rows]


def high_entropy_forward(tokens, t):
    result = torch.full((*tokens.shape, 66), -torch.inf)
    values = torch.arange(tokens.shape[1]).remainder(65)
    result[0, torch.arange(tokens.shape[1]), values] = 0
    return result


def test_attempt_initial_context_schedule_rejection_and_saved_format():
    decoder = gen.ChainDecoder(mask_id=65, top_k=64)
    decoder.prepare()
    result = gen.generate_saved(high_entropy_forward, decoder, Tokenizer(), samples=2,
        stream_namespace="unit-test-only", steps=4, length=256, warmup=True)
    assert result["schema"] == "additive_chain_saved_generations_v1"
    assert result["status"] == "complete" and len(result["attempts"]) == 2
    assert result["warmup_seconds"] > 0
    for i, row in enumerate(result["attempts"]):
        assert row["attempt_id"] == i and row["accepted"]
        assert row["token_ids"][0] == 0 and len(row["token_ids"]) == 256
        assert row["new_tokens"] == 255 and row["generation_seconds"] > 0
        assert row["text"].startswith("0 1 2 3")
        assert row["window"]["actual_forward_calls"] == 4
        assert row["stream_seeds"]["reveal"] == gen.stream_seed("unit-test-only", i, "reveal")


def test_ten_consecutive_rejections_preserved_without_cleanup_generation():
    decoder = gen.ChainDecoder(mask_id=4, top_k=1)
    decoder.prepare()
    calls = []
    def constant(tokens, t):
        calls.append(t)
        output = torch.full((*tokens.shape, 5), -torch.inf)
        output[..., 0] = 0
        return output
    result = gen.generate_saved(constant, decoder, Tokenizer(), samples=2,
        stream_namespace="unit-test-only", steps=1, length=16)
    assert result["status"] == "retry_limit" and len(result["attempts"]) == 10
    assert len(calls) == 10
    assert all(not r["accepted"] and r["token_ids"] == [0] * 16 and r["text"] is None
               and r["rejection_reason"] == "last256_entropy_below_4" for r in result["attempts"])


def test_attempt_timer_includes_decode_and_token_diagnostics(monkeypatch):
    decoder = gen.ChainDecoder(mask_id=65, top_k=64)
    decoder.prepare()
    clock = [0.]
    monkeypatch.setattr(gen.time, "perf_counter", lambda: clock[0])
    def forward(tokens, t):
        clock[0] += 1.
        return high_entropy_forward(tokens, t)
    original_decode, original_diagnostics = gen.decode_retaining_eos, gen.token_diagnostics
    def decode(*args):
        clock[0] += 2.
        return original_decode(*args)
    def diagnostics(*args):
        clock[0] += 3.
        return original_diagnostics(*args)
    monkeypatch.setattr(gen, "decode_retaining_eos", decode)
    monkeypatch.setattr(gen, "token_diagnostics", diagnostics)
    row = gen.generate_attempt(forward, decoder, Tokenizer(), attempt_id=0,
        stream_namespace="timing-test", steps=1, length=256)
    assert row["accepted"] and row["generation_seconds"] == 6.


def test_rejection_counter_resets_after_acceptance(monkeypatch):
    decoder = gen.ChainDecoder(mask_id=4)
    decoder.prepare()
    def attempt(*args, attempt_id, **kwargs):
        return {"attempt_id": attempt_id, "accepted": attempt_id in (9, 19)}
    monkeypatch.setattr(gen, "generate_attempt", attempt)
    result = gen.generate_saved(None, decoder, Tokenizer(), samples=2, stream_namespace="test")
    assert result["status"] == "complete" and len(result["attempts"]) == 20


def test_fixed_raw_population_never_replaces_or_stops_after_ten_rejections(monkeypatch):
    decoder = gen.ChainDecoder(mask_id=4)
    decoder.prepare()
    seen = []
    def attempt(*args, attempt_id, **kwargs):
        seen.append((attempt_id, kwargs))
        return {"attempt_id": attempt_id, "accepted": False, "generation_seconds": .5}
    monkeypatch.setattr(gen, "generate_attempt", attempt)
    result = gen.generate_saved(None, decoder, Tokenizer(), samples=64,
        stream_namespace="raw-test", warmup=True, population="raw")
    assert result["schema"] == "additive_chain_saved_raw_generations_v1"
    assert result["population"] == "raw" and result["status"] == "complete"
    assert result["samples_requested"] == result["raw_samples_requested"] == 64
    assert result["accepted_samples_requested"] is None
    assert result["generation_protocol"]["max_consecutive_rejections"] is None
    assert [r["attempt_id"] for r in result["attempts"]] == list(range(64))
    assert len(seen) == 65 and seen[0][0] == 1_000_000_000
    assert all(kw["stream_namespace"] == "raw-test" for _, kw in seen)


def test_raw_stopping_rule_preserves_attempt_tokens_streams_and_flags():
    decoder = gen.ChainDecoder(mask_id=65, top_k=64)
    decoder.prepare()
    arguments = dict(samples=2, stream_namespace="same-stream", steps=4, length=256)
    accepted = gen.generate_saved(high_entropy_forward, decoder, Tokenizer(), **arguments)
    raw = gen.generate_saved(high_entropy_forward, decoder, Tokenizer(), population="raw", **arguments)
    for left, right in zip(accepted["attempts"], raw["attempts"]):
        for key in ("token_ids", "accepted", "text", "stream_seeds", "window", "attempt_id"):
            assert left[key] == right[key]
    assert accepted["accepted_samples_requested"] == 2
    assert accepted["generation_protocol"]["max_consecutive_rejections"] == 10


def test_invalid_population_rejected_before_warmup(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not generate")
    monkeypatch.setattr(gen, "generate_attempt", forbidden)
    with pytest.raises(ValueError, match="population"):
        gen.generate_saved(None, None, None, samples=2, stream_namespace="test",
                           warmup=True, population="unspecified")
