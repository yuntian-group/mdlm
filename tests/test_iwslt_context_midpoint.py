from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from additive_chain import iwslt_context_midpoint as midpoint
from additive_chain.backbone import file_sha256
from additive_chain.generation import stream_seed
from additive_chain.iwslt_confirm import atomic_json, read


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.sources, self.inputs = [], []

    def source_features(self, sources, device):
        assert all(set(source) == {"source_ids"} for source in sources)
        self.sources.extend(deepcopy(sources))
        return torch.zeros(len(sources), 2, device=device)

    def forward(self, tokens, valid):
        assert bool(valid.all())
        self.inputs.append(tokens.tolist())
        return SimpleNamespace(log_probs=torch.zeros(*tokens.shape, 5), hidden=torch.zeros(*tokens.shape, 2))


class TinyLength:
    def __init__(self, length=8):
        self.length = length

    def __call__(self, features):
        logits = torch.zeros(features.shape[0], 32)
        logits[:, self.length] = 1.
        return logits


class TinyDecoder:
    device = torch.device("cpu")
    mask_id = 4

    def __init__(self):
        self.contextual_strength = 1. / .85
        self.potential = object()
        self.calls = []
        self.closed = False

    def reveal(self, output, tokens, selected, generator):
        self.calls.append(self.contextual_strength)
        result = tokens.clone()
        values = torch.randint(0, 3, (int(selected.sum()),), generator=generator)
        values = (values + round(self.contextual_strength * .85 * 2)) % 3
        result[selected] = values
        return result

    def close(self):
        self.closed = True


class TinyTokenizer:
    def decode(self, ids):
        return " ".join(("one", "two", "three", "EOS")[i] for i in ids)


@pytest.fixture
def toy(monkeypatch):
    for module in (midpoint, midpoint.crossover):
        monkeypatch.setattr(module, "MASK_ID", 4)
        monkeypatch.setattr(module, "EOS_ID", 3)
    return dict(id="dev-x", source_ids=[1, 2], target_ids=[2, 1], target="one two three")


def saved_endpoint(row, position, strength, length=8):
    policy = "global-global" if strength == 0 else "contextual-global"
    result = midpoint.crossover.decode_one(row, position, TinyModel(), TinyLength(length), TinyDecoder(), policy)
    return midpoint.crossover.attach_text(result, row, TinyTokenizer())


@pytest.mark.parametrize("position", range(10))
@pytest.mark.parametrize("length", [1, 8, 24])
@pytest.mark.parametrize("strength", [0., 1.])
def test_numeric_endpoints_exactly_reproduce_frozen_switching(toy, position, length, strength):
    decoder = TinyDecoder()
    result = midpoint.crossover.attach_text(midpoint.decode_one(toy, position, TinyModel(), TinyLength(length), decoder, strength), toy, TinyTokenizer())
    endpoint = saved_endpoint(toy, position, strength, length)
    midpoint.validate_record(result, toy, position, strength, endpoint)
    for key in ("token_ids", "raw_target_tokens", "schedule", "reveal_counts", "opportunities", "hypothesis"):
        assert result[key] == endpoint[key]
    assert decoder.contextual_strength == 1. / .85


def test_half_strength_is_distinct_and_does_not_leak_gold(toy):
    model, decoder = TinyModel(), TinyDecoder()
    half = midpoint.crossover.attach_text(midpoint.decode_one(toy, 37, model, TinyLength(), decoder, .5), toy, TinyTokenizer())
    zero, one = saved_endpoint(toy, 37, 0), saved_endpoint(toy, 37, 1)
    midpoint.validate_record(half, toy, 37, .5, zero)
    assert half["raw_target_tokens"] != zero["raw_target_tokens"] != one["raw_target_tokens"]
    assert half["raw_target_tokens"] != one["raw_target_tokens"]
    assert decoder.calls == [.5 / .85, 0.]
    assert half["contextual_strengths"] == [.5, 0.]
    assert [s["applied_contextual_strength"] for s in half["intermediate_states"]] == [.5 / .85, 0.]
    assert all(tokens[0][:4] == [3, 1, 2, 3] for tokens in model.inputs)
    altered = dict(toy, target_ids=[0] * 100, target="completely different reference")
    repeated = midpoint.decode_one(altered, 37, model, TinyLength(), decoder, .5)
    assert half["raw_target_tokens"] == repeated["raw_target_tokens"]
    assert half["predicted_length"] == repeated["predicted_length"] == 8


@pytest.mark.parametrize("first_empty", [True, False])
def test_empty_reveals_use_scheduled_time(toy, first_empty):
    position = next(i for i in range(100) if (torch.rand((1, 1), dtype=torch.float64,
        generator=torch.Generator().manual_seed(stream_seed("iwslt-101", i, "reveal"))).item() >= .5) == first_empty)
    decoder = TinyDecoder()
    result = midpoint.decode_one(toy, position, TinyModel(), TinyLength(1), decoder, .5)
    assert decoder.calls == [0. if first_empty else .5 / .85]
    assert result["reveal_counts"] == ([0, 1] if first_empty else [1, 0])
    assert result["intermediate_states"][0]["scheduled_step"] == (2 if first_empty else 1)


def test_strength_restore_cache_identity_and_invalid_inputs(monkeypatch):
    decoder = TinyDecoder()
    cache = decoder.potential
    switching = midpoint.ScheduledDecoder(decoder, .5)
    def fail(*unused):
        assert decoder.contextual_strength == .5 / .85
        raise RuntimeError("sampling failure")
    monkeypatch.setattr(decoder, "reveal", fail)
    with pytest.raises(RuntimeError, match="sampling failure"):
        switching.reveal(SimpleNamespace(scheduled_time=1.), torch.tensor([[4]]), torch.tensor([[True]]), None)
    assert decoder.contextual_strength == 1. / .85 and decoder.potential is cache
    with pytest.raises(ValueError, match="two-step"):
        switching.reveal(SimpleNamespace(scheduled_time=.25), None, None, None)
    for strength in (-1, .2, 2, float("nan")):
        with pytest.raises(ValueError, match="registered"):
            midpoint.ScheduledDecoder(decoder, strength)
    decoder.contextual_strength = 1.
    with pytest.raises(ValueError, match="coefficient"):
        midpoint.ScheduledDecoder(decoder, .5)


@pytest.mark.parametrize("field,value", [("contextual_strengths", [1., 0.]), ("stream_index", 0),
    ("predicted_length", 9), ("token_ids", []), ("reveal_counts", [3, 5]), ("reference", "changed")])
def test_record_validation_rejects_protocol_changes(toy, field, value):
    result = midpoint.crossover.attach_text(midpoint.decode_one(toy, 37, TinyModel(), TinyLength(), TinyDecoder(), .5), toy, TinyTokenizer())
    assert result[field] != value
    result[field] = value
    with pytest.raises(ValueError):
        midpoint.validate_record(result, toy, 37, .5, saved_endpoint(toy, 37, 0))


def test_state_validation_rejects_coefficient_or_hidden_token_changes(toy):
    result = midpoint.crossover.attach_text(midpoint.decode_one(toy, 37, TinyModel(), TinyLength(), TinyDecoder(), .5), toy, TinyTokenizer())
    bad = deepcopy(result)
    bad["intermediate_states"][0]["applied_contextual_strength"] = .5
    with pytest.raises(ValueError, match="coefficient"):
        midpoint.validate_record(bad, toy, 37, .5, saved_endpoint(toy, 37, 0))
    bad = deepcopy(result)
    state = bad["intermediate_states"][0]
    unselected = next(i for i in range(8) if i not in state["selected_positions"])
    state["output_tokens"][unselected] = 0
    with pytest.raises(ValueError, match="unselected"):
        midpoint.validate_record(bad, toy, 37, .5, saved_endpoint(toy, 37, 0))


def test_record_rejects_same_count_different_reveal_positions(toy):
    result = midpoint.crossover.attach_text(midpoint.decode_one(toy, 37, TinyModel(), TinyLength(), TinyDecoder(), .5), toy, TinyTokenizer())
    endpoint = saved_endpoint(toy, 37, 1)
    first = endpoint["intermediate_states"][0]["selected_positions"]
    second = endpoint["intermediate_states"][1]["selected_positions"]
    first[0], second[0] = second[0], first[0]
    first.sort()
    second.sort()
    with pytest.raises(ValueError, match="reveal positions"):
        midpoint.validate_record(result, toy, 37, .5, endpoint)


@pytest.fixture
def run_fixture(toy, tmp_path, monkeypatch):
    rows = [dict(toy, id=f"dev-{i}") for i in range(4)]
    positions = {row["id"]: 37 + i * 13 for i, row in enumerate(rows)}
    endpoints = {str(strength): [saved_endpoint(row, positions[row["id"]], strength) for row in rows]
                 for strength in (0, 1)}
    base = SimpleNamespace(rows=rows, positions=positions)
    state = SimpleNamespace(base=base, endpoints=endpoints, identity={"fixed": "assets", "ids": list(positions)})
    args = SimpleNamespace(output=tmp_path / "midpoint", max_new_examples=1)
    runtimes, decodes = [], []
    def runtime(*unused):
        value = TinyModel(), TinyTokenizer(), TinyLength(), TinyDecoder()
        runtimes.append(value)
        return value
    real_decode = midpoint.decode_one
    def decode(row, position, *rest):
        decodes.append((row["id"], rest[-1]))
        return real_decode(row, position, *rest)
    monkeypatch.setattr(midpoint, "make_runtime", runtime)
    monkeypatch.setattr(midpoint, "decode_one", decode)
    monkeypatch.setattr(midpoint, "unchanged", lambda *unused: None)
    monkeypatch.setattr(midpoint.replay, "EXAMPLES", len(rows))
    return args, state, runtimes, decodes


def test_preflight_resume_readout_reuses_saved_endpoints(run_fixture):
    args, state, runtimes, decodes = run_fixture
    with pytest.raises(ValueError, match="preflight"):
        midpoint.run_policy(args, state)
    midpoint.run_preflight(args, state)
    assert len(decodes) == 8 and all(runtime[-1].closed for runtime in runtimes)
    midpoint.run_preflight(args, state)
    assert len(decodes) == 8
    midpoint.run_policy(args, state)
    first = args.output / midpoint.POLICY / "records" / "00037.json"
    original_sha = file_sha256(first)
    assert len(decodes) == 9
    with pytest.raises(ValueError, match="every fixed"):
        midpoint.run_readout(args, state)
    args.max_new_examples = 0
    midpoint.run_policy(args, state)
    assert len(decodes) == 12 and file_sha256(first) == original_sha
    midpoint.run_policy(args, state)
    assert len(decodes) == 12
    midpoint.run_readout(args, state)
    document = read(args.output / "readout.json")
    assert document["complete"] and document["examples"] == 4
    assert set(document["metrics"]) == {"0", "0.5", "1"}
    for key in ("0", "1"):
        assert document["metrics"][key] == midpoint.crossover.corpus_metrics(state.endpoints[key])
    for name, key in (("half_minus_zero", "0"), ("half_minus_one", "1")):
        assert document["contrasts"][name]["bleu"] == document["metrics"]["0.5"]["bleu"] - document["metrics"][key]["bleu"]
    before = file_sha256(args.output / "readout.json")
    midpoint.run_readout(args, state)
    assert file_sha256(args.output / "readout.json") == before
    assert all(runtime[-1].closed for runtime in runtimes)


def test_resume_rejects_modified_records_manifest_and_parity(run_fixture):
    args, state, _, _ = run_fixture
    midpoint.run_preflight(args, state)
    midpoint.run_policy(args, state)
    path = args.output / midpoint.POLICY / "records" / "00037.json"
    original = read(path)
    changed = deepcopy(original)
    changed["record"]["contextual_strengths"] = [1., 0.]
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="hash/identity"):
        midpoint.validate_saved(args, state)
    changed["record_sha256"] = midpoint.replay.digest(changed["record"])
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="strength"):
        midpoint.validate_saved(args, state)
    atomic_json(path, original)
    manifest = args.output / midpoint.POLICY / "manifest.json"
    atomic_json(manifest, dict(identity=state.identity, policy="wrong"))
    with pytest.raises(ValueError, match="manifest"):
        midpoint.validate_saved(args, state)
    parity = read(args.output / "endpoint-parity.json")
    parity["first_step_strengths"] = [.5, 1.]
    atomic_json(args.output / "endpoint-parity.json", parity)
    with pytest.raises(ValueError, match="preflight"):
        midpoint.parity_receipt(args, state)


@pytest.fixture
def prepare_fixture(run_fixture, tmp_path, monkeypatch):
    args, state, _, _ = run_fixture
    oldroot = tmp_path / "crossover"
    oldroot.mkdir()
    args.crossover_root = oldroot
    previous_identity = dict(ids=state.identity["ids"], stream_indices=list(state.base.positions.values()), population_sha256="population")
    previous = SimpleNamespace(identity=previous_identity, base=state.base, endpoints={"global-global": state.endpoints["0"]})
    monkeypatch.setattr(midpoint.crossover, "prepare", lambda unused: previous)
    monkeypatch.setattr(midpoint.crossover, "parity_receipt", lambda *unused: None)
    monkeypatch.setattr(midpoint.crossover, "validate_saved", lambda *unused: {r["id"]: r for r in state.endpoints["1"]})
    hashes = {}
    for index, row in enumerate(state.endpoints["1"]):
        path = oldroot / "contextual-global" / "records" / f"{index:05d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, row)
        hashes[str(path.relative_to(oldroot))] = file_sha256(path)
    atomic_json(oldroot / "endpoint-parity.json", {"old": "parity"})
    document = dict(identity=previous_identity, complete=True, examples=4,
        outputs={"contextual-global": hashes, "global-contextual": {}},
        endpoint_parity_sha256=file_sha256(oldroot / "endpoint-parity.json"),
        metrics={"global-global": midpoint.crossover.corpus_metrics(state.endpoints["0"]),
                 "contextual-global": midpoint.crossover.corpus_metrics(state.endpoints["1"]),
                 "global-contextual": {}, "contextual-contextual": {}})
    atomic_json(oldroot / "readout.json", document)
    return args, document


def test_prepare_authenticates_correct_crossover_endpoints(prepare_fixture):
    args, document = prepare_fixture
    state = midpoint.prepare(args)
    assert state.identity["protocol"]["contextual_strengths"] == [.5, 0.]
    assert set(state.endpoints) == {"0", "1"}
    assert state.identity["endpoint_record_sha256"] == document["outputs"]["contextual-global"]
    assert state.identity["crossover_readout_sha256"] == file_sha256(args.crossover_root / "readout.json")


@pytest.mark.parametrize("mutation", ["identity", "incomplete", "hash", "metric", "parity", "scope"])
def test_prepare_rejects_changed_saved_outputs(prepare_fixture, mutation):
    args, document = prepare_fixture
    if mutation == "identity":
        document["identity"] = {"wrong": "assets"}
    elif mutation == "incomplete":
        document["complete"] = False
    elif mutation == "hash":
        document["outputs"]["contextual-global"] = {}
    elif mutation == "metric":
        document["metrics"]["global-global"]["bleu"] += 1
    elif mutation == "parity":
        document["endpoint_parity_sha256"] = "wrong"
    elif mutation == "scope":
        args.output = args.crossover_root / "new"
    atomic_json(args.crossover_root / "readout.json", document)
    with pytest.raises(ValueError):
        midpoint.prepare(args)
