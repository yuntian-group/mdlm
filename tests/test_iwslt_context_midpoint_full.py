from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from additive_chain import iwslt_context_midpoint_full as full
from additive_chain.backbone import file_sha256
from additive_chain.iwslt_confirm import atomic_json, read


class TinyModel:
    def source_features(self, rows, device):
        assert all(set(row) == {"source_ids"} for row in rows)
        return torch.zeros(len(rows), 2)

    def __call__(self, tokens, valid):
        return SimpleNamespace(log_probs=torch.zeros(*tokens.shape, 5), hidden=torch.zeros(*tokens.shape, 2))


class TinyLength:
    def __init__(self, size=8):
        self.size = size

    def __call__(self, features):
        value = torch.zeros(1, 32)
        value[0, self.size] = 1
        return value


class TinyDecoder:
    device, mask_id = torch.device("cpu"), 4

    def __init__(self):
        self.contextual_strength, self.sampling = 1 / .85, "joint"
        self.potential, self.calls, self.closed = object(), [], False

    def reveal(self, output, tokens, selected, generator):
        self.calls.append((self.sampling, self.contextual_strength))
        result = tokens.clone()
        values = torch.randint(0, 3, (int(selected.sum()),), generator=generator)
        offset = round(self.contextual_strength * .85 * 2) + (self.sampling == "own-marginal")
        result[selected] = (values + offset) % 3
        return result

    def close(self):
        self.closed = True


class TinyTokenizer:
    def decode(self, ids):
        return " ".join(("one", "two", "three", "EOS")[i] for i in ids)


@pytest.fixture
def toy(monkeypatch):
    for module in (full.midpoint, full.crossover):
        monkeypatch.setattr(module, "MASK_ID", 4)
        monkeypatch.setattr(module, "EOS_ID", 3)
    return dict(id="dev-x", source_ids=[1, 2], target_ids=[2, 1], target="one two three")


def decode(row, index, sampling="joint", strength=.5, size=8):
    return full.crossover.attach_text(full.decode_one(row, index, TinyModel(), TinyLength(size),
        TinyDecoder(), sampling, strength), row, TinyTokenizer())


@pytest.mark.parametrize("sampling", full.SAMPLINGS)
@pytest.mark.parametrize("index", [0, 1, 17, 501, 7000])
@pytest.mark.parametrize("size", [1, 8, 24])
def test_sampling_passes_through_both_scheduled_steps(toy, sampling, index, size):
    decoder = TinyDecoder()
    got = full.decode_one(toy, index, TinyModel(), TinyLength(size), decoder, sampling)
    direct = TinyDecoder()
    direct.sampling = sampling
    expected = full.midpoint.decode_one(toy, index, TinyModel(), TinyLength(size), direct, .5)
    assert {key: value for key, value in got.items() if key not in ("sampling", "reuse_source", "seconds")} == {
        key: value for key, value in expected.items() if key != "seconds"}
    assert all(method == sampling for method, _ in decoder.calls)
    assert decoder.calls == [(sampling, .5 / .85 if state["scheduled_step"] == 1 else 0.)
                             for state in got["intermediate_states"]]
    assert decoder.sampling == "joint" and decoder.contextual_strength == 1 / .85
    assert got["stream_index"] == index and got["sampling"] == sampling


def test_gold_never_changes_decoding_and_stream_index_does(toy):
    first = decode(toy, 7000, "own-marginal")
    changed = decode(dict(toy, target_ids=[0] * 100, target="entirely unrelated"), 7000, "own-marginal")
    assert first["token_ids"] == changed["token_ids"]
    assert first["predicted_length"] == changed["predicted_length"]
    assert first["token_ids"] != decode(toy, 0, "own-marginal")["token_ids"]
    assert first["token_ids"] != decode(toy, 7000, "joint")["token_ids"]


def test_sampling_restored_on_exception(toy, monkeypatch):
    decoder = TinyDecoder()
    def fail(*unused):
        assert decoder.sampling == "own-marginal"
        raise RuntimeError("inference failed")
    monkeypatch.setattr(full.midpoint, "decode_one", fail)
    with pytest.raises(RuntimeError, match="failed"):
        full.decode_one(toy, 0, TinyModel(), TinyLength(), decoder, "own-marginal")
    assert decoder.sampling == "joint"
    with pytest.raises(ValueError, match="Unknown"):
        full.decode_one(toy, 0, TinyModel(), TinyLength(), decoder, "invalid")


@pytest.fixture
def run_fixture(toy, tmp_path, monkeypatch):
    rows = [dict(toy, id=f"dev-{i}") for i in range(6)]
    positions = {row["id"]: index for index, row in enumerate(rows)}
    selected = [rows[i] for i in (1, 2, 4, 5)]
    endpoints = {method: [decode(row, index, method, 0.) for index, row in enumerate(rows)] for method in full.SAMPLINGS}
    subset = {row["id"]: {key: value for key, value in decode(row, positions[row["id"]]).items()
                         if key not in ("sampling", "reuse_source")} for row in selected}
    previous = SimpleNamespace(base=SimpleNamespace(rows=selected),
        endpoints={"1": [decode(row, positions[row["id"]], strength=1.) for row in selected]})
    base = SimpleNamespace(rows=rows, positions=positions)
    state = SimpleNamespace(base=base, previous=previous, endpoints=endpoints, subset=subset,
        identity={"fixed": "assets", "ids": list(positions)})
    args = SimpleNamespace(output=tmp_path / "full", max_new_examples=1, sampling="joint")
    runtimes, calls = [], []
    def runtime(*unused):
        value = TinyModel(), TinyTokenizer(), TinyLength(), TinyDecoder()
        runtimes.append(value)
        return value
    real_decode = full.decode_one
    def tracked(row, index, *rest):
        calls.append((row["id"], index, rest[-2:]))
        return real_decode(row, index, *rest)
    monkeypatch.setattr(full, "make_runtime", runtime)
    monkeypatch.setattr(full, "decode_one", tracked)
    monkeypatch.setattr(full, "unchanged", lambda *unused: None)
    return args, state, runtimes, calls


def test_resume_reuses_joint_subset_but_generates_every_own_record(run_fixture):
    args, state, runtimes, calls = run_fixture
    with pytest.raises(ValueError, match="preflight"):
        full.run_policy(args, state)
    full.run_preflight(args, state)
    assert len(calls) == 16
    full.run_preflight(args, state)
    assert len(calls) == 16
    full.run_policy(args, state)
    assert len(calls) == 17
    progress = read(args.output / "joint" / "progress.json")
    assert progress["completed"] == 5 and not progress["complete"]
    subset_path = args.output / "joint" / "records" / "00001.json"
    original_sha = file_sha256(subset_path)
    assert read(subset_path)["record"]["reuse_source"] == "previous-512-midpoint"
    args.max_new_examples = 0
    full.run_policy(args, state)
    assert len(calls) == 18 and file_sha256(subset_path) == original_sha
    full.run_policy(args, state)
    assert len(calls) == 18
    args.sampling = "own-marginal"
    args.max_new_examples = 1
    full.run_policy(args, state)
    with pytest.raises(ValueError, match="every full-development"):
        full.run_readout(args, state)
    args.max_new_examples = 0
    full.run_policy(args, state)
    assert len(calls) == 24
    full.run_readout(args, state)
    document = read(args.output / "readout.json")
    assert document["complete"] and document["examples"] == 6 and document["reused_joint_examples"] == 4
    assert set(document["metrics"]) == {"global-joint", "global-own-marginal", "half-joint", "half-own-marginal"}
    assert document["contrasts"]["half_joint_minus_own"]["bleu"] == (
        document["metrics"]["half-joint"]["bleu"] - document["metrics"]["half-own-marginal"]["bleu"])
    original_sha = file_sha256(args.output / "readout.json")
    full.run_readout(args, state)
    assert file_sha256(args.output / "readout.json") == original_sha
    assert all(runtime[-1].closed for runtime in runtimes)


@pytest.mark.parametrize("field,value", [("sampling", "own-marginal"), ("reuse_source", "new"),
    ("stream_index", 0), ("contextual_strengths", [1., 0.]), ("hypothesis", "modified")])
def test_reused_records_require_exact_original_identity(run_fixture, field, value):
    args, state, _, _ = run_fixture
    row = state.base.rows[1]
    record = dict(state.subset[row["id"]], sampling="joint", reuse_source="previous-512-midpoint")
    full.validate_record(record, row, 1, "joint", state)
    record[field] = value
    with pytest.raises(ValueError):
        full.validate_record(record, row, 1, "joint", state)


def test_mutated_record_and_receipt_cannot_resume(run_fixture):
    args, state, _, _ = run_fixture
    full.run_preflight(args, state)
    full.run_policy(args, state)
    path = args.output / "joint" / "records" / "00000.json"
    document = read(path)
    document["record"]["sampling"] = "own-marginal"
    atomic_json(path, document)
    with pytest.raises(ValueError, match="hash/identity"):
        full.validate_saved(args, state, "joint")
    document["record_sha256"] = full.replay.digest(document["record"])
    atomic_json(path, document)
    with pytest.raises(ValueError, match="selector"):
        full.validate_saved(args, state, "joint")
    receipt = read(args.output / "endpoint-parity.json")
    receipt["checks"][0]["sampling"] = "own-marginal"
    atomic_json(args.output / "endpoint-parity.json", receipt)
    with pytest.raises(ValueError, match="preflight"):
        full.parity_receipt(args, state)


@pytest.mark.parametrize("field,value", [("reference", "wrong"), ("predicted_length", 10),
    ("schedule", []), ("opportunities", []), ("length_prediction_forwards", 2)])
def test_complete_schedule_comparison(run_fixture, field, value):
    _, state, _, _ = run_fixture
    full.require_same_schedule(state.endpoints["joint"], state.endpoints["own-marginal"])
    own = deepcopy(state.endpoints["own-marginal"])
    own[0][field] = value
    with pytest.raises(ValueError, match="schedules"):
        full.require_same_schedule(state.endpoints["joint"], own)


def test_preflight_requires_saved_own_endpoint_not_joint(run_fixture):
    args, state, _, _ = run_fixture
    state.endpoints["own-marginal"] = state.endpoints["joint"]
    with pytest.raises(ValueError, match="endpoint tokens"):
        full.run_preflight(args, state)
    assert not (args.output / "endpoint-parity.json").exists()


@pytest.fixture
def prepare_fixture(run_fixture, tmp_path, monkeypatch):
    args, state, _, _ = run_fixture
    args.midpoint_root, args.crossover_root, args.data = tmp_path / "midpoint", tmp_path / "crossover", tmp_path / "data"
    state.previous.base = SimpleNamespace(rows=[row for row in state.base.rows if row["id"] in state.subset],
        positions=state.base.positions, common={"original": "assets"}, identity={"selected_global": {"name": "selected"}})
    state.previous.identity = {"old": "fixed-midpoint"}
    state.previous.endpoints["0"] = [state.endpoints["joint"][state.base.positions[row["id"]]] for row in state.previous.base.rows]
    paths = {}
    for row in state.previous.base.rows:
        path = args.midpoint_root / full.midpoint.POLICY / "records" / f"{state.base.positions[row['id']]:05d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, state.subset[row["id"]])
        paths[str(path.relative_to(args.midpoint_root))] = file_sha256(path)
    atomic_json(args.midpoint_root / "endpoint-parity.json", {"previous": "parity"})
    document = dict(identity=state.previous.identity, complete=True, examples=4, outputs=paths,
        endpoint_parity_sha256=file_sha256(args.midpoint_root / "endpoint-parity.json"),
        metrics={key: full.crossover.corpus_metrics(rows) for key, rows in {
            **state.previous.endpoints, "0.5": [state.subset[row["id"]] for row in state.previous.base.rows]}.items()})
    atomic_json(args.midpoint_root / "readout.json", document)
    endpoint_path = tmp_path / "endpoint.json"
    atomic_json(endpoint_path, {"old": "full-development"})
    monkeypatch.setattr(full.midpoint, "prepare", lambda unused: state.previous)
    monkeypatch.setattr(full.midpoint, "parity_receipt", lambda *unused: None)
    monkeypatch.setattr(full.midpoint, "validate_saved", lambda *unused: state.subset)
    monkeypatch.setattr(full.replay, "EXAMPLES", 4)
    monkeypatch.setattr(full.replay, "FULL_EXAMPLES", 6)
    monkeypatch.setattr(full.evaluation, "development_rows", lambda path, population: state.base.rows if population == "full" else None)
    monkeypatch.setattr(full.crossover, "load_endpoint", lambda *unused: (state.endpoints["joint"], {"joint": endpoint_path}))
    monkeypatch.setattr(full, "load_global_own", lambda *unused: (state.endpoints["own-marginal"], {"own": endpoint_path}))
    return args, state, document


def test_prepare_uses_entire_original_full_order(prepare_fixture):
    args, original, _ = prepare_fixture
    state = full.prepare(args)
    assert len(state.base.rows) == 6 and len(state.subset) == 4
    assert state.identity["ids"] == [row["id"] for row in original.base.rows]
    assert state.identity["stream_indices"] == list(range(6))
    assert state.identity["protocol"]["samplings"] == ["joint", "own-marginal"]
    assert state.identity["protocol"]["contextual_strengths"] == [.5, 0.]
    assert set(state.endpoints) == {"joint", "own-marginal"}


@pytest.mark.parametrize("mutation", ["incomplete", "metric", "hash", "population", "stream", "scope"])
def test_prepare_rejects_subset_or_full_population_changes(prepare_fixture, mutation):
    args, state, document = prepare_fixture
    if mutation == "incomplete":
        document["complete"] = False
    elif mutation == "metric":
        document["metrics"]["0.5"]["bleu"] += 1
    elif mutation == "hash":
        document["outputs"] = {}
    elif mutation == "population":
        state.base.rows[0] = state.base.rows[1]
    elif mutation == "stream":
        state.previous.base.positions["dev-0"] = 50
    elif mutation == "scope":
        args.output = args.midpoint_root / "nested"
    atomic_json(args.midpoint_root / "readout.json", document)
    with pytest.raises(ValueError):
        full.prepare(args)


def test_own_endpoint_authentication_uses_own_cell(run_fixture, tmp_path, monkeypatch):
    args, state, _, _ = run_fixture
    args.global_own_result, args.global_readout = tmp_path / "result.json", tmp_path / "readout.json"
    atomic_json(args.global_own_result, {"saved": "own"})
    candidate = {"name": "global-selected"}
    state.base.identity = {"selected_global": candidate}
    cell = next(c for c in full.evaluation.cells("full") if c["steps"] == 2 and c["sampling"] == "own-marginal")
    outputs = {f"{candidate['name']}/full/{full.output_name(c)}/result.json": "other" for c in full.evaluation.cells("full")}
    outputs[f"{candidate['name']}/full/{full.output_name(cell)}/result.json"] = file_sha256(args.global_own_result)
    metrics = full.crossover.corpus_metrics(state.endpoints["own-marginal"])
    document = dict(outputs=outputs, steps={"2": {"methods": {"own-marginal": metrics}}})
    atomic_json(args.global_readout, document)
    atomic_json(args.global_own_result.with_name("summary.json"), dict(cell=cell, completed=1, expected=1))
    seen = []
    def identity(base, policy, doc, requested_cell, rows):
        seen.append(requested_cell)
        return {"cell": requested_cell}
    monkeypatch.setattr(full.crossover, "endpoint_identity", identity)
    monkeypatch.setattr(full, "verify_output", lambda path, identity: dict(predictions=state.endpoints["own-marginal"], summary=metrics))
    monkeypatch.setattr(full, "verify_population", lambda *unused: None)
    records, _ = full.load_global_own(args, state.base, state.base.rows)
    assert records == state.endpoints["own-marginal"] and seen == [cell]
    document["outputs"].pop(next(iter(document["outputs"])))
    atomic_json(args.global_readout, document)
    with pytest.raises(ValueError, match="authenticated"):
        full.load_global_own(args, state.base, state.base.rows)
