from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from additive_chain import iwslt_context_crossover as crossover
from additive_chain import translation
from additive_chain.backbone import file_sha256
from additive_chain.generation import stream_seed
from additive_chain.iwslt_confirm import atomic_json, read
from additive_chain.iwslt_followup import StateObserver, output_name


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
        return SimpleNamespace(log_probs=torch.zeros(*tokens.shape, 5),
                               hidden=torch.zeros(*tokens.shape, 2))


class TinyLength:
    def __init__(self, length=8):
        self.length = length

    def __call__(self, features):
        logits = torch.zeros(features.shape[0], 16)
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
        # The same random stream with distinct, observable head effects.
        values = (values + int(self.contextual_strength != 0.)) % 3
        result[selected] = values
        return result

    def close(self):
        self.closed = True


class TinyTokenizer:
    def decode(self, ids):
        return " ".join(("one", "two", "three", "EOS")[i] for i in ids)


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(crossover, "MASK_ID", 4)
    monkeypatch.setattr(crossover, "EOS_ID", 3)
    monkeypatch.setattr(translation, "MASK_ID", 4)
    monkeypatch.setattr(translation, "EOS_ID", 3)
    return dict(row=dict(id="dev-x", source_ids=[1, 2], target_ids=[2, 1], target="one two three"),
                model=TinyModel(), length=TinyLength(), decoder=TinyDecoder())


def original_endpoint(row, position, policy, length=8):
    decoder, model = TinyDecoder(), TinyModel()
    decoder.contextual_strength = 0. if policy == "global-global" else 1. / .85
    observer = StateObserver(decoder, probe=False)
    ids, stats = translation.translate_one(model, row, TinyLength(length), decoder,
        steps=2, seed=101, example_id=position, observer=observer)
    return crossover.attach_text(dict(id=row["id"], token_ids=ids,
        opportunities=observer.steps, seconds=0., same_state_probe_seconds=0., **stats), row, TinyTokenizer())


@pytest.mark.parametrize("policy", crossover.ENDPOINTS + crossover.POLICIES)
def test_fixed_protocol_no_gold_leakage_same_rng_and_constant_endpoint_parity(toy, policy):
    row, model, length, decoder = toy.values()
    actual = crossover.attach_text(crossover.decode_one(row, 37, model, length, decoder, policy), row, TinyTokenizer())
    constant = policy.split("-")[0]
    endpoint = original_endpoint(row, 37, constant + "-" + constant)
    crossover.validate_record(actual, row, 37, policy, endpoint)
    if policy in crossover.ENDPOINTS:
        assert actual["token_ids"] == endpoint["token_ids"]
        assert actual["hypothesis"] == endpoint["hypothesis"]
    assert all(values[0][:4] == [3, 1, 2, 3] for values in model.inputs)
    changed_gold = dict(row, target_ids=[0] * 100, target="unrelated reference with another length")
    second = crossover.decode_one(changed_gold, 37, model, length, decoder, policy)
    assert actual["raw_target_tokens"] == second["raw_target_tokens"]
    assert actual["predicted_length"] == 8 != len(changed_gold["target_ids"]) + 1
    assert actual["schedule"] == second["schedule"]
    assert decoder.contextual_strength == 1. / .85


@pytest.mark.parametrize("policy,expected", [("global-contextual", 1. / .85), ("contextual-global", 0.)])
def test_empty_first_reveal_switches_by_time_not_forward_count(toy, policy, expected):
    position = next(i for i in range(100) if torch.rand((1, 1), dtype=torch.float64,
        generator=torch.Generator().manual_seed(stream_seed("iwslt-101", i, "reveal"))).item() >= .5)
    row, model, _, decoder = toy.values()
    record = crossover.decode_one(row, position, model, TinyLength(1), decoder, policy)
    assert record["reveal_counts"] == [0, 1]
    assert decoder.calls == [expected]
    assert [state["scheduled_step"] for state in record["intermediate_states"]] == [2]


@pytest.mark.parametrize("policy,expected", [("global-contextual", 0.), ("contextual-global", 1. / .85)])
def test_empty_second_reveal_does_not_invent_another_forward(toy, policy, expected):
    position = next(i for i in range(100) if torch.rand((1, 1), dtype=torch.float64,
        generator=torch.Generator().manual_seed(stream_seed("iwslt-101", i, "reveal"))).item() < .5)
    row, model, _, decoder = toy.values()
    record = crossover.decode_one(row, position, model, TinyLength(1), decoder, policy)
    assert record["reveal_counts"] == [1, 0]
    assert record["actual_forward_calls"] == 1
    assert decoder.calls == [expected]
    assert [state["scheduled_step"] for state in record["intermediate_states"]] == [1]


def test_switch_keeps_single_count_cache_and_restores_strength_on_failure(toy, monkeypatch):
    decoder = toy["decoder"]
    potential = decoder.potential
    switching = crossover.ScheduledDecoder(decoder, "global-contextual")
    tokens, selected = torch.tensor([[4]]), torch.tensor([[True]])
    switching.reveal(SimpleNamespace(scheduled_time=1.), tokens, selected, torch.Generator())
    switching.reveal(SimpleNamespace(scheduled_time=.5), tokens, selected, torch.Generator())
    assert decoder.calls == [0., 1. / .85]
    assert decoder.potential is potential
    def fail(*args):
        assert decoder.contextual_strength == 0.
        raise RuntimeError("sampling failure")
    monkeypatch.setattr(decoder, "reveal", fail)
    with pytest.raises(RuntimeError, match="sampling failure"):
        switching.reveal(SimpleNamespace(scheduled_time=1.), tokens, selected, torch.Generator())
    assert decoder.contextual_strength == 1. / .85
    with pytest.raises(ValueError, match="two-step"):
        switching.reveal(SimpleNamespace(scheduled_time=.25), tokens, selected, torch.Generator())


def test_first_commit_validation_includes_eos_but_not_unknown_post_eos(toy):
    row, model, length, decoder = toy.values()
    policy = "global-contextual"
    record = crossover.attach_text(crossover.decode_one(row, 37, model, length, decoder, policy), row, TinyTokenizer())
    endpoint = original_endpoint(row, 37, "global-global")
    selected = record["intermediate_states"][0]["selected_positions"]
    assert selected
    bad = deepcopy(endpoint)
    bad["token_ids"][selected[0]] = (bad["token_ids"][selected[0]] + 1) % 3
    with pytest.raises(ValueError, match="First committed"):
        crossover.validate_record(record, row, 37, policy, endpoint, bad)
    bad["token_ids"] = bad["token_ids"][:selected[0]]
    with pytest.raises(ValueError, match="First committed"):
        crossover.validate_record(record, row, 37, policy, endpoint, bad)
    # After a saved endpoint's first EOS, token identities genuinely are absent.
    endpoint["token_ids"] = []
    altered = deepcopy(record)
    first = altered["intermediate_states"][0]
    if 0 in first["selected_positions"]:
        for state in altered["intermediate_states"]:
            if state["scheduled_step"] != 1:
                state["input_tokens"][0] = 3
            state["output_tokens"][0] = 3
        altered["raw_target_tokens"][0] = 3
        altered["token_ids"] = []
    crossover.validate_record(altered, row, 37, policy, endpoint)


@pytest.fixture
def run_fixture(toy, tmp_path, monkeypatch):
    rows = [dict(toy["row"], id=f"dev-{i}") for i in range(4)]
    positions = {row["id"]: 37 + i * 13 for i, row in enumerate(rows)}
    endpoints = {policy: [original_endpoint(row, positions[row["id"]], policy) for row in rows]
                 for policy in crossover.ENDPOINTS}
    base = SimpleNamespace(rows=rows, positions=positions)
    state = SimpleNamespace(base=base, endpoints=endpoints, identity={"fixed": "assets", "ids": list(positions)})
    args = SimpleNamespace(output=tmp_path / "crossover", policy="global-contextual", max_new_examples=1)
    runtimes, decodes = [], []
    def runtime(*unused):
        value = TinyModel(), TinyTokenizer(), TinyLength(), TinyDecoder()
        runtimes.append(value)
        return value
    real_decode = crossover.decode_one
    def decode(row, position, *rest):
        decodes.append((row["id"], rest[-1]))
        return real_decode(row, position, *rest)
    monkeypatch.setattr(crossover, "make_runtime", runtime)
    monkeypatch.setattr(crossover, "decode_one", decode)
    monkeypatch.setattr(crossover, "unchanged", lambda *unused: None)
    monkeypatch.setattr(crossover.replay, "EXAMPLES", len(rows))
    return args, state, runtimes, decodes


def test_preflight_resume_both_complete_readout_and_no_endpoint_rerun(run_fixture):
    args, state, runtimes, decodes = run_fixture
    with pytest.raises(ValueError, match="preflight"):
        crossover.run_policy(args, state)
    crossover.run_preflight(args, state)
    assert len(decodes) == 8
    assert all(runtime[-1].closed for runtime in runtimes)
    crossover.run_preflight(args, state)
    assert len(decodes) == 8
    crossover.run_policy(args, state)
    assert len(decodes) == 9
    first = args.output / args.policy / "records" / "00037.json"
    old_sha = file_sha256(first)
    with pytest.raises(ValueError, match="finish every"):
        crossover.run_readout(args, state)
    args.max_new_examples = 0
    crossover.run_policy(args, state)
    assert file_sha256(first) == old_sha
    assert len(decodes) == 12
    crossover.run_policy(args, state)
    assert len(decodes) == 12
    args.policy = "contextual-global"
    crossover.run_policy(args, state)
    assert len(decodes) == 16
    assert decodes.count((state.base.rows[0]["id"], "global-contextual")) == 1
    crossover.run_readout(args, state)
    result = read(args.output / "readout.json")
    assert result["complete"] and result["examples"] == 4
    assert set(result["metrics"]) == set(crossover.ENDPOINTS + crossover.POLICIES)
    for policy in crossover.ENDPOINTS:
        assert result["metrics"][policy] == crossover.corpus_metrics(state.endpoints[policy])
    metrics = result["metrics"]
    assert result["contrasts"]["interaction"]["bleu"] == pytest.approx(
        metrics["contextual-contextual"]["bleu"] - metrics["contextual-global"]["bleu"]
        - metrics["global-contextual"]["bleu"] + metrics["global-global"]["bleu"])
    old_sha = file_sha256(args.output / "readout.json")
    crossover.run_readout(args, state)
    assert file_sha256(args.output / "readout.json") == old_sha


def test_preflight_rejects_changed_endpoint_tokens_and_closes_decoder(run_fixture):
    args, state, runtimes, _ = run_fixture
    state.endpoints["global-global"][0]["token_ids"][0] = 3
    with pytest.raises(ValueError, match="endpoint|committed"):
        crossover.run_preflight(args, state)
    assert runtimes[-1][-1].closed
    assert not (args.output / "endpoint-parity.json").exists()


def test_tampered_parity_or_resume_record_rejected(run_fixture):
    args, state, _, _ = run_fixture
    crossover.run_preflight(args, state)
    receipt = args.output / "endpoint-parity.json"
    value = read(receipt)
    value["checks"] = []
    atomic_json(receipt, value)
    with pytest.raises(ValueError, match="preflight"):
        crossover.run_policy(args, state)
    value["checks"] = [dict(id=row["id"], policy=policy,
        token_sha256=crossover.replay.digest(state.endpoints[policy][i]["token_ids"]),
        schedule_sha256=crossover.replay.digest(state.endpoints[policy][i]["schedule"]))
        for i, row in enumerate(state.base.rows) for policy in crossover.ENDPOINTS]
    atomic_json(receipt, value)
    crossover.run_policy(args, state)
    path = args.output / args.policy / "records" / "00037.json"
    record = read(path)
    record["record"]["stream_index"] = 0
    atomic_json(path, record)
    with pytest.raises(ValueError, match="hash/identity"):
        crossover.run_policy(args, state)
    record["record_sha256"] = crossover.replay.digest(record["record"])
    atomic_json(path, record)
    with pytest.raises(ValueError, match="population"):
        crossover.run_policy(args, state)


def test_invalid_policy_limit_and_nonempty_output_rejected(run_fixture):
    args, state, _, _ = run_fixture
    args.policy = "global-global"
    with pytest.raises(ValueError, match="registered"):
        crossover.run_policy(args, state)
    args.policy, args.max_new_examples = "global-contextual", -1
    with pytest.raises(ValueError, match="nonnegative"):
        crossover.run_policy(args, state)
    args.max_new_examples = 0
    crossover.run_preflight(args, state)
    folder = args.output / args.policy
    folder.mkdir()
    atomic_json(folder / "unrelated.json", {})
    with pytest.raises(ValueError, match="Nonempty"):
        crossover.run_policy(args, state)


@pytest.fixture
def endpoint_fixture(run_fixture, tmp_path, monkeypatch):
    args, state, _, _ = run_fixture
    full = state.base.rows
    source = {"frozen.py": "a" * 64}
    common = {"inherited_global_sha256": "parent", "fixed": "settings"}
    identity = dict(source_sha256=source, selected_global={"name": "best-loss-0025200", "sha256": "parent"},
        contextual_diagnostic={"name": "epoch-0001", "sha256": "learned"}, official_contextual={"name": "epoch-0000"},
        input_sha256=dict(global_selection="g", contextual_selection="c", reference_output="r", reference_identity="i"),
        frozen_reference_relative_path="epoch-0000/selection/T2/result.json",
        contextual_training_source_sha256={"training.py": "optimized"},
        ids=[full[0]["id"], full[2]["id"]], stream_indices=[37, 63], population_sha256="fixed512")
    base = SimpleNamespace(identity=identity, common=common, rows=full[::2], positions=state.base.positions)
    monkeypatch.setattr(crossover.replay, "FULL_EXAMPLES", len(full))
    monkeypatch.setattr(crossover.evaluation, "source_hashes", lambda: source)
    monkeypatch.setattr(crossover.replay, "prepare", lambda unused: base)
    monkeypatch.setattr(crossover.evaluation, "development_rows", lambda data, population: full)
    args.data = tmp_path / "data"
    for policy in crossover.ENDPOINTS:
        records = state.endpoints[policy]
        metrics = crossover.corpus_metrics(records)
        kind = policy.split("-")[0]
        candidate = identity["selected_global" if kind == "global" else "contextual_diagnostic"]
        doc = dict(split="dev", examples=len(full), source_sha256=source,
            steps={str(step): {"methods": {"joint": metrics}} for step in (2, 4, 8, 32)},
            outputs={f"{candidate['name']}/full/{output_name(cell)}/result.json": "pending"
                     for cell in crossover.evaluation.cells("full")})
        if kind == "global":
            doc.update(kind="global", common=dict(common, inherited_global_sha256=None),
                       selected=candidate, residual_selection_sha256="g")
        else:
            doc.update(format="iwslt_contextual_nonzero_diagnostic_v1", kind="contextual", role="diagnostic",
                policy="highest-ranked-nonzero-from-official-zero-inclusive-ranking", official_selected=identity["official_contextual"],
                diagnostic_candidate=candidate, official_selection_sha256="c", common=common,
                frozen_evaluator_reference=dict(relative_path=identity["frozen_reference_relative_path"], output_sha256="r", identity_sha256="i"),
                runner_sha256="b" * 64, training_source_sha256=identity["contextual_training_source_sha256"])
        cell = next(c for c in crossover.evaluation.cells("full") if c["steps"] == 2 and c["sampling"] == "joint")
        expected = crossover.endpoint_identity(base, policy, doc, cell, full)
        folder = tmp_path / kind
        folder.mkdir()
        result = folder / "result.json"
        atomic_json(result, dict(predictions=records, summary=metrics))
        atomic_json(result.with_name("result-identity.json"), dict(identity=expected, output_sha256=file_sha256(result)))
        atomic_json(result.with_name("summary.json"), dict(cell=cell, completed=1, expected=1))
        doc["outputs"][f"{candidate['name']}/full/{output_name(cell)}/result.json"] = file_sha256(result)
        readout = folder / "full-dev-readout.json"
        atomic_json(readout, doc)
        setattr(args, kind + "_result", result)
        setattr(args, kind + "_readout", readout)
    return args, base, full


def test_authenticated_endpoint_population_subset_and_identity(endpoint_fixture):
    args, base, full = endpoint_fixture
    state = crossover.prepare(args)
    assert len(state.endpoints["global-global"]) == 2
    assert [record["id"] for record in state.endpoints["global-global"]] == [row["id"] for row in full[::2]]
    assert state.identity["endpoint_sha256"]["global_result"] == file_sha256(args.global_result)
    assert state.identity["replay_asset_identity"] == base.identity
    # The subset's corpus metric is recomputed, not copied from the full readout.
    assert crossover.corpus_metrics(state.endpoints["global-global"])["examples"] == 2


@pytest.mark.parametrize("kind,field,value", [
    ("global", "source_sha256", {}), ("global", "selected", {"name": "epoch-0000"}),
    ("global", "common", {}), ("contextual", "common", {}),
    ("contextual", "official_selected", {"name": "epoch-0001"}),
    ("contextual", "diagnostic_candidate", {"name": "epoch-0002"}),
    ("contextual", "training_source_sha256", {}), ("contextual", "frozen_evaluator_reference", {}),
    ("global", "examples", 99), ("contextual", "outputs", {}),
])
def test_endpoint_rejects_changed_source_parent_selection_population(endpoint_fixture, kind, field, value):
    args, base, full = endpoint_fixture
    path = getattr(args, kind + "_readout")
    document = read(path)
    document[field] = value
    atomic_json(path, document)
    with pytest.raises(ValueError, match="Endpoint|endpoint"):
        crossover.load_endpoint(args, base, kind + "-" + kind, full)


def test_endpoint_rejects_resealed_wrong_reference_or_metrics(endpoint_fixture):
    args, base, full = endpoint_fixture
    path = args.global_result
    value = read(path)
    value["predictions"][0]["reference"] = "wrong reference"
    atomic_json(path, value)
    receipt = read(path.with_name("result-identity.json"))
    receipt["output_sha256"] = file_sha256(path)
    atomic_json(path.with_name("result-identity.json"), receipt)
    doc = read(args.global_readout)
    key = next(key for key in doc["outputs"] if "/T2-" in key and "-joint-" in key)
    doc["outputs"][key] = file_sha256(path)
    atomic_json(args.global_readout, doc)
    with pytest.raises(ValueError, match="Reference"):
        crossover.load_endpoint(args, base, "global-global", full)
    value["predictions"][0]["reference"] = full[0]["target"]
    value["summary"]["bleu"] += 1
    atomic_json(path, value)
    receipt["output_sha256"] = file_sha256(path)
    atomic_json(path.with_name("result-identity.json"), receipt)
    doc["outputs"][key] = file_sha256(path)
    atomic_json(args.global_readout, doc)
    with pytest.raises(ValueError, match="metrics"):
        crossover.load_endpoint(args, base, "global-global", full)


def test_endpoint_rejects_missing_registered_cells_and_nonfinite_metric(endpoint_fixture, monkeypatch):
    args, base, full = endpoint_fixture
    doc = read(args.global_readout)
    original = deepcopy(doc)
    key = next(key for key in doc["outputs"] if "/T32-" in key)
    doc["outputs"]["invented/result.json"] = doc["outputs"].pop(key)
    atomic_json(args.global_readout, doc)
    with pytest.raises(ValueError, match="eight registered"):
        crossover.load_endpoint(args, base, "global-global", full)
    original["steps"]["2"]["methods"]["joint"]["bleu"] = float("nan")
    old_read = crossover.read
    monkeypatch.setattr(crossover, "read", lambda path: original if path == args.global_readout else old_read(path))
    with pytest.raises(ValueError, match="metrics"):
        crossover.load_endpoint(args, base, "global-global", full)


def test_paired_endpoints_require_complete_schedule_identity(endpoint_fixture, monkeypatch):
    args, base, full = endpoint_fixture
    original = crossover.load_endpoint
    def changed(*values):
        records, paths = original(*values)
        if values[2] == "contextual-contextual":
            records[0]["schedule"][0]["t"] = .75
        return records, paths
    monkeypatch.setattr(crossover, "load_endpoint", changed)
    with pytest.raises(ValueError, match="complete schedules"):
        crossover.prepare(args)


def test_corpus_metrics_explicit_eos_and_empty_population():
    records = [dict(hypothesis="one two three four", reference="one two three four", token_ids=[1, 2], predicted_length=4),
               dict(hypothesis="one two three", reference="one two three four", token_ids=[1, 2, 3], predicted_length=3)]
    metrics = crossover.corpus_metrics(records)
    assert metrics["early_eos_examples"] == metrics["no_eos_examples"] == 1
    assert metrics["reference_words"] == 8 and metrics["hypothesis_words"] == 7
    assert 0 < metrics["brevity_penalty"] < 1
    with pytest.raises(ValueError, match="empty"):
        crossover.corpus_metrics([])
