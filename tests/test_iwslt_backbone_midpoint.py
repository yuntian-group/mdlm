from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from additive_chain import iwslt_backbone_midpoint as run
from additive_chain.backbone import file_sha256
from additive_chain.generation import stream_seed
from additive_chain.iwslt_confirm import atomic_json, read


def candidate(epoch):
    return dict(name=f"epoch-{epoch:04d}", head=f"/new/context-{epoch}.pt",
                sha256=str(epoch) * 64, epoch=epoch, update=epoch * 100)


def selected_document(common, winner=0):
    candidates = [candidate(0), candidate(1), candidate(2)]
    scores = {c["name"]: [dict(cell=cell, bleu=20. if c["epoch"] == winner else 10. + c["epoch"], chrf=30.)
              for cell in run.followup.cells("contextual", "selection")] for c in candidates}
    selected, ranking = run.followup.evaluation.choose_checkpoint(candidates, scores)
    return dict(format=run.followup.SELECTION_FORMAT, split="dev", stage="contextual",
        common=common, examples=1024, manifest=dict(kind="contextual", complete=True, candidates=candidates),
        evaluated=scores, selected=selected, ranking=ranking)


def test_zero_winner_uses_ranked_trained_checkpoint_without_changing_official_selection():
    document = selected_document({})
    old = deepcopy(document)
    official, trained, policy = run.choose_trained_context(document, {})
    assert official == candidate(0) and trained == candidate(2)
    assert policy == "highest-ranked-trained-contextual-when-zero-wins"
    assert document == old


def test_trained_official_winner_is_used_by_both_samplers():
    document = selected_document({}, winner=1)
    official, trained, policy = run.choose_trained_context(document, {})
    assert official == trained == candidate(1)
    assert policy == "official-selected-trained-contextual"
    assert len(run.cells()) == 2
    assert {c["steps"] for c in run.cells()} == {2}
    assert [c["sampling"] for c in run.cells()] == ["joint", "own-marginal"]
    assert run.PROTOCOL["contextual_strengths"] == [.5, 0.]


@pytest.mark.parametrize("damage", ["model", "parent", "ranking", "selected", "incomplete", "zero-only"])
def test_midpoint_cannot_bypass_official_selection(damage):
    common = dict(model_sha256="new", inherited_global_sha256="selected global")
    document = selected_document(common)
    if damage == "model":
        common = dict(common, model_sha256="old")
    elif damage == "parent":
        common = dict(common, inherited_global_sha256="other global")
    elif damage == "ranking":
        document["ranking"] = list(reversed(document["ranking"]))
    elif damage == "selected":
        document["selected"] = candidate(1)
    elif damage == "incomplete":
        document["manifest"]["complete"] = False
    else:
        document["manifest"]["candidates"] = [candidate(0)]
        document["evaluated"] = {"epoch-0000": document["evaluated"]["epoch-0000"]}
        document["selected"], document["ranking"] = run.followup.evaluation.choose_checkpoint(
            document["manifest"]["candidates"], document["evaluated"])
    with pytest.raises(ValueError):
        run.choose_trained_context(document, common)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []

    def source_features(self, sources, device):
        assert all(set(source) == {"source_ids"} for source in sources)
        return torch.zeros(len(sources), 2, device=device)

    def forward(self, tokens, valid):
        self.inputs.append(tokens.tolist())
        return SimpleNamespace(log_probs=torch.zeros(*tokens.shape, 5), hidden=torch.zeros(*tokens.shape, 2))


class Length:
    def __init__(self, length=8):
        self.length = length

    def __call__(self, features):
        result = torch.zeros(features.shape[0], 32)
        result[:, self.length] = 1.
        return result


class Decoder:
    device = torch.device("cpu")
    mask_id = 4

    def __init__(self, sampling="joint"):
        self.contextual_strength = 1. / .85
        self.global_strength = 1. / .85
        self.count_strength = .2 / .85
        self.sampling = sampling
        self.calls = []

    def reveal(self, output, tokens, selected, generator):
        self.calls.append((self.sampling, self.contextual_strength, self.global_strength, self.count_strength))
        result = tokens.clone()
        values = torch.randint(0, 3, (int(selected.sum()),), generator=generator)
        result[selected] = (values + round(self.contextual_strength * .85 * 2)) % 3
        return result


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(run.midpoint, "MASK_ID", 4)
    monkeypatch.setattr(run.midpoint, "EOS_ID", 3)
    return dict(id="dev-1", source_ids=[1, 2], target_ids=[2, 1], target="one two three")


def record(row, index, sampling, strength=.5, length=8):
    value = run.midpoint.decode_one(row, index, Model(), Length(length), Decoder(sampling), strength)
    text = " ".join(("one", "two", "three", "EOS")[i] for i in value["token_ids"])
    return dict(value, hypothesis=text, reference=row["target"], sampling=sampling,
                seconds=1., same_state_probe_seconds=0.)


@pytest.mark.parametrize("sampling", run.SAMPLINGS)
@pytest.mark.parametrize("first_empty", [True, False])
def test_empty_first_reveal_uses_scheduled_time_and_preserves_both_step_samplers(toy, sampling, first_empty):
    index = next(i for i in range(100) if (torch.rand((1, 1), dtype=torch.float64,
        generator=torch.Generator().manual_seed(stream_seed("iwslt-101", i, "reveal"))).item() >= .5) == first_empty)
    decoder = Decoder(sampling)
    value = run.midpoint.decode_one(toy, index, Model(), Length(1), decoder, .5)
    assert decoder.calls == [(sampling, 0. if first_empty else .5 / .85, 1. / .85, .2 / .85)]
    assert value["reveal_counts"] == ([0, 1] if first_empty else [1, 0])
    assert decoder.contextual_strength == 1. / .85


@pytest.mark.parametrize("sampling", run.SAMPLINGS)
def test_both_nonempty_steps_keep_sampling_method_and_exact_coefficients(toy, sampling):
    decoder, model = Decoder(sampling), Model()
    value = run.midpoint.decode_one(toy, 37, model, Length(), decoder, .5)
    assert decoder.calls == [(sampling, .5 / .85, 1. / .85, .2 / .85),
                             (sampling, 0., 1. / .85, .2 / .85)]
    changed = dict(toy, target_ids=[0] * 100, target="unseen target")
    repeated = run.midpoint.decode_one(changed, 37, Model(), Length(), Decoder(sampling), .5)
    assert value["raw_target_tokens"] == repeated["raw_target_tokens"]
    assert value["stream_index"] == 37
    assert all(tokens[0][:4] == [3, 1, 2, 3] for tokens in model.inputs)


@pytest.fixture
def outputs(toy, tmp_path, monkeypatch):
    state = SimpleNamespace(rows=[toy], endpoints={s: [record(toy, 0, s, 0.)] for s in run.SAMPLINGS},
        identity=dict(format=run.FORMAT, role="diagnostic", ids=[toy["id"]], protocol=run.PROTOCOL,
                      official_contextual=candidate(0), trained_contextual=candidate(2)))
    args = SimpleNamespace(output=tmp_path)
    monkeypatch.setattr(run, "prepare", lambda args: state)
    monkeypatch.setattr(run, "unchanged", lambda args, state: None)
    for sampling in run.SAMPLINGS:
        path = tmp_path / sampling / "result.json"
        path.parent.mkdir()
        records = [record(toy, 0, sampling)]
        summary = dict(examples=1, role="diagnostic", seconds=1., same_state_probe_seconds=0.,
                       same_state_probability_probes=False, **run.metrics(records))
        atomic_json(path, dict(predictions=records, summary=summary))
        atomic_json(path.with_name("result-identity.json"),
                    dict(identity=run.cell_identity(state, sampling), output_sha256=file_sha256(path)))
        atomic_json(path.with_name("endpoint-parity.json"), run.parity_receipt(state, sampling))
    return args, state


def test_readout_recomputes_scores_and_preserves_diagnostic_label(outputs):
    args, state = outputs
    run.run_readout(args)
    result = read(args.output / "readout.json")
    assert result["complete"] is True and result["role"] == "diagnostic"
    assert result["identity"]["official_contextual"]["update"] == 0
    assert result["identity"]["trained_contextual"]["update"] > 0
    assert set(result["methods"]) == set(run.SAMPLINGS)
    assert result["joint_minus_own_marginal"] == dict(bleu=0., chrf=0.)
    assert "confidence_interval" not in result


@pytest.mark.parametrize("sampling", run.SAMPLINGS)
def test_zero_context_preflight_matches_the_new_global_endpoint(outputs, sampling):
    args, state = outputs
    decoder = Decoder(sampling)
    assert run.check_endpoint_parity(Model(), Length(), decoder, state, sampling) == run.parity_receipt(state, sampling)
    broken = deepcopy(state)
    broken.endpoints[sampling][0]["token_ids"] = []
    with pytest.raises(ValueError, match="new global endpoint"):
        run.check_endpoint_parity(Model(), Length(), decoder, broken, sampling)
    wrong = "own-marginal" if sampling == "joint" else "joint"
    with pytest.raises(ValueError, match="sampler"):
        run.check_endpoint_parity(Model(), Length(), decoder, state, wrong)


@pytest.mark.parametrize("damage", ["score", "coefficient", "sampling", "position", "length", "reference", "schedule", "missing", "role"])
def test_midpoint_readout_rejects_corruption(outputs, damage):
    args, state = outputs
    path = args.output / "joint/result.json"
    value = read(path)
    row = value["predictions"][0]
    if damage == "score":
        value["summary"]["bleu"] = -1.
    elif damage == "coefficient":
        row["contextual_strengths"] = [1., 0.]
    elif damage == "sampling":
        row["sampling"] = "own-marginal"
    elif damage == "position":
        row["stream_index"] = 10
    elif damage == "length":
        row["predicted_length"] += 1
    elif damage == "reference":
        row["reference"] = "changed"
    elif damage == "schedule":
        row["intermediate_states"][0]["applied_contextual_strength"] = .5
    elif damage == "missing":
        value["predictions"] = []
    else:
        value["summary"]["role"] = "selected"
    atomic_json(path, value)
    atomic_json(path.with_name("result-identity.json"),
                dict(identity=run.cell_identity(state, "joint"), output_sha256=file_sha256(path)))
    with pytest.raises(ValueError):
        run.run_readout(args)


@pytest.fixture
def prepared_assets(toy, tmp_path, monkeypatch):
    from additive_chain.heads import GlobalAdditiveHead, ContextualAdditiveHead
    from additive_chain.iwslt_residual_assets import PROTOCOL, head_checkpoint
    rows = [toy]
    monkeypatch.setattr(run, "ordered_rows", lambda data, split: rows if split == "dev" else pytest.fail("not dev"))
    data = tmp_path / "data"
    data.mkdir()
    (data / "counts.pt").write_bytes(b"counts")
    atomic_json(data / "manifest.json", dict(counts_sha256=file_sha256(data / "counts.pt")))
    trained, length, protocol = (tmp_path / name for name in ("trained.pt", "length.pt", "protocol.json"))
    for path in (trained, length, protocol):
        path.write_bytes(path.name.encode())
    common = dict(model_sha256=file_sha256(trained), data_sha256=file_sha256(data / "manifest.json"),
                  counts_sha256=file_sha256(data / "counts.pt"), inherited_global_sha256=None)

    def snapshots(stage, parent=None):
        result = []
        for epoch in (0, 1):
            head = (GlobalAdditiveHead(4, 1) if stage == "global" else
                    ContextualAdditiveHead(torch.zeros(4, 2), hidden_size=2, rank=1, context_size=2))
            path = tmp_path / f"{stage}-{epoch}.pt"
            bound = dict(PROTOCOL, **dict(common, inherited_global_sha256=parent))
            payload = head_checkpoint(head, stage, bound, checkpoint_id=f"epoch-{epoch:04d}",
                                      epoch=epoch, update=epoch * 100, metrics={})
            torch.save(payload, path)
            result.append(run.followup.evaluation.candidate_entry(path, payload))
        return result

    def document(stage, candidates, bound, winner):
        scores = {c["name"]: [dict(cell=cell, bleu=20. if c["epoch"] == winner else 10., chrf=30.)
                  for cell in run.followup.cells(stage, "selection")] for c in candidates}
        selected, ranking = run.followup.evaluation.choose_checkpoint(candidates, scores)
        return dict(format=run.followup.SELECTION_FORMAT, split="dev", stage=stage, common=bound,
            examples=1024, manifest=dict(kind=stage, complete=True, candidates=candidates),
            evaluated=scores, selected=selected, ranking=ranking)

    global_candidates = snapshots("global")
    parent = global_candidates[1]
    context_common = dict(common, inherited_global_sha256=parent["sha256"])
    context_candidates = snapshots("contextual", parent["sha256"])
    def common_assets(args):
        assert args.trained == trained
        if args.stage == "contextual":
            assert args.global_head == Path(parent["head"])
            return context_common
        assert args.stage == "global" and args.global_head is None
        return common
    monkeypatch.setattr(run.followup, "common_assets", common_assets)
    global_selection, context_selection = tmp_path / "global-selected.json", tmp_path / "context-selected.json"
    atomic_json(global_selection, document("global", global_candidates, common, 1))
    atomic_json(context_selection, document("contextual", context_candidates, context_common, 0))
    args = SimpleNamespace(trained=trained, length_head=length, protocol=protocol, data=data,
        global_selection=global_selection, contextual_selection=context_selection,
        global_readout=tmp_path / "global-readout.json", evaluation_root=tmp_path / "evaluations",
        output=tmp_path / "midpoint")
    output_hashes, values = {}, {}
    for cell in run.followup.cells("global"):
        path = run.followup.folder_for(args.evaluation_root, "global", parent, "full", cell) / "result.json"
        relative = str(path.relative_to(args.evaluation_root))
        if cell["steps"] != 2:
            output_hashes[relative] = "unused nonendpoint hash"
            continue
        records = [record(toy, 0, cell["sampling"], 0.)]
        summary = dict(examples=1, **run.metrics(records))
        path.parent.mkdir(parents=True)
        atomic_json(path, dict(predictions=records, summary=summary))
        expected = run.followup.identity(cell, rows, "global", "full", common, parent, file_sha256(global_selection))
        atomic_json(path.with_name("result-identity.json"), dict(identity=expected, output_sha256=file_sha256(path)))
        atomic_json(path.with_name("summary.json"), dict(completed=1, expected=1, cell=cell, **summary))
        output_hashes[relative] = file_sha256(path)
        values[cell["sampling"]] = summary
    atomic_json(args.global_readout, dict(split="dev", stage="global", examples=1, common=common,
        selected=parent, residual_selection_sha256=file_sha256(global_selection),
        source_sha256=run.followup.source_hashes(), outputs=output_hashes, steps={"2": dict(methods=values)}))
    return args, context_candidates, parent


def test_prepare_authenticates_new_heads_selections_and_global_endpoints(prepared_assets):
    args, contextual, global_head = prepared_assets
    state = run.prepare(args)
    assert state.identity["global_candidate"] == global_head
    assert state.identity["official_contextual"] == contextual[0]
    assert state.identity["trained_contextual"] == contextual[1]
    assert state.identity["common"]["inherited_global_sha256"] == global_head["sha256"]
    assert set(state.endpoints) == set(run.SAMPLINGS)
    assert state.identity["role"] == "diagnostic"


@pytest.mark.parametrize("damage", ["backbone", "parent", "ranking", "endpoint", "readout"])
def test_prepare_rejects_stale_backbone_parent_or_changed_saved_evidence(prepared_assets, damage):
    args, contextual, global_head = prepared_assets
    if damage in ("backbone", "parent"):
        path = Path(contextual[1]["head"])
        payload = torch.load(path, weights_only=True)
        key = "model_sha256" if damage == "backbone" else "inherited_global_sha256"
        payload["identity"][key] = "stale"
        torch.save(payload, path)
    elif damage == "ranking":
        document = read(args.contextual_selection)
        document["ranking"].reverse()
        atomic_json(args.contextual_selection, document)
    elif damage == "endpoint":
        path = run.followup.folder_for(args.evaluation_root, "global", global_head, "full", run.cells()[0]) / "result.json"
        payload = read(path)
        payload["predictions"][0]["token_ids"] = []
        atomic_json(path, payload)
    else:
        document = read(args.global_readout)
        document["common"]["model_sha256"] = "old backbone"
        atomic_json(args.global_readout, document)
    with pytest.raises(ValueError):
        run.prepare(args)
