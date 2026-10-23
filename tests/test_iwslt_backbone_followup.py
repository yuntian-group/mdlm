from pathlib import Path
from types import SimpleNamespace

import pytest

from additive_chain import iwslt_backbone_followup as followup
from additive_chain.backbone import file_sha256
from additive_chain.iwslt_confirm import atomic_json, dev_cells, read
from additive_chain.iwslt_followup import calibration_cells, select_settings


def source_selection(data_sha):
    values = []
    for steps in (8, 32):
        for cell in dev_cells(steps) + calibration_cells(steps):
            score = 10. + (cell["temperature"] == .85) + cell["strength"]
            values.append(dict(cell, bleu=score))
    return dict(model_sha256="a" * 64, data_sha256=data_sha,
                cells=values, selected=select_settings(values),
                source_sha256={"historical-evaluator": "b" * 64})


@pytest.fixture
def assets(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "counts.pt").write_bytes(b"counts")
    atomic_json(data / "manifest.json", dict(counts_sha256=file_sha256(data / "counts.pt")))
    trained, length = tmp_path / "trained.pt", tmp_path / "length.pt"
    trained.write_bytes(b"new backbone")
    length.write_bytes(b"new length head")
    source = tmp_path / "old-selection.json"
    atomic_json(source, source_selection(file_sha256(data / "manifest.json")))
    calls = []
    def load_head(path, model, manifest, device):
        calls.append((path, model, manifest, device))
        assert model == file_sha256(trained)
        assert manifest == file_sha256(data / "manifest.json")
        return SimpleNamespace(state_dict=lambda: "new length state")
    monkeypatch.setattr(followup, "load_refit_head", load_head)
    monkeypatch.setattr(followup, "head_fingerprint", lambda state: "c" * 64)
    monkeypatch.setattr(followup, "source_hashes", lambda: {"new-evaluator": "d" * 64})
    return SimpleNamespace(trained=trained, length_head=length, data=data,
        source_selection=source, output=tmp_path / "protocol.json", calls=calls)


def test_freeze_transfers_settings_without_relabeling_selection(assets):
    original = assets.source_selection.read_bytes()
    followup.run_freeze(assets)
    document = read(assets.output)
    assert document["format"] == followup.FORMAT
    assert document["policy"] == "transfer-fixed-settings-without-retuning"
    assert document["source_selection"]["model_sha256"] == "a" * 64
    assert document["model_sha256"] == file_sha256(assets.trained)
    assert document["source_selection"]["sha256"] == file_sha256(assets.source_selection)
    assert document["source_selection"]["source_sha256"] == {"historical-evaluator": "b" * 64}
    assert assets.source_selection.read_bytes() == original
    assert assets.calls and all(call[3] == "cpu" for call in assets.calls)
    followup.run_freeze(assets)  # Idempotent only for the same inputs.
    assets.trained.write_bytes(b"a different backbone")
    with pytest.raises(ValueError, match="frozen"):
        followup.run_freeze(assets)


@pytest.mark.parametrize("damage", ["scores", "data", "strength", "temperature", "model"])
def test_inherited_selection_is_checked_not_forged(damage):
    source = source_selection("data")
    if damage == "scores":
        source["cells"][-1]["bleu"] = 200
    elif damage == "data":
        source["data_sha256"] = "wrong"
    elif damage == "strength":
        for c in source["cells"]:
            c["bleu"] += 20 if c["strength"] == .1 else 0
        source["selected"] = select_settings(source["cells"])
    elif damage == "temperature":
        for c in source["cells"]:
            c["bleu"] += 20 if c["temperature"] == 1. else 0
        source["selected"] = select_settings(source["cells"])
    else:
        source["model_sha256"] = "not a hash"
    with pytest.raises(ValueError):
        followup.check_inherited_settings(source, "data")


@pytest.mark.parametrize("damage", ["source", "model", "length", "data", "counts", "code", "protocol"])
def test_manifest_rejects_changed_inputs(assets, monkeypatch, damage):
    followup.run_freeze(assets)
    args = SimpleNamespace(**dict(vars(assets), protocol=assets.output, stage="count", global_head=None))
    common = followup.common_assets(args)
    assert common["protocol_sha256"] == file_sha256(assets.output)
    if damage == "source":
        data = read(assets.source_selection)
        data["extra"] = "modified"
        atomic_json(assets.source_selection, data)
    elif damage in ("model", "length", "data", "counts"):
        path = {"model": assets.trained, "length": assets.length_head,
                "data": assets.data / "manifest.json", "counts": assets.data / "counts.pt"}[damage]
        if damage == "data":
            data = read(path)
            data["changed"] = True
            atomic_json(path, data)
        else:
            path.write_bytes(b"changed")
    elif damage == "code":
        monkeypatch.setattr(followup, "source_hashes", lambda: {"new-evaluator": "different"})
    else:
        data = read(args.protocol)
        data["protocol"]["temperature"] = 1.
        atomic_json(args.protocol, data)
    with pytest.raises(ValueError):
        followup.common_assets(args)


def test_global_parent_is_only_allowed_and_required_for_contextual(assets):
    followup.run_freeze(assets)
    parent = assets.output.parent / "global.pt"
    parent.write_bytes(b"parent")
    args = SimpleNamespace(**dict(vars(assets), protocol=assets.output, stage="contextual", global_head=None))
    with pytest.raises(ValueError, match="require"):
        followup.common_assets(args)
    args.global_head = parent
    assert followup.common_assets(args)["inherited_global_sha256"] == file_sha256(parent)
    args.stage = "count"
    with pytest.raises(ValueError, match="Only contextual"):
        followup.common_assets(args)


def test_only_one_seed_and_fixed_registered_full_dev_cells():
    assert len(followup.cells("baseline")) == 4
    assert len(followup.cells("count")) == 8
    for stage in followup.STAGES:
        values = followup.cells(stage)
        assert {c["steps"] for c in values} == {2, 4, 8, 32}
        assert {c["seed"] for c in values} == {101}
        assert {c["temperature"] for c in values} == {.85}
        assert {c["strength"] for c in values} == ({0.} if stage == "baseline" else {.2})
    assert followup.cells("global", "selection") == followup.evaluation.cells("selection")
    for stage, population in (("baseline", "selection"), ("count", "selection"),
                              ("test", "full"), ("global", "test")):
        with pytest.raises(ValueError):
            followup.cells(stage, population)


def test_rows_are_dev_only(monkeypatch):
    calls = []
    rows = [dict(id=i) for i in range(7283)]
    monkeypatch.setattr(followup, "ordered_rows", lambda data, split: calls.append(split) or rows)
    monkeypatch.setattr(followup, "stratified_rows", lambda rows, count: rows[:count])
    args = SimpleNamespace(data=Path("data"), stage="global", population="full")
    assert len(followup.rows_for(args)) == 7283
    args.population = "selection"
    assert len(followup.rows_for(args)) == 1024
    assert calls == ["dev", "dev"]
    args.population = "test"
    with pytest.raises(ValueError):
        followup.rows_for(args)
    assert len(calls) == 2


def candidate(epoch):
    return dict(name=f"epoch-{epoch:04d}", head=f"/remote/epoch-{epoch:04d}.pt",
                sha256=str(epoch) * 64, epoch=epoch, update=epoch * 100)


def selection(common, stage="global"):
    candidates = [candidate(0), candidate(1)]
    evaluated = {c["name"]: [dict(cell=cell, bleu=10. - c["epoch"], chrf=30.)
        for cell in followup.cells(stage, "selection")] for c in candidates}
    winner, ranking = followup.evaluation.choose_checkpoint(candidates, evaluated)
    return dict(format=followup.SELECTION_FORMAT, split="dev", stage=stage,
        common=common, manifest=dict(kind=stage, complete=True, candidates=candidates),
        evaluated=evaluated, selected=winner, ranking=ranking)


def test_zero_inclusive_checkpoint_selection_remains_shared():
    document = selection({"model": "new"})
    assert followup.validate_selected(document, {"model": "new"}, "global") == candidate(0)
    with pytest.raises(ValueError, match="different comparison"):
        followup.validate_selected(document, {"model": "old"}, "global")
    document["selected"] = candidate(1)
    with pytest.raises(ValueError, match="criterion"):
        followup.validate_selected(document, {"model": "new"}, "global")


@pytest.mark.parametrize("damage", ["zero", "incomplete", "population", "kind"])
def test_selected_checkpoint_cannot_bypass_controls(damage):
    document = selection({})
    if damage == "zero":
        document["manifest"]["candidates"].pop(0)
    elif damage == "incomplete":
        document["manifest"]["complete"] = False
    elif damage == "population":
        document["split"] = "test"
    else:
        document["manifest"]["kind"] = "contextual"
    with pytest.raises(ValueError):
        followup.validate_selected(document, {}, "global")


def test_learned_head_uses_new_model_binding(monkeypatch, tmp_path):
    path = tmp_path / "epoch-0000.pt"
    path.write_bytes(b"new model residual")
    saved = dict(checkpoint_id="epoch-0000", epoch=0, update=0)
    called = []
    monkeypatch.setattr(followup.evaluation, "load_heads", lambda path, kind, common, parent:
                        called.append((kind, common, parent)) or ("head", None, saved))
    args = SimpleNamespace(stage="global", population="selection", head=path,
                           global_head=None, residual_selection=None)
    _, _, chosen, selected_sha = followup.resolve_head(args, {"model_sha256": "new"})
    assert called == [("global", {"model_sha256": "new"}, None)]
    assert chosen["sha256"] == file_sha256(path) and selected_sha is None
    args.population = "full"
    with pytest.raises(ValueError, match="shared selected"):
        followup.resolve_head(args, {})


def test_count_cannot_silently_consume_a_learned_head():
    args = SimpleNamespace(stage="count", head=Path("old.pt"), residual_selection=None)
    with pytest.raises(ValueError, match="do not consume"):
        followup.resolve_head(args, {})


def test_real_residual_loader_rejects_an_old_backbone_head(tmp_path):
    import torch
    from additive_chain.heads import GlobalAdditiveHead
    from additive_chain.iwslt_residual_assets import PROTOCOL, head_checkpoint
    head = GlobalAdditiveHead(4, 1)
    bound = dict(PROTOCOL, model_sha256="a" * 64, data_sha256="b" * 64,
                 counts_sha256="c" * 64, inherited_global_sha256=None)
    path = tmp_path / "epoch-0000.pt"
    torch.save(head_checkpoint(head, "global", bound, checkpoint_id="epoch-0000",
                             epoch=0, update=0, metrics={}), path)
    args = SimpleNamespace(stage="global", population="selection", head=path,
                           global_head=None, residual_selection=None)
    assert followup.resolve_head(args, bound)[2]["update"] == 0
    with pytest.raises(ValueError, match="model/data/protocol"):
        followup.resolve_head(args, dict(bound, model_sha256="d" * 64))


def test_freeze_rejects_old_backbone_length_head(tmp_path):
    import torch
    from additive_chain.iwslt_length_refit import FORMAT, source_hashes
    data = tmp_path / "data"
    data.mkdir()
    (data / "counts.pt").write_bytes(b"counts")
    atomic_json(data / "manifest.json", dict(counts_sha256=file_sha256(data / "counts.pt")))
    data_sha = file_sha256(data / "manifest.json")
    source = tmp_path / "selection.json"
    atomic_json(source, source_selection(data_sha))
    trained, length = tmp_path / "trained.pt", tmp_path / "length.pt"
    trained.write_bytes(b"new model")
    torch.save(dict(format=FORMAT, identity=dict(model_sha256="a" * 64,
                   data_sha256=data_sha, source_sha256=source_hashes())), length)
    args = SimpleNamespace(data=data, trained=trained, length_head=length, source_selection=source)
    with pytest.raises(ValueError, match="Refit head model/data/source"):
        followup.freeze_document(args)


def test_identity_preserves_new_runner_and_inherited_protocol(monkeypatch):
    monkeypatch.setattr(followup, "source_hashes", lambda: {"new": "code"})
    common = dict(protocol_sha256="new manifest", source_selection_sha256="original selection")
    cell = followup.cells("count")[0]
    result = followup.identity(cell, [dict(id="dev-1")], "count", "full", common, None)
    assert result["common"] == common and result["source_sha256"] == {"new": "code"}
    assert result["split"] == "dev" and result["candidate"] is None
    damaged = dict(cell, seed=202)
    with pytest.raises(ValueError):
        followup.identity(damaged, [], "count", "full", common, None)


def test_checkpoint_outputs_live_in_separate_namespaces(tmp_path):
    cell = followup.cells("count")[0]
    count = followup.folder_for(tmp_path, "count", None, "full", cell)
    learned = followup.folder_for(tmp_path, "global", candidate(0), "full", cell)
    assert count.parent != learned.parent
    assert "fixed" in count.parts and "epoch-0000" in learned.parts


@pytest.fixture
def completed(tmp_path, monkeypatch):
    from sacrebleu.metrics import BLEU, CHRF
    monkeypatch.setattr(followup, "source_hashes", lambda: {"runner": "fixed"})
    common = dict(model_sha256="new model", data_sha256="data", counts_sha256="counts")
    rows = [dict(id=f"dev-{i}", source_ids=[1, 2], target_ids=[3, 4, 5],
                 target="this is a correct translation of the source") for i in range(2)]
    monkeypatch.setattr(followup, "common_assets", lambda args: common)
    monkeypatch.setattr(followup, "rows_for", lambda args: rows)
    args = SimpleNamespace(stage="count", population="full", data=tmp_path,
        input=tmp_path / "results", output=tmp_path / "readout.json", residual_selection=None)

    def write_grid(stage, population="full", entry=None, selected_sha=None):
        for cell in followup.cells(stage, population):
            records = [dict(id=r["id"], hypothesis=r["target"], reference=r["target"],
                predicted_length=4, reveal_counts=[3, 1] + [0] * (cell["steps"] - 2),
                scheduled_steps=cell["steps"], length_prediction_forwards=1,
                actual_forward_calls=2, seconds=1., same_state_probe_seconds=0.,
                opportunities=[dict(masked=4, revealed=3, component_reveals=[3]),
                               dict(masked=1, revealed=1, component_reveals=[1])]) for r in rows]
            hyps, refs = [r["hypothesis"] for r in records], [[r["reference"] for r in records]]
            summary = dict(examples=len(rows),
                bleu=BLEU(lowercase=True, tokenize="13a").corpus_score(hyps, refs).score,
                chrf=CHRF().corpus_score(hyps, refs).score)
            path = followup.folder_for(args.input, stage, entry, population, cell) / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, dict(summary=summary, predictions=records))
            expected = followup.identity(cell, rows, stage, population, common, entry, selected_sha)
            atomic_json(path.with_name("result-identity.json"),
                        dict(identity=expected, output_sha256=file_sha256(path)))
            atomic_json(path.with_name("summary.json"), dict(completed=1, expected=1, cell=cell, **summary))
    write_grid("count")
    write_grid("baseline")
    return SimpleNamespace(args=args, common=common, rows=rows, write_grid=write_grid)


def test_count_and_baseline_readout_recompute_full_grid(completed):
    args = completed.args
    followup.run_readout(args)
    result = read(args.output)
    assert result["split"] == "dev" and result["selected"] is None
    assert len(result["outputs"]) == 8
    assert set(result["steps"]) == {"2", "4", "8", "32"}
    for row in result["steps"].values():
        assert row["joint_minus_own_marginal"] == dict(bleu=0., chrf=0.)
        assert row["methods"]["joint"]["bleu"] == pytest.approx(100.)
    args.stage, args.output = "baseline", args.output.with_name("baseline-readout.json")
    followup.run_readout(args)
    result = read(args.output)
    assert len(result["outputs"]) == 4
    assert all(set(row["methods"]) == {"baseline"} for row in result["steps"].values())


@pytest.mark.parametrize("damage", ["reference", "score", "id", "length", "oracle", "schedule", "receipt", "missing"])
def test_readout_rejects_corrupted_outputs(completed, damage):
    args = completed.args
    path = followup.folder_for(args.input, "count", None, "full", followup.cells("count")[0]) / "result.json"
    if damage == "missing":
        path.parent.rename(path.parent.with_name("not-a-cell"))
    else:
        value = read(path)
        row = value["predictions"][0]
        if damage == "reference":
            row["reference"] = "different reference"
        elif damage == "score":
            value["summary"]["bleu"] = 1.
        elif damage == "id":
            row["id"] = "not-dev"
        elif damage == "length":
            row["predicted_length"] = 5
        elif damage == "oracle":
            row["length_mode"] = "reference-diagnostic"
        elif damage == "schedule":
            row["reveal_counts"] = [1, 3] + [0] * 30
            row["opportunities"] = [dict(masked=4, revealed=1, component_reveals=[1]),
                                    dict(masked=3, revealed=3, component_reveals=[3])]
        atomic_json(path, value)
        receipt = read(path.with_name("result-identity.json"))
        receipt["output_sha256"] = file_sha256(path) if damage != "receipt" else "wrong"
        atomic_json(path.with_name("result-identity.json"), receipt)
        summary = read(path.with_name("summary.json"))
        summary.update(value["summary"])
        atomic_json(path.with_name("summary.json"), summary)
    with pytest.raises(ValueError):
        followup.run_readout(args)


def test_learned_selection_and_readout_share_new_assets(completed, monkeypatch):
    args, saved, candidates = completed.args, {}, []
    for epoch in (0, 1):
        path = args.output.parent / f"epoch-{epoch:04d}.pt"
        path.write_bytes(f"residual-{epoch}".encode())
        saved[str(path)] = dict(checkpoint_id=f"epoch-{epoch:04d}", epoch=epoch, update=100 * epoch)
        entry = followup.evaluation.candidate_entry(path, saved[str(path)])
        candidates.append(entry)
        completed.write_grid("global", "selection", entry)
    monkeypatch.setattr(followup.evaluation, "load_heads", lambda path, *args: (None, None, saved[str(path)]))
    args.stage, args.population, args.global_head = "global", "selection", None
    args.candidates = args.output.with_name("candidates.json")
    atomic_json(args.candidates, dict(kind="global", complete=True, candidates=candidates))
    args.output = args.output.with_name("selected.json")
    followup.run_select(args)
    selected = read(args.output)
    assert selected["selected"] == candidates[0]
    assert selected["ranking"][0]["mean_bleu"] == pytest.approx(100.)
    completed.write_grid("global", "full", candidates[0], file_sha256(args.output))
    args.population, args.residual_selection = "full", args.output
    args.output = args.output.with_name("global-readout.json")
    followup.run_readout(args)
    assert read(args.output)["selected"]["update"] == 0
    assert len(read(args.output)["outputs"]) == 8
