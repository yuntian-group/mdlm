import json
from types import SimpleNamespace

import pytest
import torch

from additive_chain.backbone import file_sha256
from additive_chain.iwslt_confirm import atomic_json


@pytest.fixture
def completed_full_dev(tmp_path, monkeypatch):
    from sacrebleu.metrics import BLEU, CHRF
    from additive_chain import iwslt_confirm as confirm
    from additive_chain import iwslt_followup as followup
    from additive_chain import iwslt_full_dev as full
    from additive_chain import iwslt_length_refit as refit
    from additive_chain.iwslt_controls import head_fingerprint
    from additive_chain.translation import LengthPredictor
    monkeypatch.setitem(confirm.COUNTS, "dev", 2)
    data = tmp_path / "data"
    data.mkdir()
    raw_rows = [
        dict(id=str(i), target="this is a correct translation of the original sentence",
             source_ids=[8 + i], target_ids=list(range(8 + i))) for i in range(2)]
    (data / "dev.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    (data / "counts.pt").write_bytes(b"synthetic counts")
    atomic_json(data / "manifest.json",
        dict(splits=dict(dev=dict(sha256=file_sha256(data / "dev.jsonl"))),
             counts_sha256=file_sha256(data / "counts.pt")))
    rows = confirm.ordered_rows(data, "dev")
    old_head, head = LengthPredictor(hidden_size=2), LengthPredictor(hidden_size=2)
    trained, head_path = tmp_path / "trained.pt", tmp_path / "refit.pt"
    torch.save(dict(length_head=old_head.state_dict()), trained)
    model_sha, data_sha = file_sha256(trained), file_sha256(data / "manifest.json")
    torch.save(dict(format=refit.FORMAT, hidden_size=2, length_head=head.state_dict(),
                    identity=dict(model_sha256=model_sha, data_sha256=data_sha,
                                  source_sha256=refit.source_hashes())), head_path)
    head_sha, old_head_sha = head_fingerprint(head.state_dict()), head_fingerprint(old_head.state_dict())
    selection_cells = [
        dict(c, bleu=10. + c["strength"] - c["temperature"])
        for steps in (8, 32) for c in confirm.dev_cells(steps) + followup.calibration_cells(steps)]
    selection_path = tmp_path / "selection.json"
    atomic_json(selection_path, dict(source_sha256=followup.source_hashes(),
        cells=selection_cells, selected=followup.select_settings(selection_cells),
        model_sha256=model_sha, data_sha256=data_sha))
    selection_sha = file_sha256(selection_path)
    root = tmp_path / "full-dev"
    for cell in full.full_cells():
        records = []
        for i, row in enumerate(rows):
            length = 4 + i
            records.append(dict(id=row["id"], reference=row["target"], hypothesis=row["target"],
                scheduled_steps=cell["steps"], predicted_length=length,
                reveal_counts=[length - 1, 1] + [0] * (cell["steps"] - 2),
                length_prediction_forwards=1, actual_forward_calls=2, seconds=1.,
                same_state_probe_seconds=0., opportunities=[
                    dict(masked=length, revealed=length - 1, component_reveals=[length - 1]),
                    dict(masked=1, revealed=1, component_reveals=[1])]))
        hyps, refs = [r["hypothesis"] for r in records], [[r["reference"] for r in records]]
        summary = dict(examples=len(rows),
            bleu=BLEU(lowercase=True, tokenize="13a").corpus_score(hyps, refs).score,
            chrf=CHRF().corpus_score(hyps, refs).score)
        folder = root / followup.output_name(cell)
        folder.mkdir(parents=True)
        path = folder / "result.json"
        atomic_json(path, dict(predictions=records, summary=summary))
        identity = full.build_identity(cell, rows, model_sha, data_sha, selection_sha,
                                       head_sha, file_sha256(head_path), old_head_sha)
        atomic_json(folder / "result-identity.json",
                    dict(identity=identity, output_sha256=file_sha256(path)))
        atomic_json(folder / "summary.json",
                    dict(completed=1, expected=1, cell=cell, **summary))
    return SimpleNamespace(data=data, selection=selection_path, trained=trained,
        length_head=head_path, input=root, output=tmp_path / "readout.json")


def test_full_grid_is_recomputed_without_resampling(completed_full_dev):
    from additive_chain.iwslt_full_dev_readout import run
    run(completed_full_dev)
    result = json.loads(completed_full_dev.output.read_text())
    assert result["split"] == "dev"
    assert result["seed"] == 101
    assert len(result["outputs"]) == 12
    assert set(result["steps"]) == {"2", "4", "8", "32"}
    for entry in result["steps"].values():
        assert set(entry["methods"]) == {"baseline", "joint", "own-marginal"}
        assert entry["contrasts"]["joint_minus_own-marginal"]["bleu"] == 0.
        assert "interval_95" not in entry


def change_output(path, mutate):
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))
    receipt_path = path.with_name("result-identity.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["output_sha256"] = file_sha256(path)
    atomic_json(receipt_path, receipt)


@pytest.mark.parametrize("damage", ["reference", "score", "nan", "length", "schedule",
                                  "steps", "id", "oracle", "length-forwards"])
def test_full_readout_rejects_semantic_corruption(completed_full_dev, damage):
    from additive_chain.iwslt_full_dev_readout import run
    path = next(completed_full_dev.input.glob("T2-*-lambda0.2-joint-*/result.json"))
    def mutate(payload):
        record = payload["predictions"][0]
        if damage == "reference":
            record["reference"] = "a different target"
        elif damage in ("score", "nan"):
            payload["summary"]["bleu"] = -1. if damage == "score" else float("nan")
        elif damage == "length":
            record["predicted_length"] += 1
        elif damage == "schedule":
            n = record["predicted_length"]
            record["reveal_counts"] = [1, n - 1]
            record["opportunities"] = [
                dict(masked=n, revealed=1, component_reveals=[1]),
                dict(masked=n - 1, revealed=n - 1, component_reveals=[n - 1])]
        elif damage == "steps":
            record["scheduled_steps"] = 8
        elif damage == "id":
            record["id"] = "test-example"
        elif damage == "oracle":
            record["length_mode"] = "reference-diagnostic"
        else:
            record["length_prediction_forwards"] = 2
    change_output(path, mutate)
    with pytest.raises(ValueError):
        run(completed_full_dev)
    assert not completed_full_dev.output.exists()


@pytest.mark.parametrize("field,value", [
    ("split", "test"), ("head_sha256", "different-head"), ("nucleus_p", 1.),
    ("top_k", 2048)])
def test_full_readout_rejects_different_protocol(completed_full_dev, field, value):
    from additive_chain.iwslt_full_dev_readout import run
    path = next(completed_full_dev.input.glob("*/result-identity.json"))
    receipt = json.loads(path.read_text())
    receipt["identity"][field] = value
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="identity"):
        run(completed_full_dev)


def test_full_readout_rejects_different_seed(completed_full_dev):
    from additive_chain.iwslt_full_dev_readout import run
    path = next(completed_full_dev.input.glob("*/result-identity.json"))
    receipt = json.loads(path.read_text())
    receipt["identity"]["cell"]["seed"] = 202
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="identity"):
        run(completed_full_dev)


def test_full_readout_requires_every_result(completed_full_dev):
    from additive_chain.iwslt_full_dev_readout import run
    next(completed_full_dev.input.glob("*/result.json")).unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        run(completed_full_dev)


def test_full_readout_rejects_changed_counts(completed_full_dev):
    from additive_chain.iwslt_full_dev_readout import run
    (completed_full_dev.data / "counts.pt").write_bytes(b"changed counts")
    with pytest.raises(ValueError, match="counts"):
        run(completed_full_dev)


def test_full_readout_rejects_lengths_changed_across_steps(completed_full_dev):
    from additive_chain.iwslt_full_dev_readout import run
    for path in completed_full_dev.input.glob("T2-*/result.json"):
        def mutate(payload):
            record = payload["predictions"][0]
            n = record["predicted_length"] + 1
            record.update(predicted_length=n, reveal_counts=[n - 1, 1],
                opportunities=[
                    dict(masked=n, revealed=n - 1, component_reveals=[n - 1]),
                    dict(masked=1, revealed=1, component_reveals=[1])])
        change_output(path, mutate)
    with pytest.raises(ValueError, match="across"):
        run(completed_full_dev)
