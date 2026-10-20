import copy
import json
from types import SimpleNamespace

import pytest
import torch

from additive_chain.backbone import file_sha256
from additive_chain.iwslt_confirm import atomic_json
from additive_chain.iwslt_controls_readout import (
    differences, length_errors, require_same_schedule, summarize)


def test_summary_preserves_fixed_seed_scores_and_differences():
    cells = [dict(method=m, seed=s, bleu=i + s / 100., chrf=2 * i + s / 100.)
             for i, m in enumerate(("baseline", "joint", "own-marginal"))
             for s in (101, 202, 303)]
    means = summarize(cells)
    assert means["baseline"]["bleu"]["mean"] == 2.02
    diff = differences(means["joint"], means["baseline"])
    assert diff["bleu"]["mean"] == pytest.approx(1.)
    assert [x["seed"] for x in diff["bleu"]["by_seed"]] == [101, 202, 303]
    bad = copy.deepcopy(means["baseline"])
    bad["bleu"]["by_seed"][0]["seed"] = 42
    with pytest.raises(ValueError, match="seeds"):
        differences(means["joint"], bad)


def test_length_metrics_measure_allocated_lengths_including_eos():
    result = length_errors([dict(predicted_length=n) for n in (4, 5, 9)],
                           [dict(target_ids=[0] * n) for n in (3, 5, 6)])
    assert result["mean_absolute_error"] == 1.
    assert result["exact_fraction"] == 1 / 3
    assert result["within_one_fraction"] == 2 / 3
    assert result["under_fraction"] == result["over_fraction"] == 1 / 3


def test_schedule_comparison_checks_lengths_and_reveals():
    a = [dict(predicted_length=4, reveal_counts=[3, 1])]
    with pytest.raises(ValueError, match="length"):
        require_same_schedule(a, [dict(predicted_length=5, reveal_counts=[4, 1])])
    with pytest.raises(ValueError, match="schedule"):
        require_same_schedule(a, [dict(predicted_length=4, reveal_counts=[1, 3])])


def make_record(row, length, steps):
    counts = [length - 1, 1] + [0] * (steps - 2)
    opportunities = [dict(masked=length, revealed=length - 1, component_reveals=[length - 1]),
                     dict(masked=1, revealed=1, component_reveals=[1])]
    return dict(id=row["id"], hypothesis=row["target"], reference=row["target"],
        predicted_length=length, reveal_counts=counts, opportunities=opportunities,
        scheduled_steps=steps, actual_forward_calls=2, length_prediction_forwards=1,
        seconds=1., same_state_probe_seconds=0.)


@pytest.fixture
def completed_controls(tmp_path, monkeypatch):
    from sacrebleu.metrics import BLEU, CHRF
    from additive_chain import iwslt_confirm as confirm
    from additive_chain import iwslt_followup as followup
    from additive_chain import iwslt_controls as controls
    from additive_chain import iwslt_controls_readout as readout
    from additive_chain import iwslt_length_refit as refit
    from additive_chain.translation import LengthPredictor

    monkeypatch.setitem(confirm.COUNTS, "dev", 2)
    monkeypatch.setattr(readout, "stratified_rows", lambda rows, n: rows)
    data = tmp_path / "data"
    data.mkdir()
    raw_rows = [dict(id=str(i), target="this is a correct translation of the original sentence",
                    target_ids=list(range(8 + i)), source_ids=[10 + i]) for i in range(2)]
    (data / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw_rows))
    atomic_json(data / "manifest.json",
                dict(splits=dict(dev=dict(sha256=file_sha256(data / "dev.jsonl")))))
    rows = confirm.ordered_rows(data, "dev")
    old_head = LengthPredictor(hidden_size=2)
    trained = tmp_path / "trained.pt"
    torch.save(dict(length_head=old_head.state_dict()), trained)
    cells = [dict(c, bleu=10. + c["strength"]) for t in (8, 32)
             for c in confirm.dev_cells(t) + followup.calibration_cells(t)]
    selection = dict(source_sha256=followup.source_hashes(), cells=cells,
        selected=followup.select_settings(cells), data_sha256=file_sha256(data / "manifest.json"),
        model_sha256=file_sha256(trained), development_artifacts={})
    selection_path = tmp_path / "selection.json"
    atomic_json(selection_path, selection)
    selection_sha = file_sha256(selection_path)
    old_sha = controls.head_fingerprint(old_head.state_dict())
    new_head = LengthPredictor(hidden_size=2)
    head_path = tmp_path / "refit.pt"
    torch.save(dict(format=refit.FORMAT, hidden_size=2, length_head=new_head.state_dict(),
        identity=dict(model_sha256=selection["model_sha256"], data_sha256=selection["data_sha256"],
                      source_sha256=refit.source_hashes())), head_path)
    new_sha = controls.head_fingerprint(new_head.state_dict())
    previous, current = tmp_path / "previous", tmp_path / "controls"
    for experiment in ("previous", "p1", "new-length"):
        for steps in (2, 4, 8, 32):
            if experiment == "p1" and steps not in (2, 4):
                continue
            stage = "low" if steps in (2, 4) else "length"
            cells = (followup.followup_cells(stage, steps, selection)
                     if experiment == "previous" else
                     controls.control_cells("new-length", steps, selection))
            folder = ((previous / stage) if experiment == "previous"
                      else (current / experiment)) / f"step-{steps}"
            folder.mkdir(parents=True)
            for cell in cells:
                records = [make_record(row, 4 + i + int(experiment == "new-length"), steps)
                           for i, row in enumerate(rows)]
                if cell["length_mode"] == "reference-diagnostic":
                    for record in records:
                        record["length_mode"] = "reference-diagnostic"
                refs, hyps = [[r["reference"] for r in records]], [r["hypothesis"] for r in records]
                payload = dict(predictions=records, summary=dict(examples=2,
                    bleu=BLEU(lowercase=True, tokenize="13a").corpus_score(hyps, refs).score,
                    chrf=CHRF().corpus_score(hyps, refs).score))
                path = folder / (followup.output_name(cell) + ".json")
                atomic_json(path, payload)
                identity = dict(cell=cell, split="dev", ids=[r["id"] for r in rows],
                    model_sha256=selection["model_sha256"], data_sha256=selection["data_sha256"],
                    selection_sha256=selection_sha)
                if experiment == "previous":
                    identity.update(stage=stage, source_sha256=selection["source_sha256"])
                else:
                    identity.update(experiment=experiment, source_sha256=controls.control_source_hashes(experiment),
                        nucleus_p=1. if experiment == "p1" else .9, top_k=500,
                        head_sha256=old_sha if experiment == "p1" else new_sha,
                        old_head_sha256=old_sha,
                        head_artifact_sha256=None if experiment == "p1" else file_sha256(head_path))
                atomic_json(path.with_name(path.stem + "-identity.json"),
                            dict(identity=identity, output_sha256=file_sha256(path)))
            atomic_json(folder / "summary.json", dict(completed=len(cells),
                        expected=len(cells), cells=cells))
    return SimpleNamespace(data=data, selection=selection_path, trained=trained, previous=previous,
        controls=current, output=tmp_path / "result.json", experiment="p1", length_head=None,
        new_head=head_path)


def test_complete_p1_readout(completed_controls):
    from additive_chain.iwslt_controls_readout import run
    run(completed_controls)
    result = json.loads(completed_controls.output.read_text())
    assert set(result["steps"]) == {"2", "4"}
    assert len(result["outputs"]) == 36
    assert result["steps"]["2"]["changed_length_fraction"] == 0.
    assert result["steps"]["2"]["control"]["contrasts"]["joint_minus_own-marginal"]["bleu"]["mean"] == 0.


def test_refit_readout_allows_changed_lengths_but_same_population(completed_controls):
    from additive_chain.iwslt_controls_readout import run
    completed_controls.experiment = "new-length"
    completed_controls.length_head = completed_controls.new_head
    run(completed_controls)
    result = json.loads(completed_controls.output.read_text())
    assert set(result["steps"]) == {"2", "4", "8", "32"}
    assert len(result["outputs"]) == 48
    assert result["steps"]["2"]["changed_length_fraction"] == 1.


def tamper(path, mutate):
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))
    receipt_path = path.with_name(path.stem + "-identity.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["output_sha256"] = file_sha256(path)
    atomic_json(receipt_path, receipt)


@pytest.mark.parametrize("damage", ["reference", "schedule", "score", "nan", "oracle", "steps", "id"])
def test_readout_rejects_corruption_with_updated_hash(completed_controls, damage):
    from additive_chain.iwslt_controls_readout import run
    folder = completed_controls.controls / "p1" / "step-2"
    path = next(p for p in folder.glob("*-joint-seed101-predicted.json")
                if "-lambda0.0-" not in p.name)
    def mutate(payload):
        r = payload["predictions"][0]
        if damage == "reference":
            r["reference"] = "a different text"
        elif damage == "schedule":
            r["reveal_counts"] = [1, r["predicted_length"] - 1]
        elif damage in ("score", "nan"):
            payload["summary"]["bleu"] = -1. if damage == "score" else float("nan")
        elif damage == "oracle":
            r["length_mode"] = "reference-diagnostic"
        elif damage == "steps":
            r["scheduled_steps"] = 32
        else:
            r["id"] = "test-example"
    tamper(path, mutate)
    with pytest.raises(ValueError):
        run(completed_controls)
    assert not completed_controls.output.exists()


def test_receipt_cannot_switch_to_test_split(completed_controls):
    from additive_chain.iwslt_controls_readout import run
    folder = completed_controls.controls / "p1" / "step-2"
    path = next(folder.glob("*-identity.json"))
    payload = json.loads(path.read_text())
    payload["identity"]["split"] = "test"
    atomic_json(path, payload)
    with pytest.raises(ValueError, match="identity"):
        run(completed_controls)


def test_incomplete_grid_rejected(completed_controls):
    from additive_chain.iwslt_controls_readout import run
    folder = completed_controls.controls / "p1" / "step-2"
    next(folder.glob("*-predicted.json")).unlink()
    with pytest.raises(ValueError, match="grid"):
        run(completed_controls)


def test_head_hash_must_match_original_model(completed_controls):
    from additive_chain.iwslt_controls_readout import run
    folder = completed_controls.controls / "p1" / "step-2"
    path = next(folder.glob("*-identity.json"))
    payload = json.loads(path.read_text())
    payload["identity"]["old_head_sha256"] = "wrong-head"
    atomic_json(path, payload)
    with pytest.raises(ValueError, match="identity"):
        run(completed_controls)
