import gzip
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from additive_chain import iwslt_output_length_audit as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def text_record(ids, length=5):
    return dict(token_ids=ids, predicted_length=length, hypothesis="one two three four",
                reference="one two three four")


def test_pre_eos_contract_distinguishes_early_final_and_absent_eos():
    records = [text_record([1, 2]), text_record([1, 2, 3, 4]),
               text_record([1, 2, 3, 4, 5]), text_record([])]
    result = audit.token_audit(records, [dict(target_ids=[1, 2, 3, 4])] * 4)
    measured = result["measurements"]
    eos = measured["eos_inferred_from_verified_return_contract"]
    assert eos["examples_with_eos"] == 3
    assert {key: value["examples"] for key, value in eos["categories"].items()} == {
        "early_eos": 2, "final_slot_eos": 1, "no_eos": 1}
    assert eos["first_eos_zero_based_position"]["total"] == 6
    assert eos["discarded_slots_after_first_eos"]["total"] == 6
    assert measured["returned_lexical_tokens"]["total"] == 11
    assert measured["returned_minus_predicted_lexical_capacity"]["total"] == -5
    assert measured["empty_returned_sequences"] == 1
    assert measured["returned_special_token_counts"] == {"50256": 0, "50257": 0}


def test_missing_ids_are_unavailable_not_inferred_from_text():
    records = [dict(hypothesis="<|endoftext|>", reference="a", predicted_length=5)]
    result = audit.token_audit(records, [dict(target_ids=[1])])
    assert result["available_examples"] == 0
    assert result["unavailable_examples"] == 1
    assert result["measurements"] is None


@pytest.mark.parametrize("ids", [[50256], [-1], [True], [50258], list(range(6))])
def test_invalid_saved_token_contract_rejected(ids):
    with pytest.raises(ValueError):
        audit.token_audit([text_record(ids)], [dict(target_ids=[1])])


def test_returned_mask_is_counted_not_confused_with_eos():
    result = audit.token_audit([text_record([50257] * 5)], [dict(target_ids=[1])])
    value = result["measurements"]
    assert value["returned_special_token_counts"]["50257"] == 5
    assert value["eos_inferred_from_verified_return_contract"]["examples_with_eos"] == 0


def test_bleu_components_expose_brevity_penalty():
    records = [dict(hypothesis="one two three four", reference="one two three four five six")]
    value = audit.quality(records)
    assert value["bleu"] < 100
    assert value["bleu_without_brevity_penalty"] == pytest.approx(100)
    assert value["precisions"] == [100] * 4
    assert (value["system_length"], value["reference_length"]) == (4, 6)
    assert value["brevity_penalty"] == pytest.approx(__import__("math").exp(-.5))


@pytest.fixture
def experiment(tmp_path):
    data, output = tmp_path / "data", tmp_path / "outputs"
    data.mkdir()
    rows = [dict(id=f"dev-{i}", source_ids=[1], target_ids=[2, 3, 4, 5],
                 target="one two three four") for i in range(2)]
    (data / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write(data / "manifest.json", dict(counts_sha256="counts", splits={
        "dev": dict(rows=2, sha256=audit.sha(data / "dev.jsonl"))}))
    random.Random(31415).shuffle(rows)
    source = {name: audit.sha(Path(audit.__file__).with_name(name)) for name in audit.SHARED_SOURCES}
    assets = dict(model_sha256="model", data_sha256=audit.sha(data / "manifest.json"),
        counts_sha256="counts", selection_sha256="selection", head_sha256="length",
        head_artifact_sha256="length-file", nucleus_p=.9, top_k=500)
    report = dict(split="dev", examples=2, outputs={}, steps={}, source_sha256=source, **assets)
    for step in audit.STEPS:
        report["steps"][str(step)] = dict(methods={})
        for method in ("baseline", "joint", "own-marginal"):
            cell = dict(steps=step, method=method, sampling="joint" if method == "baseline" else method,
                temperature=.85, strength=0. if method == "baseline" else .2,
                seed=101, length_mode="predicted")
            records = [dict(id=r["id"], hypothesis=r["target"], reference=r["target"],
                token_ids=[2, 3, 4, 5], predicted_length=5,
                reveal_counts=[2, 3] + [0] * (step - 2), actual_forward_calls=2,
                length_prediction_forwards=1, scheduled_steps=step) for r in rows]
            metrics = audit.quality(records)
            payload = dict(predictions=records, summary=dict(examples=2, bleu=metrics["bleu"], chrf=metrics["chrf"]))
            identity = dict(split="dev", ids=[r["id"] for r in rows], cell=cell,
                            source_sha256=source, **assets)
            relative = f"T{step}-{method}/result.json"
            path = output / relative
            write(path, payload)
            write(path.with_name("result-identity.json"), dict(identity=identity, output_sha256=audit.sha(path)))
            report["outputs"][relative] = audit.sha(path)
            report["steps"][str(step)]["methods"][method] = metrics
    readout = tmp_path / "readout.json"
    write(readout, report)
    return data, readout, output


def test_full_cpu_audit_and_cli_without_torch(experiment, tmp_path):
    data, readout, output = experiment
    value = audit.audit(data, [("baseline-count", readout, output)])
    assert value["runner_sha256"] == audit.sha(audit.__file__)
    assert value["checks"]["exact_reveal_schedules"]
    assert len(value["groups"]["baseline-count"]["cells"]) == 12
    assert value["predicted_lengths"]["predicted_minus_reference_including_eos"]["total"] == 0
    destination = tmp_path / "audit.json"
    code = ("import sys,runpy; "
            "sys.modules['torch']=None; "
            f"sys.argv={['audit', '--data', str(data), '--group', 'base', str(readout), str(output), '--output', str(destination)]!r}; "
            f"runpy.run_path({audit.__file__!r},run_name='__main__')")
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert audit.read(destination)["checks"]["model_weights_loaded"] is False
    repeated = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert repeated.returncode != 0  # Never replace a previous audit.


def test_compact_archives_preserve_metrics_but_declare_missing_token_fields(experiment):
    data, readout, output = experiment
    manifest = dict(split="dev", files={})
    for relative, expected in audit.read(readout)["outputs"].items():
        path = output / relative
        value = audit.read(path)
        for row in value["predictions"]:
            del row["token_ids"]
        value.update(raw_sha256=expected, receipt=audit.read(path.with_name("result-identity.json")))
        compressed = Path(str(path) + ".gz")
        with gzip.open(compressed, "wt") as stream:
            json.dump(value, stream)
        manifest["files"][relative + ".gz"] = dict(sha256=audit.sha(compressed), raw_sha256=expected)
        path.unlink()
    write(output / "manifest.json", manifest)
    result = audit.audit(data, [("compact", readout, output)])
    assert all(cell["tokens"]["measurements"] is None for cell in result["groups"]["compact"]["cells"])


@pytest.mark.parametrize("damage", ["test", "ids", "reference", "schedule", "length", "source",
    "quality", "support", "checkpoint", "missing-cell", "duplicate-cell", "data"])
def test_protocol_damage_rejected(experiment, damage):
    data, readout, output = experiment
    report = audit.read(readout)
    relative = next(iter(report["outputs"]))
    path = output / relative
    payload, receipt = audit.read(path), audit.read(path.with_name("result-identity.json"))
    identity = receipt["identity"]
    if damage == "test":
        identity["split"] = "test"
    elif damage == "ids":
        payload["predictions"].reverse()
    elif damage == "reference":
        payload["predictions"][0]["reference"] = "different"
    elif damage == "schedule":
        payload["predictions"][0]["reveal_counts"] = [1, 4]
    elif damage == "length":
        payload["predictions"][0].update(predicted_length=6, reveal_counts=[2, 4])
    elif damage == "source":
        identity["source_sha256"]["translation.py"] = "different"
    elif damage == "quality":
        payload["summary"]["bleu"] -= 1
    elif damage == "support":
        identity["nucleus_p"] = 1.
    elif damage == "checkpoint":
        identity["candidate"] = dict(name="another")
    elif damage == "missing-cell":
        del report["outputs"][relative]
    elif damage == "duplicate-cell":
        second = output / "duplicate/result.json"
        write(second, payload)
        write(second.with_name("result-identity.json"), receipt)
        report["outputs"]["duplicate/result.json"] = audit.sha(second)
    else:
        (data / "dev.jsonl").write_text("{}\n")
    write(path, payload)
    receipt["output_sha256"] = audit.sha(path)
    write(path.with_name("result-identity.json"), receipt)
    if damage != "missing-cell":
        report["outputs"][relative] = audit.sha(path)
    write(readout, report)
    with pytest.raises(ValueError):
        audit.audit(data, [("damaged", readout, output)])


def test_changed_raw_hash_and_path_escape_rejected(experiment):
    _, readout, output = experiment
    relative, expected = next(iter(audit.read(readout)["outputs"].items()))
    with pytest.raises(ValueError, match="Changed prediction"):
        audit.load_output(output, relative, "wrong")
    with pytest.raises(ValueError, match="inside"):
        audit.load_output(output, "../test/result.json", expected)
