import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from additive_chain.iwslt_followup_readout import diagnostics, span_summary


def span_record(i, supported, j, m):
    return dict(id=str(i), start=0, gold_tokens=[0, 1],
        gold_full_vocab_ranks=[2, 700],
        gold_full_vocab_log_probabilities=[-1., -8.], supported=supported,
        joint_gold_probability=j, independent_gold_probability=m)


def test_wider_candidates_use_same_examples_and_keep_unsupported_zeros():
    default = [span_record(0, False, 0., 0.), span_record(1, True, .5, .4)]
    wide = [span_record(0, True, .3, .2), span_record(1, True, .5, .4)]
    result = span_summary(default, wide, 100)
    assert result["default"]["supported"] == 1
    assert result["wide"]["supported"] == 2
    assert result["default"]["mean_joint_gold_probability"] == .25
    assert result["wide"]["joint_minus_independent"] == pytest.approx(.1)
    assert result["default"]["token_coverage_top500"] == .5
    assert result["wide"]["token_coverage_top2048"] == 1.
    with pytest.raises(ValueError, match="populations"):
        span_summary(default, wide[::-1], 10)


def test_span_model_outputs_must_match():
    a = [span_record(0, False, 0., 0.)]
    b = [dict(a[0], gold_full_vocab_ranks=[1, 1])]
    with pytest.raises(ValueError, match="outputs"):
        span_summary(a, b, 10)
    b = [dict(a[0], gold_full_vocab_log_probabilities=[-2., -9.])]
    with pytest.raises(ValueError, match="outputs"):
        span_summary(a, b, 10)


@pytest.mark.parametrize("j,m", [
    (float("nan"), .2), (.2, float("inf")), (-.1, .2), (.2, 1.1)])
def test_span_probabilities_must_be_finite_and_in_range(j, m):
    rows = [span_record(0, True, j, m)]
    with pytest.raises(ValueError):
        span_summary(rows, rows, 10)


def test_unsupported_spans_have_zero_probability():
    rows = [span_record(0, False, .1, 0.)]
    with pytest.raises(ValueError):
        span_summary(rows, rows, 10)


@pytest.mark.parametrize("replicates", [0, -1])
def test_span_readout_requires_positive_replicates(replicates):
    rows = [span_record(0, True, .2, .1)]
    with pytest.raises(ValueError):
        span_summary(rows, rows, replicates)


def test_opportunities_measure_components_not_only_adjacency():
    rows = [dict(predicted_length=4, actual_forward_calls=2, seconds=1.,
        length_prediction_forwards=1, same_state_probe_seconds=.2, reveal_counts=[3, 1], opportunities=[
            dict(masked=4, revealed=3, component_reveals=[2, 1]),
            dict(masked=1, revealed=1, component_reveals=[1])])]
    result = diagnostics(rows)
    assert result["singleton_update_fraction"] == .5
    assert result["tokens_revealed_with_another_in_same_component_fraction"] == .5
    assert result["same_state_pair_count"] == 0
    assert result["same_state_pair_mean_mi"] is None
    assert result["mean_denoising_model_calls"] == 2
    assert result["mean_total_model_calls"] == 3


def diagnostic_record(length=4, counts=None):
    counts = counts or [length-1, 1]
    masked = length
    opportunities = []
    for count in counts:
        if count:
            opportunities.append(dict(masked=masked, revealed=count,
                                      component_reveals=[count]))
            masked -= count
    return dict(predicted_length=length, actual_forward_calls=len(opportunities),
        length_prediction_forwards=1, seconds=1., same_state_probe_seconds=0.,
        reveal_counts=counts, opportunities=opportunities)


@pytest.mark.parametrize("damage", [
    lambda r: r.update(reveal_counts=[2, 2]),
    lambda r: r.update(actual_forward_calls=1),
    lambda r: r.update(predicted_length=5),
    lambda r: r["opportunities"][1].update(masked=2),
    lambda r: r["opportunities"][0].update(component_reveals=[2]),
])
def test_diagnostics_require_consistent_observed_schedule(damage):
    row = diagnostic_record()
    damage(row)
    with pytest.raises(ValueError):
        diagnostics([row])


def test_zero_reveal_steps_are_not_counted_as_model_calls():
    row = diagnostic_record(counts=[0, 3, 0, 1])
    result = diagnostics([row])
    assert result["nonempty_updates"] == 2


@pytest.fixture
def completed_followup(tmp_path, monkeypatch):
    """Exercise the actual readout on small files, without replacing its checks."""
    from sacrebleu.metrics import BLEU, CHRF
    from sacremoses import MosesDetokenizer
    from additive_chain import iwslt_confirm as confirm
    from additive_chain import iwslt_followup as followup
    from additive_chain import iwslt_followup_readout as readout
    from additive_chain.backbone import file_sha256
    from additive_chain.iwslt_spans import span_population

    monkeypatch.setitem(confirm.COUNTS, "dev", 2)
    monkeypatch.setattr(readout, "stratified_rows", lambda rows, n: rows)
    data = tmp_path / "data"
    data.mkdir()
    raw_rows = [
        dict(id=str(i), target="this is a correct translation of the original sentence",
             target_ids=list(range(8+i)), source_ids=[10+i])
        for i in range(2)]
    (data / "dev.jsonl").write_text("".join(json.dumps(r)+"\n" for r in raw_rows))
    confirm.atomic_json(data / "manifest.json",
        dict(splits=dict(dev=dict(sha256=file_sha256(data / "dev.jsonl")))))
    rows = confirm.ordered_rows(data, "dev")
    cells = [dict(c, bleu=10.+c["strength"]) for t in (8, 32)
             for c in confirm.dev_cells(t)+followup.calibration_cells(t)]
    selection = dict(source_sha256=followup.source_hashes(), cells=cells,
        selected=followup.select_settings(cells),
        data_sha256=file_sha256(data / "manifest.json"),
        model_sha256="synthetic-model", development_artifacts={})
    selection_path = tmp_path / "selection.json"
    confirm.atomic_json(selection_path, selection)
    selection_sha = file_sha256(selection_path)
    root = tmp_path / "followup"
    root.mkdir()
    detok = MosesDetokenizer(lang="en")
    for stage, steps_list in (("low", (2, 4)), ("length", (8, 32))):
        for steps in steps_list:
            folder = root / stage / f"step-{steps}"
            folder.mkdir(parents=True)
            stage_cells = followup.followup_cells(stage, steps, selection)
            for cell in stage_cells:
                records = []
                for i, row in enumerate(rows):
                    reference = detok.detokenize(row["target"].split())
                    length = len(row["target_ids"])+1 if cell["length_mode"] == "reference-diagnostic" else 4+i
                    record = dict(id=row["id"], reference=reference, hypothesis=reference,
                        **diagnostic_record(length, [length-1, 1]+[0]*(steps-2)))
                    if cell["length_mode"] == "reference-diagnostic":
                        record["length_mode"] = "reference-diagnostic"
                    records.append(record)
                refs = [[r["reference"] for r in records]]
                hyps = [r["hypothesis"] for r in records]
                payload = dict(predictions=records, summary=dict(examples=2,
                    bleu=BLEU(lowercase=True, tokenize="13a").corpus_score(hyps, refs).score,
                    chrf=CHRF().corpus_score(hyps, refs).score))
                path = folder / (followup.output_name(cell)+".json")
                confirm.atomic_json(path, payload)
                identity = dict(cell=cell, stage=stage, split="dev",
                    ids=[r["id"] for r in rows], model_sha256=selection["model_sha256"],
                    data_sha256=selection["data_sha256"], source_sha256=selection["source_sha256"],
                    selection_sha256=selection_sha)
                confirm.atomic_json(path.with_name(path.stem+"-identity.json"),
                    dict(identity=identity, output_sha256=file_sha256(path)))
            confirm.atomic_json(folder / "summary.json",
                dict(completed=len(stage_cells), expected=len(stage_cells)))
    span_hashes = {name: file_sha256(Path(readout.__file__).parent / name) for name in
        ("iwslt_spans.py", "translation.py", "generation.py", "chain.py", "counts.py", "candidates.py")}
    for label, k, p in (("default", 500, .9), ("wide", 2048, 1.)):
        folder = root / ("spans-"+label)
        folder.mkdir()
        for length in (2, 4, 8):
            selected, eligible = span_population(raw_rows, length, 1000, 101)
            expected = [dict(id=row["id"], start=start) for row, start in selected]
            setting = selection["selected"]["8"]
            identity = dict(model_sha256=selection["model_sha256"],
                data_sha256=selection["data_sha256"], length=length,
                eligible=eligible, selected=expected, strength=setting["strength"],
                temperature=setting["crf_temperature"], seed=101, top_k=k, top_p=p,
                source_sha256=span_hashes,
                protocol="Same masked target span, visible gold context; unchanged model-derived candidates; unsupported gold has zero probability")
            records = [dict(span_record(row["id"], True, .3, .2), start=start)
                       for row, start in selected]
            confirm.atomic_json(folder / f"span-{length}.json",
                                dict(identity=identity, records=records))
    return SimpleNamespace(input=root, data=data, selection=selection_path,
        output=tmp_path / "result.json", replicates=2)


def test_complete_readout_checks_all_three_stages(completed_followup):
    from additive_chain.iwslt_followup_readout import run
    run(completed_followup)
    result = json.loads(completed_followup.output.read_text())
    assert set(result["low"]) == {"2", "4"}
    assert set(result["length"]) == {"8", "32"}
    assert set(result["spans"]) == {"2", "4", "8"}
    assert len(result["outputs"]) == 36
    assert result["low"]["2"]["contrasts"]["predicted"]["joint_minus_baseline"]["bleu"]["delta"] == 0.
    assert set(result["length"]["8"]["reference_minus_predicted"]) == {"baseline", "joint", "own-marginal"}


@pytest.mark.parametrize("damage", ["reference", "schedule", "score", "nan-score"])
def test_readout_rejects_semantic_tampering_even_with_updated_receipt(completed_followup, damage):
    from additive_chain.backbone import file_sha256
    from additive_chain.iwslt_followup_readout import run
    folder = completed_followup.input / "low" / "step-2"
    path = next(p for p in sorted(folder.glob("*-joint-seed101-predicted.json"))
                if "-lambda0.0-" not in p.name)
    payload = json.loads(path.read_text())
    row = payload["predictions"][0]
    if damage == "reference":
        row["reference"] = "a different reference"
    elif damage == "schedule":
        row.update(diagnostic_record(row["predicted_length"], [1, row["predicted_length"]-1]))
    else:
        payload["summary"]["bleu"] = float("nan") if damage == "nan-score" else -1.
    path.write_text(json.dumps(payload))
    receipt_path = path.with_name(path.stem+"-identity.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["output_sha256"] = file_sha256(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        run(completed_followup)
    assert not completed_followup.output.exists()
