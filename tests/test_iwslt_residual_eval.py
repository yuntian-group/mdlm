import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from additive_chain import iwslt_residual_eval as evaluation
from additive_chain.backbone import file_sha256
from additive_chain.iwslt_confirm import atomic_json, read
from additive_chain.iwslt_followup import output_name


def candidate(epoch):
    return dict(name=f"epoch-{epoch:04d}", head=f"/remote/epoch-{epoch:04d}.pt",
                epoch=epoch, update=epoch * 100, sha256=str(epoch % 10) * 64)


def scores(value):
    return [dict(cell=c, bleu=value, chrf=value) for c in evaluation.cells("selection")]


def test_fixed_residual_grid():
    assert len(evaluation.cells("selection")) == 4
    full = evaluation.cells("full")
    assert len(full) == 8
    assert {c["steps"] for c in full} == {2, 4, 8, 32}
    assert {c["seed"] for c in full} == {101}
    assert {c["temperature"] for c in full} == {.85}
    assert {c["strength"] for c in full} == {.2}
    assert {c["method"] for c in full} == {"joint", "own-marginal"}
    with pytest.raises(ValueError):
        evaluation.cells("test")


def test_selection_uses_both_samplers_and_both_steps():
    candidates = [candidate(0), candidate(1)]
    old = scores(10.)
    new = scores(12.)
    new[0]["bleu"] = 8.
    winner, ranking = evaluation.choose_checkpoint(candidates,
        {candidates[0]["name"]: old, candidates[1]["name"]: new})
    assert winner == candidate(1)
    assert ranking[0]["mean_bleu"] == 11.
    # A dramatic advantage for joint alone must not drive checkpoint selection.
    new[0]["bleu"], new[1]["bleu"], new[2]["bleu"], new[3]["bleu"] = 30., 0., 0., 0.
    winner, _ = evaluation.choose_checkpoint(candidates,
        {candidates[0]["name"]: old, candidates[1]["name"]: new})
    assert winner == candidate(0)


def test_checkpoint_ties_choose_zero_epoch():
    winner, _ = evaluation.choose_checkpoint([candidate(1), candidate(0)],
        {"epoch-0000": scores(10.), "epoch-0001": scores(10.)})
    assert winner == candidate(0)


def test_best_loss_candidate_is_bound_to_update(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"head")
    saved = dict(checkpoint_id="best-loss-0000075", epoch=0, update=75)
    best = evaluation.candidate_entry(path, saved)
    assert best == dict(name="best-loss-0000075", head=str(path.resolve()),
                        epoch=0, update=75, sha256=file_sha256(path))
    manifest = dict(kind="global", candidates=[best, candidate(0)])
    assert evaluation.validate_candidates(manifest) == [candidate(0), best]
    saved["update"] = 76
    with pytest.raises(ValueError, match="epoch/update"):
        evaluation.candidate_entry(path, saved)


@pytest.mark.parametrize("name,epoch,update", [
    ("best-loss-0000000", 0, 0),
    ("best-loss-0000010", 0, 11),
    ("best-loss-10", 0, 10),
    ("../best-loss-0000010", 0, 10),
    ("epoch-0000", 0, 10),
    ("epoch-0001", 2, 10),
    ("epoch-0001", 1, -1),
    ("best-loss-0000010", True, 10),
    ("best-loss-0000010", 0, 10.),
])
def test_candidate_names_cannot_escape_or_disagree_with_checkpoint(name, epoch, update):
    changed = dict(candidate(0), name=name, epoch=epoch, update=update)
    with pytest.raises(ValueError, match="candidate"):
        evaluation.validate_candidates(dict(kind="global", candidates=[changed, candidate(1)]))


def test_best_loss_from_epoch_zero_does_not_replace_zero_residual_control():
    changed = dict(candidate(0), name="best-loss-0000001", update=1)
    with pytest.raises(ValueError, match="zero residual"):
        evaluation.validate_candidates(dict(kind="global", candidates=[changed]))


def test_tie_breaks_by_epoch_then_update_then_name():
    zero = candidate(0)
    epoch = candidate(1)
    best_late = dict(candidate(1), name="best-loss-0000100")
    best_early = dict(candidate(1), name="best-loss-0000075", update=75)
    candidates = [epoch, best_late, best_early, zero]
    evaluated = {c["name"]: scores(11. if c != zero else 10.) for c in candidates}
    winner, _ = evaluation.choose_checkpoint(candidates, evaluated)
    assert winner == best_early
    evaluated[best_early["name"]] = scores(9.)
    winner, _ = evaluation.choose_checkpoint(candidates, evaluated)
    assert winner == best_late  # Same epoch/update; name is the last deterministic tie break.
    next_epoch = dict(candidate(2), update=50)
    candidates.append(next_epoch)
    evaluated[next_epoch["name"]] = scores(11.)
    winner, _ = evaluation.choose_checkpoint(candidates, evaluated)
    assert winner == best_late  # Epoch precedes update in the key.


@pytest.mark.parametrize("damage", ["missing", "duplicate", "seed", "nan", "method"])
def test_selection_rejects_incomplete_or_changed_settings(damage):
    values = scores(10.)
    if damage == "missing":
        values.pop()
    elif damage == "duplicate":
        values[-1] = copy.deepcopy(values[0])
    elif damage == "seed":
        values[0]["cell"]["seed"] = 202
    elif damage == "nan":
        values[0]["bleu"] = float("nan")
    else:
        values[0]["cell"]["method"] = "baseline"
    with pytest.raises(ValueError):
        evaluation.choose_checkpoint([candidate(0)], {"epoch-0000": values})


@pytest.mark.parametrize("damage", ["zero", "duplicate", "relative", "name", "sha", "kind"])
def test_manifest_is_explicit_and_requires_zero_epoch(damage):
    manifest = dict(kind="global", candidates=[candidate(0), candidate(1)])
    if damage == "zero":
        manifest["candidates"].pop(0)
    elif damage == "duplicate":
        manifest["candidates"].append(candidate(0))
    elif damage == "relative":
        manifest["candidates"][0]["head"] = "head.pt"
    elif damage == "name":
        manifest["candidates"][0]["name"] = "../outside"
    elif damage == "sha":
        manifest["candidates"][0]["sha256"] = "x" * 64
    else:
        manifest["kind"] = "baseline"
    with pytest.raises(ValueError):
        evaluation.validate_candidates(manifest)


def test_population_reads_development_only(monkeypatch):
    rows = [dict(id=i) for i in range(7283)]
    calls = []
    monkeypatch.setattr(evaluation, "ordered_rows",
                        lambda data, split: calls.append(split) or rows)
    monkeypatch.setattr(evaluation, "stratified_rows",
                        lambda data, limit: data[:limit])
    assert len(evaluation.development_rows(Path("data"), "full")) == 7283
    assert len(evaluation.development_rows(Path("data"), "selection")) == 1024
    assert calls == ["dev", "dev"]
    with pytest.raises(ValueError):
        evaluation.development_rows(Path("data"), "test")


@pytest.fixture
def completed(tmp_path, monkeypatch):
    from sacrebleu.metrics import BLEU, CHRF
    monkeypatch.setattr(evaluation, "source_hashes", lambda: {"eval": "frozen"})
    rows = [
        dict(id=str(i), source_ids=[1, 2], target_ids=[3, 4, 5],
             target="this is a correct translation of the source") for i in range(2)]
    monkeypatch.setattr(evaluation, "development_rows", lambda data, population: rows)
    common = dict(model_sha256="model", data_sha256="data", counts_sha256="counts",
                  inherited_global_sha256=None, length_head_sha256="refitted")
    monkeypatch.setattr(evaluation, "common_assets", lambda args: common)
    candidates = []
    saved = {}
    for epoch in (0, 1):
        head = tmp_path / f"epoch-{epoch:04d}.pt"
        head.write_bytes(b"head-" + str(epoch).encode())
        saved[str(head)] = dict(checkpoint_id=f"epoch-{epoch:04d}", epoch=epoch,
                               update=epoch * 100, kind="global")
        candidates.append(evaluation.candidate_entry(head, saved[str(head)]))
    monkeypatch.setattr(evaluation, "load_heads",
                        lambda path, *args: (None, None, saved[str(path)]))
    manifest = tmp_path / "candidates.json"
    atomic_json(manifest, dict(kind="global", candidates=candidates))
    args = SimpleNamespace(data=tmp_path, trained=tmp_path / "trained",
        length_head=tmp_path / "length", selection=tmp_path / "old-selection",
        global_head=None, kind="global", input=tmp_path / "results",
        output=tmp_path / "selected.json", candidates=manifest)

    def write_grid(entry, population, residual_selection_sha=None):
        for cell in evaluation.cells(population):
            records = []
            for row in rows:
                records.append(dict(id=row["id"], hypothesis=row["target"], reference=row["target"],
                    predicted_length=4, reveal_counts=[3, 1] + [0] * (cell["steps"] - 2),
                    scheduled_steps=cell["steps"], length_prediction_forwards=1,
                    actual_forward_calls=2, seconds=1., same_state_probe_seconds=0.,
                    opportunities=[dict(masked=4, revealed=3, component_reveals=[3]),
                                   dict(masked=1, revealed=1, component_reveals=[1])]))
            hyps, refs = [r["hypothesis"] for r in records], [[r["reference"] for r in records]]
            summary = dict(examples=len(rows),
                bleu=BLEU(lowercase=True, tokenize="13a").corpus_score(hyps, refs).score,
                chrf=CHRF().corpus_score(hyps, refs).score)
            path = args.input / entry["name"] / population / output_name(cell) / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, dict(summary=summary, predictions=records))
            identity = evaluation.build_identity(
                cell, rows, population, "global", entry, common, residual_selection_sha)
            atomic_json(path.with_name("result-identity.json"),
                        dict(identity=identity, output_sha256=file_sha256(path)))
            atomic_json(path.with_name("summary.json"),
                        dict(cell=cell, completed=1, expected=1, **summary))
    for entry in candidates:
        write_grid(entry, "selection")
    return SimpleNamespace(args=args, common=common, rows=rows,
                           candidates=candidates, write_grid=write_grid)


def test_selection_and_full_readout_recompute_scores(completed):
    args = completed.args
    evaluation.run_select(args)
    selected = read(args.output)
    assert selected["selected"] == completed.candidates[0]
    assert len(selected["outputs"]) == 8
    assert selected["ranking"][0]["mean_bleu"] == pytest.approx(100.)
    completed.write_grid(selected["selected"], "full", file_sha256(args.output))
    args.residual_selection = args.output
    args.output = args.output.with_name("readout.json")
    evaluation.run_readout(args)
    result = read(args.output)
    assert result["split"] == "dev"
    assert set(result["steps"]) == {"2", "4", "8", "32"}
    assert len(result["outputs"]) == 8
    for row in result["steps"].values():
        assert row["joint_minus_own_marginal"] == dict(bleu=0., chrf=0.)
        assert "confidence_interval" not in row


def mutate_output(path, mutate):
    payload = read(path)
    mutate(payload)
    atomic_json(path, payload)
    receipt_path = path.with_name("result-identity.json")
    receipt = read(receipt_path)
    receipt["output_sha256"] = file_sha256(path)
    atomic_json(receipt_path, receipt)


@pytest.mark.parametrize("damage", ["reference", "score", "schedule", "id", "oracle",
                                  "steps", "length-forwards", "length"])
def test_select_rejects_corrupted_semantics(completed, damage):
    path = next(completed.args.input.glob("epoch-0001/selection/T2-*-joint-*/result.json"))
    def mutate(payload):
        row = payload["predictions"][0]
        if damage == "reference":
            row["reference"] = "different"
        elif damage == "score":
            payload["summary"]["bleu"] = 1.
        elif damage == "schedule":
            row["reveal_counts"] = [1, 3]
            row["opportunities"] = [dict(masked=4, revealed=1, component_reveals=[1]),
                                    dict(masked=3, revealed=3, component_reveals=[3])]
        elif damage == "id":
            row["id"] = "not-a-dev-example"
        elif damage == "oracle":
            row["length_mode"] = "reference-diagnostic"
        elif damage == "steps":
            row["scheduled_steps"] = 32
        elif damage == "length-forwards":
            row["length_prediction_forwards"] = 2
        else:
            row["predicted_length"] = 5
    mutate_output(path, mutate)
    with pytest.raises(ValueError):
        evaluation.run_select(completed.args)


def test_select_requires_all_four_outputs_for_every_candidate(completed):
    folder = next(completed.args.input.glob("epoch-0001/selection/T2-*-joint-*"))
    folder.rename(folder.with_name("not-a-registered-cell"))
    with pytest.raises(ValueError, match="grid"):
        evaluation.run_select(completed.args)


def test_selection_cannot_be_rewritten_to_prefer_joint(completed):
    evaluation.run_select(completed.args)
    value = read(completed.args.output)
    value["selected"] = completed.candidates[1]
    with pytest.raises(ValueError, match="shared"):
        evaluation.validate_selected(value, completed.common, "global")


def test_resume_completed_cell_does_not_load_translator(completed, monkeypatch):
    from additive_chain import translation
    args = completed.args
    entry = completed.candidates[0]
    args.head, args.candidate_name = Path(entry["head"]), entry["name"]
    args.population, args.cell_index, args.residual_selection = "selection", 0, None
    args.output = args.input
    monkeypatch.setattr(translation, "load_conditional",
                        lambda *args: pytest.fail("completed score must not load translator"))
    evaluation.run_score(args)


def test_out_of_range_cell_fails_before_loading_assets(monkeypatch):
    monkeypatch.setattr(evaluation, "common_assets",
                        lambda *args: pytest.fail("invalid cell must not load assets"))
    with pytest.raises(ValueError, match="cell index"):
        evaluation.run_score(SimpleNamespace(population="full", cell_index=8))


def test_full_score_requires_selected_checkpoint(completed):
    args = completed.args
    args.head = Path(completed.candidates[0]["head"])
    args.candidate_name = completed.candidates[0]["name"]
    args.population, args.cell_index, args.residual_selection = "full", 0, None
    with pytest.raises(ValueError, match="frozen selected"):
        evaluation.run_score(args)


def test_score_records_no_count_only_information_probe(monkeypatch):
    from additive_chain import generation, iwslt_followup, translation
    observed = []
    class Observer:
        def __init__(self, decoder, probe):
            observed.append(probe)
            self.steps = []
    monkeypatch.setattr(iwslt_followup, "StateObserver", Observer)
    monkeypatch.setattr(generation, "synchronize", lambda device: None)
    monkeypatch.setattr(translation, "translate_one", lambda *args, **kwargs: (
        [1], dict(predicted_length=1, scheduled_steps=2, length_prediction_forwards=1)))
    row = dict(id="dev-0", target="a sentence that is translated correctly")
    result = evaluation.score_translations(None,
        SimpleNamespace(decode=lambda ids: row["target"]), None, [row],
        SimpleNamespace(device="cpu"), evaluation.cells("selection")[0])
    assert observed == [False]
    assert result["summary"]["same_state_probability_probes"] is False
    assert result["summary"]["same_state_probe_seconds"] == 0.
    assert result["predictions"][0]["id"] == "dev-0"


@pytest.mark.parametrize("kind,inherit", [("global", False), ("contextual", False),
                                        ("contextual", True)])
def test_decoder_wires_fixed_settings_and_actual_learned_heads(completed, monkeypatch, kind, inherit):
    import torch
    from additive_chain import generation, iwslt_length_refit, translation
    args = completed.args
    entry = completed.candidates[0]
    args.kind, args.head, args.candidate_name = kind, Path(entry["head"]), entry["name"]
    args.population, args.cell_index, args.residual_selection = "selection", 1, None
    args.checkpoint, args.tokenizer = Path("checkpoint"), Path("tokenizer")
    args.output = args.input.with_name("fresh-score")
    args.global_head = Path("inherited") if inherit else None
    captured = {}
    class FakeHead:
        def cuda(self):
            return self
    learned, inherited = FakeHead(), FakeHead() if inherit else None
    saved = dict(checkpoint_id=entry["name"], epoch=entry["epoch"], update=entry["update"], kind=kind)
    monkeypatch.setattr(evaluation, "load_heads", lambda *args: (learned, inherited, saved))
    class Model:
        def load_state_dict(self, state):
            captured["model_state"] = state
        def eval(self):
            return self
        def requires_grad_(self, value):
            captured["frozen"] = not value
            return self
    model, tokenizer, length_head = Model(), object(), object()
    monkeypatch.setattr(translation, "load_conditional", lambda *args: (model, tokenizer))
    monkeypatch.setattr(torch, "load", lambda path, **kwargs:
                        dict(model="frozen-model") if path == args.trained else "counts")
    monkeypatch.setattr(iwslt_length_refit, "load_refit_head", lambda *args: length_head)
    class Decoder:
        def __init__(self, **kwargs):
            captured["decoder"] = kwargs
        def prepare(self):
            captured["prepared"] = True
        def close(self):
            captured["closed"] = True
    monkeypatch.setattr(generation, "ChainDecoder", Decoder)
    def fake_score(actual_model, actual_tokenizer, actual_head, rows, decoder, cell):
        assert (actual_model, actual_tokenizer, actual_head) == (model, tokenizer, length_head)
        assert rows == completed.rows
        assert cell == evaluation.cells("selection")[1]
        assert captured["prepared"]
        return dict(predictions=[dict(id=r["id"]) for r in rows],
                    summary=dict(examples=len(rows), bleu=1., chrf=2.))
    monkeypatch.setattr(evaluation, "score_translations", fake_score)
    evaluation.run_score(args)
    decoder = captured["decoder"]
    assert decoder["count_strength"] == .2
    assert decoder["temperature"] == .85
    assert decoder["top_k"] == 500 and decoder["nucleus_p"] == .9
    assert decoder["sampling"] == "own-marginal"
    assert decoder["global_head"] is (learned if kind == "global" else inherited)
    assert decoder["global_strength"] == (1. if kind == "global" or inherit else 0.)
    assert decoder["contextual_head"] is (learned if kind == "contextual" else None)
    assert decoder["contextual_strength"] == (1. if kind == "contextual" else 0.)
    assert captured["closed"] and captured["frozen"]
    receipt = next(args.output.glob("*/selection/*/result-identity.json"))
    assert read(receipt)["identity"]["candidate"] == entry


def test_head_loader_rejects_wrong_inherited_or_count_asset(monkeypatch):
    from additive_chain import iwslt_residual_assets
    common = dict(model_sha256="model", data_sha256="data", counts_sha256="counts",
                  inherited_global_sha256="inherited")
    saved = dict(kind="contextual", identity=dict(counts_sha256="counts",
                                                inherited_global_sha256="different"))
    monkeypatch.setattr(iwslt_residual_assets, "load_residual_head",
                        lambda *args: (object(), saved))
    with pytest.raises(ValueError, match="Inherited"):
        evaluation.load_heads(Path("context"), "contextual", common, Path("global"))
    saved["identity"]["inherited_global_sha256"] = "inherited"
    saved["identity"]["counts_sha256"] = "different"
    with pytest.raises(ValueError, match="counts"):
        evaluation.load_heads(Path("context"), "contextual", common, Path("global"))
