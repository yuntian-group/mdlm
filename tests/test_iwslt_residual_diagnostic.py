import ast
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from additive_chain import iwslt_residual_diagnostic as diagnostic
from additive_chain import iwslt_residual_eval as evaluation
from additive_chain.backbone import file_sha256
from additive_chain.iwslt_confirm import atomic_json, read
from additive_chain.iwslt_followup import output_name


def candidate(epoch):
    return dict(name=f"epoch-{epoch:04d}", head=f"/remote/epoch-{epoch:04d}.pt",
                sha256=str(epoch) * 64, epoch=epoch, update=epoch * 100)


def selection(common, candidates=None, values=(15., 13., 14.)):
    candidates = candidates or [candidate(i) for i in range(3)]
    evaluated = {c["name"]: [dict(cell=cell, bleu=value, chrf=value)
                 for cell in evaluation.cells("selection")]
                 for c, value in zip(candidates, values)}
    official, ranking = evaluation.choose_checkpoint(candidates, evaluated)
    return dict(format=evaluation.FORMAT, split="dev", kind="contextual", examples=1024,
        manifest=dict(kind="contextual", candidates=candidates, complete=True), common=common,
        selected=official, ranking=ranking, evaluated=evaluated)


def test_choice_uses_existing_shared_ranking_and_preserves_official():
    document = selection({})
    before = deepcopy(document)
    official, learned = diagnostic.diagnostic_choice(document, {})
    assert official == candidate(0)
    assert learned == candidate(2)
    assert document == before


def test_choice_keeps_tie_rule_from_official_ranking():
    document = selection({}, values=(15., 14., 14.))
    assert diagnostic.diagnostic_choice(document, {})[1] == candidate(1)


@pytest.mark.parametrize("damage", ["nonzero-winner", "incomplete", "no-learned", "test",
    "population", "common", "ranking", "cell", "selected"])
def test_choice_rejects_invalid_or_inapplicable_selection(damage):
    document = selection({})
    if damage == "nonzero-winner":
        document = selection({}, values=(10., 12., 11.))
    elif damage == "incomplete":
        document["manifest"]["complete"] = False
    elif damage == "no-learned":
        document = selection({}, [candidate(0)], [15.])
    elif damage == "test":
        document["split"] = "test"
    elif damage == "population":
        document["examples"] = 7283
    elif damage == "common":
        document["common"] = dict(temperature=1.)
    elif damage == "ranking":
        document["ranking"].reverse()
    elif damage == "cell":
        document["evaluated"]["epoch-0001"][0]["cell"]["seed"] = 202
    else:
        document["selected"] = candidate(2)
    with pytest.raises(ValueError):
        diagnostic.diagnostic_choice(document, {})


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    from sacrebleu.metrics import BLEU, CHRF
    rows = [dict(id=f"dev-{i}", source_ids=[1, 2], target_ids=[3, 4, 5],
                 target="this is a correct translation of the source") for i in range(2)]
    monkeypatch.setattr(diagnostic, "FULL_EXAMPLES", len(rows))
    monkeypatch.setattr(diagnostic, "SELECTION_EXAMPLES", len(rows))
    populations = []
    monkeypatch.setattr(evaluation, "development_rows",
                        lambda data, population: populations.append(population) or rows)
    common = dict(model_sha256="model", data_sha256="data", counts_sha256="counts",
        inherited_global_sha256="global", length_head_sha256="length", temperature=.85,
        count_strength=.2, residual_strength=1., top_k=500, nucleus_p=.9, seed=101)
    monkeypatch.setattr(evaluation, "common_assets", lambda args: common)
    monkeypatch.setattr(evaluation, "source_hashes", lambda: {"heads.py": "frozen-evaluator"})
    candidates, saved = [], {}
    class Head:
        def cuda(self):
            return self
    contextual, inherited = Head(), Head()
    loaded = []
    for epoch in range(3):
        path = tmp_path / f"epoch-{epoch:04d}.pt"
        path.write_bytes(f"checkpoint {epoch}".encode())
        metadata = dict(checkpoint_id=f"epoch-{epoch:04d}", epoch=epoch, update=epoch * 100,
                        kind="contextual", identity=dict(source_sha256={"heads.py": "training"}))
        saved[str(path)] = metadata
        candidates.append(evaluation.candidate_entry(path, metadata))
    def load(path, kind, assets, parent):
        assert kind == "contextual" and assets == common and parent == tmp_path / "global.pt"
        loaded.append(str(path))
        return contextual, inherited, saved[str(path)]
    monkeypatch.setattr(evaluation, "load_heads", load)
    official_path = tmp_path / "selected.json"
    atomic_json(official_path, selection(common, candidates))
    args = SimpleNamespace(trained=tmp_path / "trained.pt", data=tmp_path / "data",
        length_head=tmp_path / "length.pt", selection=tmp_path / "count-selection.json",
        residual_selection=official_path, output=tmp_path / "diagnostics",
        official_evaluation_root=tmp_path / "official-evaluation",
        global_head=tmp_path / "global.pt", checkpoint=tmp_path / "pretrained",
        tokenizer=tmp_path / "tokenizer", cell_index=0)
    def payload(cell):
        records = [dict(id=row["id"], hypothesis=row["target"], reference=row["target"],
            predicted_length=4, scheduled_steps=cell["steps"], actual_forward_calls=2,
            length_prediction_forwards=1, reveal_counts=[3, 1] + [0] * (cell["steps"] - 2),
            seconds=1., same_state_probe_seconds=0.,
            opportunities=[dict(masked=4, revealed=3, component_reveals=[3]),
                           dict(masked=1, revealed=1, component_reveals=[1])]) for row in rows]
        hypotheses = [r["hypothesis"] for r in records]
        references = [[r["reference"] for r in records]]
        return dict(predictions=records, summary=dict(examples=len(rows), seconds=float(len(rows)),
            same_state_probe_seconds=0., same_state_probability_probes=False,
            bleu=BLEU(lowercase=True, tokenize="13a").corpus_score(hypotheses, references).score,
            chrf=CHRF().corpus_score(hypotheses, references).score))
    reference_cell = evaluation.cells("selection")[0]
    reference_relative = f"{candidates[0]['name']}/selection/{output_name(reference_cell)}/result.json"
    reference = args.official_evaluation_root / reference_relative
    reference.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(reference, payload(reference_cell))
    atomic_json(reference.with_name("result-identity.json"), dict(
        identity=evaluation.build_identity(reference_cell, rows, "selection", "contextual",
                                           candidates[0], common),
        output_sha256=file_sha256(reference)))
    document = read(official_path)
    document["outputs"] = {reference_relative: file_sha256(reference)}
    atomic_json(official_path, document)
    def write_grid():
        state = diagnostic.prepare(args)
        paths = []
        for cell in evaluation.cells("full"):
            path = diagnostic.cell_folder(state, cell) / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            value = payload(cell)
            atomic_json(path, value)
            atomic_json(path.with_name("result-identity.json"), dict(
                identity=diagnostic.build_identity(state, cell), output_sha256=file_sha256(path)))
            atomic_json(path.with_name("summary.json"),
                        dict(cell=cell, completed=1, expected=1, **value["summary"]))
            paths.append(path)
        return paths
    return SimpleNamespace(args=args, rows=rows, common=common, candidates=candidates,
        saved=saved, contextual=contextual, inherited=inherited, payload=payload,
        write_grid=write_grid, populations=populations, loaded=loaded,
        original=official_path.read_bytes(), reference=reference)


def test_prepare_checks_actual_checkpoint_and_full_development(experiment):
    state = diagnostic.prepare(experiment.args)
    assert state.candidate == experiment.candidates[2]
    assert state.official == experiment.candidates[0]
    assert experiment.loaded == [experiment.candidates[2]["head"]]
    assert experiment.populations == ["selection", "full"]
    assert state.source_sha256 != state.training_source_sha256
    assert state.runner_sha256 == file_sha256(Path(diagnostic.__file__))


def test_prepare_rejects_wrong_imported_evaluator(experiment, monkeypatch):
    monkeypatch.setattr(evaluation, "source_hashes", lambda: {"heads.py": "different-runtime"})
    with pytest.raises(ValueError, match="identity/hash changed"):
        diagnostic.prepare(experiment.args)


def test_prepare_rejects_changed_official_reference_output(experiment):
    experiment.reference.write_bytes(b"changed official output")
    with pytest.raises(ValueError, match="reference differs"):
        diagnostic.prepare(experiment.args)


def test_prepare_rejects_wrong_official_reference_population(experiment):
    rewrite_payload(experiment.reference,
                    lambda value: value["predictions"][0].update(id="not-original"))
    document = read(experiment.args.residual_selection)
    relative = next(iter(document["outputs"]))
    document["outputs"][relative] = file_sha256(experiment.reference)
    atomic_json(experiment.args.residual_selection, document)
    with pytest.raises(ValueError, match="Incomplete or reordered"):
        diagnostic.prepare(experiment.args)


@pytest.mark.parametrize("damage", ["hash", "metadata", "population", "duplicate", "assets"])
def test_prepare_rejects_changed_assets_or_population(experiment, monkeypatch, damage):
    if damage == "hash":
        Path(experiment.candidates[2]["head"]).write_bytes(b"changed")
    elif damage == "metadata":
        experiment.saved[experiment.candidates[2]["head"]]["update"] = 199
    elif damage == "population":
        monkeypatch.setattr(evaluation, "development_rows", lambda *args: experiment.rows[:1])
    elif damage == "duplicate":
        monkeypatch.setattr(evaluation, "development_rows", lambda *args: [experiment.rows[0]] * 2)
    else:
        monkeypatch.setattr(evaluation, "common_assets", lambda args: dict(experiment.common, temperature=1.))
    with pytest.raises(ValueError):
        diagnostic.prepare(experiment.args)


def test_complete_readout_is_distinct_and_preserves_official(experiment):
    experiment.write_grid()
    diagnostic.run_readout(experiment.args)
    result = read(experiment.args.output / diagnostic.NAMESPACE / "full-dev-readout.json")
    assert result["role"] == "diagnostic" and result["policy"] == diagnostic.POLICY
    assert result["official_selected"] == experiment.candidates[0]
    assert result["diagnostic_candidate"] == experiment.candidates[2]
    assert result["official_selection_sha256"] == file_sha256(experiment.args.residual_selection)
    assert result["runner_sha256"] == file_sha256(Path(diagnostic.__file__))
    assert len(result["outputs"]) == 8 and set(result["steps"]) == {"2", "4", "8", "32"}
    assert all(block["joint_minus_own_marginal"] == dict(bleu=0., chrf=0.)
               for block in result["steps"].values())
    assert experiment.args.residual_selection.read_bytes() == experiment.original
    assert not (experiment.args.output / "full-dev-readout.json").exists()


def rewrite_payload(path, transform):
    value = read(path)
    transform(value)
    atomic_json(path, value)
    receipt_path = path.with_name("result-identity.json")
    receipt = read(receipt_path)
    receipt["output_sha256"] = file_sha256(path)
    atomic_json(receipt_path, receipt)


@pytest.mark.parametrize("damage", ["incomplete", "extra", "wrong-head", "source", "runner",
    "policy", "selection-hash", "seed", "reference", "id", "score", "schedule", "oracle",
    "probe", "length-forwards", "nonfinite-time"])
def test_readout_rejects_incomplete_mixed_or_invalid_cells(experiment, damage):
    paths = experiment.write_grid()
    path = paths[-1]
    if damage == "incomplete":
        path.parent.rename(path.parent.with_name("missing-cell"))
    elif damage == "extra":
        (path.parent.parent / "T999-extra").mkdir()
    elif damage in {"wrong-head", "source", "runner", "policy", "selection-hash", "seed"}:
        receipt_path = path.with_name("result-identity.json")
        receipt = read(receipt_path)
        identity = receipt["identity"]
        if damage == "wrong-head":
            identity["diagnostic_candidate"] = experiment.candidates[1]
        elif damage == "source":
            identity["source_sha256"] = {"heads.py": "different"}
        elif damage == "runner":
            identity["runner_sha256"] = "different"
        elif damage == "policy":
            identity["role"] = "official"
        elif damage == "selection-hash":
            identity["official_selection_sha256"] = "different"
        else:
            identity["cell"]["seed"] = 202
        atomic_json(receipt_path, receipt)
    else:
        def transform(value):
            row = value["predictions"][0]
            if damage in {"reference", "id"}:
                row[damage] = "changed"
            elif damage == "score":
                value["summary"]["bleu"] = 1.
            elif damage == "schedule":
                row["reveal_counts"] = [2, 2]
                row["opportunities"] = [dict(masked=4, revealed=2, component_reveals=[2]),
                                        dict(masked=2, revealed=2, component_reveals=[2])]
            elif damage == "oracle":
                row["length_mode"] = "reference-diagnostic"
            elif damage == "probe":
                row["same_state_probe_seconds"] = .1
            elif damage == "length-forwards":
                row["length_prediction_forwards"] = 2
            else:
                # JSON writer forbids NaN, so use a negative duration instead.
                row["seconds"] = -1.
        rewrite_payload(path, transform)
    with pytest.raises((ValueError, FileNotFoundError)):
        diagnostic.run_readout(experiment.args)
    assert experiment.args.residual_selection.read_bytes() == experiment.original


def test_completed_cell_resume_does_not_load_backbone(experiment, monkeypatch):
    from additive_chain import translation
    experiment.write_grid()
    monkeypatch.setattr(translation, "load_conditional",
                        lambda *args: pytest.fail("completed diagnostic should not load backbone"))
    diagnostic.run_score(experiment.args)


@pytest.mark.parametrize("cell_index", range(8))
def test_score_uses_one_learned_head_and_fixed_frozen_decoder(experiment, monkeypatch, cell_index):
    import torch
    from additive_chain import generation, iwslt_length_refit, translation
    args = experiment.args
    args.cell_index = cell_index
    captured = {}
    class Model:
        def load_state_dict(self, state):
            captured["model"] = state
        def eval(self):
            return self
        def requires_grad_(self, value):
            captured["frozen"] = not value
            return self
    model, tokenizer, length_head = Model(), object(), object()
    monkeypatch.setattr(translation, "load_conditional", lambda *args: (model, tokenizer))
    monkeypatch.setattr(torch, "load", lambda path, **kwargs:
                        dict(model="translator") if path == args.trained else "counts")
    monkeypatch.setattr(iwslt_length_refit, "load_refit_head", lambda *args: length_head)
    class Decoder:
        def __init__(self, **kwargs):
            captured["decoder"] = kwargs
        def prepare(self):
            captured["prepared"] = True
        def close(self):
            captured["closed"] = True
    monkeypatch.setattr(generation, "ChainDecoder", Decoder)
    cell = evaluation.cells("full")[cell_index]
    def score(actual_model, actual_tokenizer, actual_length, rows, decoder, actual_cell):
        assert (actual_model, actual_tokenizer, actual_length) == (model, tokenizer, length_head)
        assert rows == experiment.rows and actual_cell == cell and captured["prepared"]
        return experiment.payload(cell)
    monkeypatch.setattr(evaluation, "score_translations", score)
    diagnostic.run_score(args)
    assert captured["decoder"] == dict(mask_id=translation.MASK_ID, device="cuda",
        count_statistics="counts", count_strength=.2, top_k=500, nucleus_p=.9,
        fast_candidates=True, sampling=cell["sampling"], temperature=.85,
        global_head=experiment.inherited, global_strength=1.,
        contextual_head=experiment.contextual, contextual_strength=1.)
    assert captured["closed"] and captured["frozen"]
    receipt_path = (args.output / diagnostic.NAMESPACE / experiment.candidates[2]["name"] /
                    "full" / output_name(cell) / "result-identity.json")
    receipt = read(receipt_path)
    assert receipt["identity"]["diagnostic_candidate"] == experiment.candidates[2]
    assert receipt["identity"]["official_selected"] == experiment.candidates[0]
    assert experiment.args.residual_selection.read_bytes() == experiment.original


def test_invalid_cell_fails_before_reading_assets(monkeypatch):
    monkeypatch.setattr(evaluation, "common_assets", lambda *args: pytest.fail("assets accessed"))
    with pytest.raises(ValueError, match="cell index"):
        diagnostic.run_score(SimpleNamespace(cell_index=8))


def test_runner_uses_only_absolute_imports():
    parsed = ast.parse(Path(diagnostic.__file__).read_text())
    assert all(node.level == 0 for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom))


def test_runner_can_import_frozen_package_when_run_by_external_path(tmp_path):
    runner = tmp_path / "diagnostic_runner.py"
    runner.write_bytes(Path(diagnostic.__file__).read_bytes())
    repository = Path(evaluation.__file__).resolve().parent.parent
    environment = dict(os.environ, PYTHONPATH=str(repository))
    command = [sys.executable, "-c", "import runpy, sys; "
               "scope = runpy.run_path(sys.argv[1]); "
               "print(scope['evaluation'].__file__); print(scope['runner_sha256']())", str(runner)]
    completed = subprocess.run(command, cwd=tmp_path, env=environment, text=True,
                               capture_output=True, check=True, timeout=30)
    assert completed.stdout.splitlines() == [str(Path(evaluation.__file__).resolve()), file_sha256(runner)]
