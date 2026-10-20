"""Evaluate a learned contextual head when zero wins checkpoint selection.

The diagnostic uses the highest-ranked nonzero checkpoint from the completed
development ranking. It does not replace the official, zero-inclusive winner.
Run this file by absolute path to use an existing frozen evaluator package on
PYTHONPATH. Absolute imports keep decoding in that package; the runner's own
source hash is recorded separately.
"""
import argparse
import math
from pathlib import Path
from types import SimpleNamespace

from additive_chain import iwslt_residual_eval as evaluation
from additive_chain.backbone import file_sha256
from additive_chain.iwslt_confirm import atomic_json, read
from additive_chain.iwslt_controls_readout import (
    length_errors, require_same_schedule, verify_population)
from additive_chain.iwslt_followup import output_name, verify_output


FORMAT = "iwslt_contextual_nonzero_diagnostic_v1"
POLICY = "highest-ranked-nonzero-from-official-zero-inclusive-ranking"
NAMESPACE = "highest-ranked-nonzero"
FULL_EXAMPLES = 7283
SELECTION_EXAMPLES = 1024


def diagnostic_choice(document, common):
    official = evaluation.validate_selected(document, common, "contextual")
    if document.get("examples") != 1024 or document["manifest"].get("complete") is not True:
        raise ValueError("Diagnostic requires completed development checkpoint selection")
    if (official["name"], official["epoch"], official["update"]) != ("epoch-0000", 0, 0):
        raise ValueError("Diagnostic requires the official zero-residual winner")
    candidate = next((row["candidate"] for row in document["ranking"]
                      if row["candidate"]["update"] > 0), None)
    if candidate is None:
        raise ValueError("Official ranking contains no nonzero checkpoint")
    return official, candidate


def runner_sha256():
    return file_sha256(Path(__file__))


def frozen_reference(args, document, common, official):
    """Bind the imported evaluator to a completed original-selection output."""
    cell = evaluation.cells("selection")[0]
    relative = f"{official['name']}/selection/{output_name(cell)}/result.json"
    path = args.official_evaluation_root / relative
    expected_sha = document["outputs"].get(relative)
    if expected_sha is None or file_sha256(path) != expected_sha:
        raise ValueError("Official evaluator reference differs from checkpoint selection")
    rows = evaluation.development_rows(args.data, "selection")
    if len(rows) != SELECTION_EXAMPLES or len({row["id"] for row in rows}) != SELECTION_EXAMPLES:
        raise ValueError("Invalid official checkpoint-selection population")
    identity = evaluation.build_identity(cell, rows, "selection", "contextual", official, common)
    # This also rejects a different imported evaluator's source hashes.
    verify_output(path, identity)
    return dict(relative_path=relative, output_sha256=expected_sha,
                identity_sha256=file_sha256(path.with_name("result-identity.json")))


def prepare(args):
    common = evaluation.common_assets(args)
    selection_sha = file_sha256(args.residual_selection)
    document = read(args.residual_selection)
    official, candidate = diagnostic_choice(document, common)
    reference = frozen_reference(args, document, common, official)
    head, inherited, saved = evaluation.load_heads(
        Path(candidate["head"]), "contextual", common, args.global_head)
    if evaluation.candidate_entry(candidate["head"], saved) != candidate:
        raise ValueError("Diagnostic checkpoint metadata or hash changed")
    rows = evaluation.development_rows(args.data, "full")
    if len(rows) != FULL_EXAMPLES or len({row["id"] for row in rows}) != FULL_EXAMPLES:
        raise ValueError("Diagnostic requires the full, unique development population")
    root = args.output / NAMESPACE
    if args.residual_selection.resolve().is_relative_to(root.resolve()):
        raise ValueError("Official selection must remain outside diagnostic outputs")
    return SimpleNamespace(common=common, selection_sha=selection_sha, official=official,
        candidate=candidate, head=head, inherited=inherited, rows=rows, root=root,
        frozen_evaluator_reference=reference,
        source_sha256=evaluation.source_hashes(), runner_sha256=runner_sha256(),
        training_source_sha256=saved["identity"]["source_sha256"])


def build_identity(state, cell):
    if cell not in evaluation.cells("full"):
        raise ValueError("Unregistered diagnostic decoding cell")
    return dict(format=FORMAT, experiment="iwslt-contextual-nonzero-diagnostic",
        split="dev", kind="contextual", population="full", role="diagnostic",
        policy=POLICY, official_selected=state.official, diagnostic_candidate=state.candidate,
        official_selection_sha256=state.selection_sha, common=state.common,
        frozen_evaluator_reference=state.frozen_evaluator_reference,
        cell=cell, ids=[row["id"] for row in state.rows],
        source_sha256=state.source_sha256, runner_sha256=state.runner_sha256,
        training_source_sha256=state.training_source_sha256)


def unchanged(args, state):
    reference = args.official_evaluation_root / state.frozen_evaluator_reference["relative_path"]
    if (evaluation.common_assets(args) != state.common
            or file_sha256(args.residual_selection) != state.selection_sha
            or file_sha256(Path(state.candidate["head"])) != state.candidate["sha256"]
            or evaluation.source_hashes() != state.source_sha256
            or runner_sha256() != state.runner_sha256
            or file_sha256(reference) != state.frozen_evaluator_reference["output_sha256"]
            or file_sha256(reference.with_name("result-identity.json"))
                != state.frozen_evaluator_reference["identity_sha256"]):
        raise ValueError("Diagnostic inputs or source changed")


def cell_folder(state, cell):
    return state.root / state.candidate["name"] / "full" / output_name(cell)


def validate_payload(payload, state, cell):
    from sacrebleu.metrics import BLEU, CHRF
    from sacremoses import MosesDetokenizer
    records, summary = payload["predictions"], payload["summary"]
    if (summary["examples"] != len(state.rows)
            or [row["id"] for row in records] != [row["id"] for row in state.rows]):
        raise ValueError("Incomplete or reordered diagnostic development population")
    detok = MosesDetokenizer(lang="en")
    references = [detok.detokenize(row["target"].split()) for row in state.rows]
    diagnostics = verify_population(records, state.rows, references, cell["steps"])
    if any(len(row["reveal_counts"]) != cell["steps"]
           or row["length_prediction_forwards"] != 1
           or any(type(n) is not int for n in row["reveal_counts"])
           or row["same_state_probe_seconds"] != 0.
           or not math.isfinite(row["seconds"]) or row["seconds"] < 0.
           or any(s.get("same_state_pairs") for s in row["opportunities"])
           for row in records):
        raise ValueError("Diagnostic schedule, length forwards, timing, or probes changed")
    if (summary.get("same_state_probability_probes") is not False
            or summary["same_state_probe_seconds"] != 0.
            or not math.isfinite(summary["seconds"])
            or abs(summary["seconds"] - sum(row["seconds"] for row in records)) > 1e-10):
        raise ValueError("Invalid diagnostic timing/probe summary")
    metrics = {}
    hypotheses = [row["hypothesis"] for row in records]
    for key, metric in (("bleu", BLEU(lowercase=True, tokenize="13a")), ("chrf", CHRF())):
        value = metric.corpus_score(hypotheses, [references]).score
        if not math.isfinite(summary[key]) or abs(value - summary[key]) > 1e-10:
            raise ValueError("Diagnostic quality score cannot be reproduced")
        metrics[key] = value
    return dict(cell=cell, **metrics, diagnostics=diagnostics)


def run_score(args):
    import torch
    from additive_chain.generation import ChainDecoder
    from additive_chain.iwslt_length_refit import load_refit_head
    from additive_chain.iwslt_run import log, run_lock
    from additive_chain.translation import MASK_ID, load_conditional
    registered = evaluation.cells("full")
    if type(args.cell_index) is not int or not 0 <= args.cell_index < len(registered):
        raise ValueError("Unregistered diagnostic cell index")
    cell = registered[args.cell_index]
    state = prepare(args)
    identity = build_identity(state, cell)
    folder = cell_folder(state, cell)
    with run_lock(folder):
        path = folder / "result.json"
        completed = path.with_name("result-identity.json").exists()
        if completed:
            result = verify_output(path, identity)
        else:
            if path.exists():
                raise ValueError("Diagnostic output without completion receipt")
            model, tokenizer = load_conditional(args.checkpoint, args.tokenizer)
            trained = torch.load(args.trained, map_location="cuda", weights_only=False)
            model.load_state_dict(trained["model"])
            del trained
            model.eval().requires_grad_(False)
            length_head = load_refit_head(args.length_head, state.common["model_sha256"],
                                          state.common["data_sha256"], "cuda")
            head = state.head.cuda()
            inherited = state.inherited.cuda() if state.inherited is not None else None
            decoder = ChainDecoder(mask_id=MASK_ID, device="cuda",
                count_statistics=torch.load(args.data / "counts.pt", weights_only=True),
                count_strength=.2, top_k=500, nucleus_p=.9, fast_candidates=True,
                sampling=cell["sampling"], temperature=.85,
                global_head=inherited, global_strength=1. if inherited is not None else 0.,
                contextual_head=head, contextual_strength=1.)
            try:
                decoder.prepare()
                result = evaluation.score_translations(model, tokenizer, length_head,
                                                        state.rows, decoder, cell)
            finally:
                decoder.close()
        validate_payload(result, state, cell)
        unchanged(args, state)
        if not completed:
            atomic_json(path, result)
            atomic_json(path.with_name("result-identity.json"),
                        dict(identity=identity, output_sha256=file_sha256(path)))
        atomic_json(folder / "summary.json",
                    dict(completed=1, expected=1, cell=cell, **result["summary"]))
        log("contextual_nonzero_diagnostic_complete", candidate=state.candidate["name"],
            official_candidate=state.official["name"], **cell, **result["summary"])


def run_readout(args):
    state = prepare(args)
    registered = evaluation.cells("full")
    folder = state.root / state.candidate["name"] / "full"
    observed = {path.name for path in folder.glob("T*") if path.is_dir()}
    if observed != {output_name(cell) for cell in registered}:
        raise ValueError("Incomplete or unexpected eight-cell diagnostic grid")
    scores, outputs, paired, first = [], {}, {}, None
    for cell in registered:
        path = cell_folder(state, cell) / "result.json"
        progress = read(path.with_name("summary.json"))
        if progress["cell"] != cell or (progress["completed"], progress["expected"]) != (1, 1):
            raise ValueError("Incomplete diagnostic cell summary")
        payload = verify_output(path, build_identity(state, cell))
        scores.append(validate_payload(payload, state, cell))
        records = payload["predictions"]
        if first is None:
            first = records
        elif [row["predicted_length"] for row in records] != [row["predicted_length"] for row in first]:
            raise ValueError("Diagnostic predicted lengths differ across cells")
        if cell["steps"] in paired:
            previous = paired[cell["steps"]]
            require_same_schedule(previous, records)
            for key in ("actual_forward_calls", "length_prediction_forwards", "opportunities"):
                if [row[key] for row in previous] != [row[key] for row in records]:
                    raise ValueError("Diagnostic paired reveal diagnostics differ")
        else:
            paired[cell["steps"]] = records
        outputs[str(path.relative_to(state.root))] = file_sha256(path)
    steps = {}
    for step in (2, 4, 8, 32):
        methods = {row["cell"]["method"]: row for row in scores if row["cell"]["steps"] == step}
        steps[str(step)] = dict(methods=methods, joint_minus_own_marginal={
            metric: methods["joint"][metric] - methods["own-marginal"][metric]
            for metric in ("bleu", "chrf")})
    result = dict(format=FORMAT, split="dev", kind="contextual", role="diagnostic",
        policy=POLICY, examples=len(state.rows), official_selected=state.official,
        diagnostic_candidate=state.candidate, official_selection_sha256=state.selection_sha,
        common=state.common, source_sha256=state.source_sha256,
        frozen_evaluator_reference=state.frozen_evaluator_reference,
        runner_sha256=state.runner_sha256, training_source_sha256=state.training_source_sha256,
        steps=steps, outputs=outputs, length_errors=length_errors(first, state.rows))
    unchanged(args, state)
    output = state.root / "full-dev-readout.json"
    if output.exists() and read(output) != result:
        raise ValueError("Refusing to replace a different diagnostic readout")
    atomic_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("score", "readout"):
        sub = commands.add_parser(command)
        for name in ("trained", "data", "length-head", "selection", "residual-selection",
                     "official-evaluation-root", "output"):
            sub.add_argument("--" + name, type=Path, required=True)
        sub.add_argument("--global-head", type=Path)
        if command == "score":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--tokenizer", type=Path, required=True)
            sub.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args()
    {"score": run_score, "readout": run_readout}[args.command](args)


if __name__ == "__main__":
    main()
