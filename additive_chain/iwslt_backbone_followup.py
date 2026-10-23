"""Fixed-setting development comparisons after continuing the IWSLT backbone.

The original temperature/count-strength selection remains immutable. A separate
manifest records that those settings are transferred, not retuned, to a new
backbone and a length head fitted for it. Learned pair heads must belong to the
new backbone. Checkpoint selection retains the zero residual and uses both
samplers at two and four steps.
"""
import argparse
import math
from pathlib import Path

from . import iwslt_residual_eval as evaluation
from .backbone import file_sha256
from .iwslt_confirm import atomic_json, ordered_rows, read
from .iwslt_controls import head_fingerprint
from .iwslt_controls_readout import require_same_schedule, verify_population
from .iwslt_followup import output_name, select_settings, stratified_rows, verify_output
from .iwslt_length_refit import load_refit_head

FORMAT = "iwslt_backbone_fixed_transfer_v1"
SELECTION_FORMAT = "iwslt_backbone_residual_selection_v1"
STAGES = ("baseline", "count", "global", "contextual")
PROTOCOL = dict(temperature=.85, count_strength=.2, global_strength=1.,
                contextual_strength=1., nucleus_p=.9, top_k=500, seed=101,
                length_mode="predicted", steps=[2, 4, 8, 32])


def cells(stage, population="full"):
    if stage not in STAGES or population not in ("full", "selection"):
        raise ValueError("Unknown development stage or population")
    if population == "selection" and stage not in ("global", "contextual"):
        raise ValueError("Only learned residuals use checkpoint selection")
    if stage == "baseline":
        return [dict(steps=t, temperature=.85, strength=0., sampling="joint",
                     seed=101, method="baseline", length_mode="predicted")
                for t in (32, 8, 4, 2)]
    return evaluation.cells(population)


def source_hashes():
    result = evaluation.source_hashes()
    for name in ("iwslt_backbone_followup.py", "iwslt_confirm.py", "iwslt_controls.py",
                 "iwslt_followup_readout.py"):
        result[name] = file_sha256(Path(__file__).with_name(name))
    return result


def check_inherited_settings(document, data_sha):
    """Check the old selection without pretending it was run on the new model."""
    if document.get("data_sha256") != data_sha:
        raise ValueError("Inherited selection used different data")
    if any(not math.isfinite(row[metric]) or not 0. <= row[metric] <= 100.
           for row in document["cells"] for metric in ("bleu", "chrf") if metric in row):
        raise ValueError("Invalid inherited development quality score")
    if document.get("selected") != select_settings(document["cells"]):
        raise ValueError("Inherited selection does not follow its recorded scores")
    if set(document["selected"]) != {"8", "32"} or any(
            (v["baseline_temperature"], v["crf_temperature"], v["strength"])
            != (.85, .85, .2) for v in document["selected"].values()):
        raise ValueError("Inherited settings differ from the fixed protocol")
    old_model = document.get("model_sha256")
    if (not isinstance(old_model, str) or len(old_model) != 64
            or any(c not in "0123456789abcdef" for c in old_model)):
        raise ValueError("Invalid source-selection model hash")


def freeze_document(args):
    model_sha = file_sha256(args.trained)
    data_sha = file_sha256(args.data / "manifest.json")
    counts_sha = file_sha256(args.data / "counts.pt")
    if counts_sha != read(args.data / "manifest.json")["counts_sha256"]:
        raise ValueError("Training counts changed")
    old = read(args.source_selection)
    check_inherited_settings(old, data_sha)
    length = load_refit_head(args.length_head, model_sha, data_sha, "cpu")
    return dict(format=FORMAT, split="dev", policy="transfer-fixed-settings-without-retuning",
        protocol=PROTOCOL, model_sha256=model_sha, data_sha256=data_sha,
        counts_sha256=counts_sha,
        length_head_sha256=head_fingerprint(length.state_dict()),
        length_head_artifact_sha256=file_sha256(args.length_head),
        source_selection=dict(path=str(args.source_selection.resolve()),
            sha256=file_sha256(args.source_selection), model_sha256=old["model_sha256"],
            selected=old["selected"], source_sha256=old["source_sha256"]),
        source_sha256=source_hashes())


def immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and read(path) != value:
        raise ValueError("Refusing to change a frozen document")
    atomic_json(path, value)


def run_freeze(args):
    document = freeze_document(args)
    if freeze_document(args) != document:
        raise ValueError("Inputs changed while freezing settings")
    immutable_json(args.output, document)


def common_assets(args):
    from types import SimpleNamespace
    frozen = read(args.protocol)
    if frozen.get("format") != FORMAT:
        raise ValueError("Wrong fixed-transfer manifest")
    source = Path(frozen["source_selection"]["path"])
    actual = freeze_document(SimpleNamespace(**dict(vars(args), source_selection=source)))
    if frozen != actual:
        raise ValueError("Transferred settings, backbone, length head, data or code changed")
    if args.stage != "contextual" and getattr(args, "global_head", None) is not None:
        raise ValueError("Only contextual residuals inherit a global head")
    if args.stage == "contextual" and getattr(args, "global_head", None) is None:
        raise ValueError("Contextual residuals require the selected global head")
    return dict(model_sha256=actual["model_sha256"], data_sha256=actual["data_sha256"],
        counts_sha256=actual["counts_sha256"],
        length_head_sha256=actual["length_head_sha256"],
        length_head_artifact_sha256=actual["length_head_artifact_sha256"],
        inherited_global_sha256=(file_sha256(args.global_head)
            if getattr(args, "global_head", None) else None),
        protocol_sha256=file_sha256(args.protocol),
        source_selection_sha256=actual["source_selection"]["sha256"],
        protocol=PROTOCOL)


def rows_for(args):
    cells(args.stage, args.population)
    rows = ordered_rows(args.data, "dev")
    return stratified_rows(rows, 1024) if args.population == "selection" else rows


def identity(cell, rows, stage, population, common, candidate, selection_sha=None):
    if cell not in cells(stage, population):
        raise ValueError("Unregistered fixed-setting comparison")
    return dict(experiment="iwslt-continued-backbone", split="dev", stage=stage,
        population=population, common=common, candidate=candidate,
        residual_selection_sha256=selection_sha, cell=cell,
        ids=[r["id"] for r in rows], source_sha256=source_hashes())


def folder_for(root, stage, candidate, population, cell):
    name = candidate["name"] if candidate else "fixed"
    return root / stage / name / population / output_name(cell)


def validate_selected(document, common, stage):
    if (document.get("format") != SELECTION_FORMAT or document.get("split") != "dev"
            or document.get("stage") != stage or document.get("common") != common):
        raise ValueError("Residual selection belongs to a different comparison")
    manifest = document["manifest"]
    if manifest.get("kind") != stage or manifest.get("complete") is not True:
        raise ValueError("Completed candidate manifest required")
    candidates = evaluation.validate_candidates(manifest)
    winner, ranking = evaluation.choose_checkpoint(candidates, document["evaluated"])
    if document.get("selected") != winner or document.get("ranking") != ranking:
        raise ValueError("Residual selection does not follow the shared development criterion")
    return winner


def resolve_head(args, common):
    if args.stage in ("baseline", "count"):
        if getattr(args, "head", None) or getattr(args, "residual_selection", None):
            raise ValueError("Baseline/count do not consume learned-head selections")
        return None, None, None, None
    if not getattr(args, "head", None):
        raise ValueError("Learned residual checkpoint required")
    head, parent, saved = evaluation.load_heads(args.head, args.stage, common, args.global_head)
    candidate = evaluation.candidate_entry(args.head, saved)
    selection_sha = None
    if args.population == "full":
        if not args.residual_selection:
            raise ValueError("Full development requires a shared selected checkpoint")
        if validate_selected(read(args.residual_selection), common, args.stage) != candidate:
            raise ValueError("Head differs from the selected checkpoint")
        selection_sha = file_sha256(args.residual_selection)
    elif args.residual_selection:
        raise ValueError("Checkpoint selection must not consume its winner")
    return head, parent, candidate, selection_sha


def run_score(args):
    import torch
    from .generation import ChainDecoder
    from .iwslt_run import log, run_lock
    from .translation import MASK_ID, load_conditional
    registered = cells(args.stage, args.population)
    if type(args.cell_index) is not int or not 0 <= args.cell_index < len(registered):
        raise ValueError("Unregistered cell index")
    cell, common, rows = registered[args.cell_index], common_assets(args), rows_for(args)
    head, parent, candidate, selected_sha = resolve_head(args, common)
    expected = identity(cell, rows, args.stage, args.population, common, candidate, selected_sha)
    folder = folder_for(args.output, args.stage, candidate, args.population, cell)
    with run_lock(folder):
        path = folder / "result.json"
        if path.with_name("result-identity.json").exists():
            result = verify_output(path, expected)
        else:
            if path.exists():
                raise ValueError("Result without completion identity")
            model, tokenizer = load_conditional(args.checkpoint, args.tokenizer)
            saved = torch.load(args.trained, map_location="cuda", weights_only=False)
            model.load_state_dict(saved["model"])
            del saved
            model.eval().requires_grad_(False)
            length = load_refit_head(args.length_head, common["model_sha256"], common["data_sha256"], "cuda")
            if head is not None:
                head = head.cuda()
            if parent is not None:
                parent = parent.cuda()
            decoder = ChainDecoder(mask_id=MASK_ID, device="cuda", top_k=500,
                nucleus_p=.9, fast_candidates=True, temperature=.85,
                count_statistics=torch.load(args.data / "counts.pt", weights_only=True),
                count_strength=cell["strength"], sampling=cell["sampling"],
                global_head=head if args.stage == "global" else parent,
                global_strength=1. if args.stage in ("global", "contextual") else 0.,
                contextual_head=head if args.stage == "contextual" else None,
                contextual_strength=1. if args.stage == "contextual" else 0.)
            decoder.prepare()
            try:
                result = evaluation.score_translations(model, tokenizer, length, rows, decoder, cell)
            finally:
                decoder.close()
            if (common_assets(args) != common
                    or (candidate and file_sha256(args.head) != candidate["sha256"])
                    or (selected_sha and file_sha256(args.residual_selection) != selected_sha)):
                raise ValueError("Comparison inputs changed during decoding")
            atomic_json(path, result)
            atomic_json(path.with_name("result-identity.json"),
                        dict(identity=expected, output_sha256=file_sha256(path)))
        atomic_json(folder / "summary.json", dict(completed=1, expected=1, cell=cell, **result["summary"]))
        log("backbone_followup_complete", stage=args.stage, **cell, **result["summary"])


def verified_scores(root, stage, population, common, rows, candidate=None, selection_sha=None):
    from sacrebleu.metrics import BLEU, CHRF
    from sacremoses import MosesDetokenizer
    registered = cells(stage, population)
    base = folder_for(root, stage, candidate, population, registered[0]).parent
    if {p.name for p in base.glob("T*") if p.is_dir()} != {output_name(c) for c in registered}:
        raise ValueError("Incomplete or unexpected comparison grid")
    detok = MosesDetokenizer(lang="en")
    references = [detok.detokenize(r["target"].split()) for r in rows]
    scores, hashes, paired, first_lengths = [], {}, {}, None
    for cell in registered:
        path = folder_for(root, stage, candidate, population, cell) / "result.json"
        payload = verify_output(path, identity(cell, rows, stage, population, common, candidate, selection_sha))
        records, summary = payload["predictions"], payload["summary"]
        if read(path.with_name("summary.json")) != dict(completed=1, expected=1, cell=cell, **summary):
            raise ValueError("Comparison summary differs from completed output")
        diagnostics = verify_population(records, rows, references, cell["steps"])
        if any(r["length_prediction_forwards"] != 1 or len(r["reveal_counts"]) != cell["steps"] for r in records):
            raise ValueError("Changed length or reveal schedule")
        lengths = [r["predicted_length"] for r in records]
        if first_lengths is not None and first_lengths != lengths:
            raise ValueError("Predicted lengths differ across comparison cells")
        first_lengths = lengths
        if cell["steps"] in paired:
            require_same_schedule(paired[cell["steps"]], records)
        paired[cell["steps"]] = records
        metrics = {}
        for name, metric in (("bleu", BLEU(lowercase=True, tokenize="13a")), ("chrf", CHRF())):
            value = metric.corpus_score([r["hypothesis"] for r in records], [references]).score
            if not math.isfinite(summary[name]) or abs(summary[name] - value) > 1e-10:
                raise ValueError("Saved quality score cannot be reproduced")
            metrics[name] = value
        scores.append(dict(cell=cell, **metrics, diagnostics=diagnostics))
        hashes[str(path.relative_to(root))] = file_sha256(path)
    return scores, hashes, first_lengths


def run_select(args):
    common, rows = common_assets(args), rows_for(args)
    manifest = read(args.candidates)
    if manifest.get("complete") is not True or manifest.get("kind") != args.stage:
        raise ValueError("Completed candidate manifest required")
    candidates = evaluation.validate_candidates(manifest)
    manifest_sha = file_sha256(args.candidates)
    evaluated, hashes, first_lengths = {}, {}, None
    for candidate in candidates:
        _, _, saved = evaluation.load_heads(Path(candidate["head"]), args.stage, common, args.global_head)
        if evaluation.candidate_entry(Path(candidate["head"]), saved) != candidate:
            raise ValueError("Candidate checkpoint changed")
        scored, outputs, lengths = verified_scores(args.input, args.stage, "selection", common, rows, candidate)
        if first_lengths is not None and first_lengths != lengths:
            raise ValueError("Length predictions differ across candidates")
        first_lengths = lengths
        evaluated[candidate["name"]] = scored
        hashes.update(outputs)
    winner, ranking = evaluation.choose_checkpoint(candidates, evaluated)
    document = dict(format=SELECTION_FORMAT, split="dev", stage=args.stage, common=common,
        examples=len(rows), manifest=manifest, manifest_sha256=manifest_sha,
        evaluated=evaluated, selected=winner, ranking=ranking, outputs=hashes,
        policy="Mean BLEU over joint and own-marginal at 2 and 4 steps, including zero.")
    if (common_assets(args) != common or file_sha256(args.candidates) != manifest_sha
            or any(file_sha256(Path(c["head"])) != c["sha256"] for c in candidates)):
        raise ValueError("Selection inputs changed")
    immutable_json(args.output, document)


def run_readout(args):
    common, rows = common_assets(args), rows_for(args)
    candidate, selected_sha = None, None
    if args.stage in ("global", "contextual"):
        candidate = validate_selected(read(args.residual_selection), common, args.stage)
        selected_sha = file_sha256(args.residual_selection)
        _, _, saved = evaluation.load_heads(Path(candidate["head"]), args.stage, common, args.global_head)
        if evaluation.candidate_entry(Path(candidate["head"]), saved) != candidate:
            raise ValueError("Selected checkpoint changed")
    elif args.residual_selection:
        raise ValueError("Baseline/count do not consume a residual selection")
    scored, hashes, _ = verified_scores(args.input, args.stage, "full", common, rows, candidate, selected_sha)
    steps = {}
    for t in (2, 4, 8, 32):
        methods = {s["cell"]["method"]: s for s in scored if s["cell"]["steps"] == t}
        steps[str(t)] = dict(methods=methods)
        if args.stage != "baseline":
            steps[str(t)]["joint_minus_own_marginal"] = {
                m: methods["joint"][m] - methods["own-marginal"][m] for m in ("bleu", "chrf")}
    if (common_assets(args) != common
            or (selected_sha and file_sha256(args.residual_selection) != selected_sha)
            or (candidate and file_sha256(Path(candidate["head"])) != candidate["sha256"])):
        raise ValueError("Readout inputs changed")
    immutable_json(args.output, dict(split="dev", stage=args.stage, examples=len(rows),
        common=common, selected=candidate, residual_selection_sha256=selected_sha,
        steps=steps, outputs=hashes, source_sha256=source_hashes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "score", "select", "readout"):
        sub = commands.add_parser(command)
        for name in ("trained", "data", "length-head", "output"):
            sub.add_argument("--" + name, type=Path, required=True)
        if command == "freeze":
            sub.add_argument("--source-selection", type=Path, required=True)
            continue
        sub.add_argument("--protocol", type=Path, required=True)
        sub.add_argument("--stage", choices=STAGES if command != "select" else STAGES[2:], required=True)
        sub.add_argument("--global-head", type=Path)
        if command == "score":
            for name in ("checkpoint", "tokenizer"):
                sub.add_argument("--" + name, type=Path, required=True)
            sub.add_argument("--head", type=Path)
            sub.add_argument("--residual-selection", type=Path)
            sub.add_argument("--population", choices=("full", "selection"), default="full")
            sub.add_argument("--cell-index", type=int, required=True)
        else:
            sub.add_argument("--input", type=Path, required=True)
            sub.add_argument("--candidates" if command == "select" else "--residual-selection", type=Path,
                             required=command == "select")
            sub.set_defaults(population="selection" if command == "select" else "full")
    args = parser.parse_args()
    {"freeze": run_freeze, "score": run_score, "select": run_select, "readout": run_readout}[args.command](args)


if __name__ == "__main__":
    main()
