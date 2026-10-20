"""Development-only checkpoint selection and paired additive-residual decoding."""
import argparse
import math
from pathlib import Path
import time

from .backbone import file_sha256
from .iwslt_confirm import atomic_json, ordered_rows, read
from .iwslt_controls import head_fingerprint
from .iwslt_controls_readout import length_errors, require_same_schedule, verify_population
from .iwslt_followup import output_name, stratified_rows, verify_output
from .iwslt_full_dev import check_fixed_selection

METHODS = ("joint", "own-marginal")
FORMAT = "iwslt_residual_selection_v1"


def cells(population):
    if population not in ("selection", "full"):
        raise ValueError("Unknown development population")
    steps = (2, 4) if population == "selection" else (32, 8, 4, 2)
    return [dict(steps=t, temperature=.85, strength=.2, sampling=method,
                 seed=101, method=method, length_mode="predicted")
            for t in steps for method in METHODS]


def source_hashes():
    from .iwslt_residual_assets import residual_source_hashes
    result = residual_source_hashes()
    for name in ("iwslt_residual_eval.py", "iwslt_followup.py", "generation.py",
                 "iwslt_controls_readout.py", "iwslt_full_dev.py", "iwslt_length_refit.py",
                 "cuda.py", "_additive_kernels.py", "_count_inference.py",
                 "_count_kernels.py"):
        result[name] = file_sha256(Path(__file__).parent / name)
    return result


def development_rows(data, population):
    cells(population)
    rows = ordered_rows(data, "dev")
    return stratified_rows(rows, 1024) if population == "selection" else rows


def valid_checkpoint_name(name, epoch, update):
    if type(epoch) is not int or epoch < 0 or type(update) is not int or update < 0:
        return False
    if name == f"epoch-{epoch:04d}":
        return epoch != 0 or update == 0
    return update > 0 and name == f"best-loss-{update:07d}"


def candidate_entry(path, saved):
    if not valid_checkpoint_name(saved["checkpoint_id"], saved["epoch"], saved["update"]):
        raise ValueError("Checkpoint name and epoch/update differ")
    return dict(name=saved["checkpoint_id"], head=str(Path(path).resolve()),
                sha256=file_sha256(path), epoch=saved["epoch"], update=saved["update"])


def validate_candidates(manifest):
    candidates = manifest.get("candidates")
    if manifest.get("kind") not in ("global", "contextual") or not isinstance(candidates, list) or not candidates:
        raise ValueError("Invalid checkpoint manifest")
    if any(set(c) != {"name", "head", "sha256", "epoch", "update"} or
           not valid_checkpoint_name(c["name"], c["epoch"], c["update"]) or
           not Path(c["head"]).is_absolute() or
           len(c["sha256"]) != 64 or any(ch not in "0123456789abcdef" for ch in c["sha256"])
           for c in candidates):
        raise ValueError("Invalid checkpoint candidate")
    if len({c["name"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate checkpoint candidate")
    if "epoch-0000" not in [c["name"] for c in candidates]:
        raise ValueError("Checkpoint selection must include the zero residual")
    return sorted(candidates, key=lambda c: (c["epoch"], c["update"], c["name"]))


def choose_checkpoint(candidates, evaluated):
    """Choose one checkpoint for both samplers; no separate strength tuning."""
    expected = cells("selection")
    if set(evaluated) != {c["name"] for c in candidates}:
        raise ValueError("Incomplete checkpoint selection")
    ranking = []
    for candidate in candidates:
        scores = evaluated[candidate["name"]]
        if (len(scores) != len(expected) or
                {output_name(s["cell"]) for s in scores} != {output_name(c) for c in expected}):
            raise ValueError("Incomplete or duplicate selection grid")
        for row in scores:
            if row["cell"] not in expected or any(
                    not math.isfinite(row[metric]) for metric in ("bleu", "chrf")):
                raise ValueError("Invalid development score")
        ranking.append(dict(candidate=candidate,
                            mean_bleu=sum(row["bleu"] for row in scores) / 4,
                            cells=scores))
    ranking.sort(key=lambda r: (-r["mean_bleu"], r["candidate"]["epoch"],
                               r["candidate"]["update"], r["candidate"]["name"]))
    return ranking[0]["candidate"], ranking


def validate_selected(document, common, kind):
    if (document.get("format") != FORMAT or document.get("split") != "dev" or
            document.get("kind") != kind or document.get("common") != common):
        raise ValueError("Checkpoint selection protocol or assets changed")
    candidates = validate_candidates(document["manifest"])
    if document["manifest"]["kind"] != kind:
        raise ValueError("Selection manifest stage differs from requested stage")
    winner, ranking = choose_checkpoint(candidates, document["evaluated"])
    if document.get("selected") != winner or document.get("ranking") != ranking:
        raise ValueError("Checkpoint selection does not follow shared development BLEU")
    return winner


def common_assets(args):
    from .iwslt_length_refit import load_refit_head
    model_sha = file_sha256(args.trained)
    data_sha = file_sha256(args.data / "manifest.json")
    check_fixed_selection(read(args.selection), model_sha, data_sha)
    counts_sha = file_sha256(args.data / "counts.pt")
    if counts_sha != read(args.data / "manifest.json")["counts_sha256"]:
        raise ValueError("Training counts changed")
    head = load_refit_head(args.length_head, model_sha, data_sha, "cpu")
    return dict(model_sha256=model_sha, data_sha256=data_sha,
                counts_sha256=counts_sha, selection_sha256=file_sha256(args.selection),
                length_head_sha256=head_fingerprint(head.state_dict()),
                length_head_artifact_sha256=file_sha256(args.length_head),
                inherited_global_sha256=file_sha256(args.global_head) if args.global_head else None,
                nucleus_p=.9, top_k=500, temperature=.85, count_strength=.2,
                residual_strength=1., seed=101)


def load_heads(path, kind, common, global_path=None, device="cpu"):
    from .iwslt_residual_assets import load_residual_head
    head, saved = load_residual_head(path, common["model_sha256"], common["data_sha256"], device)
    if saved["kind"] != kind:
        raise ValueError("Residual kind differs from requested stage")
    identity = saved["identity"]
    if identity["counts_sha256"] != common["counts_sha256"]:
        raise ValueError("Residual counts changed")
    if identity.get("inherited_global_sha256") != common["inherited_global_sha256"]:
        raise ValueError("Inherited global residual changed")
    inherited = None
    if global_path:
        if kind != "contextual":
            raise ValueError("Only contextual residuals can inherit a global head")
        inherited, base = load_residual_head(global_path, common["model_sha256"],
                                             common["data_sha256"], device)
        if base["kind"] != "global" or base["identity"]["counts_sha256"] != common["counts_sha256"]:
            raise ValueError("Invalid inherited global head")
    return head, inherited, saved


def build_identity(cell, rows, population, kind, candidate, common, selection_sha=None):
    if cell not in cells(population):
        raise ValueError("Unregistered residual decoding cell")
    return dict(experiment="iwslt-learned-residual", split="dev", kind=kind,
                population=population, candidate=candidate, common=common,
                cell=cell, ids=[r["id"] for r in rows],
                residual_selection_sha256=selection_sha, source_sha256=source_hashes())


def score_translations(model, tokenizer, length_head, rows, decoder, cell):
    """Keep schedule diagnostics, but do not use the count-only MI probe."""
    from sacrebleu.metrics import BLEU, CHRF
    from sacremoses import MosesDetokenizer
    from .generation import synchronize
    from .iwslt_followup import StateObserver
    from .iwslt_run import log
    from .translation import translate_one
    detok, records = MosesDetokenizer(lang="en"), []
    for index, row in enumerate(rows):
        observer = StateObserver(decoder, probe=False)
        synchronize(decoder.device)
        start = time.perf_counter()
        ids, stats = translate_one(model, row, length_head, decoder, steps=cell["steps"],
            seed=cell["seed"], example_id=index, observer=observer)
        synchronize(decoder.device)
        records.append(dict(id=row["id"], hypothesis=detok.detokenize(tokenizer.decode(ids).split()),
            reference=detok.detokenize(row["target"].split()), token_ids=ids,
            seconds=time.perf_counter() - start, same_state_probe_seconds=0.,
            **stats, opportunities=observer.steps))
        if (index + 1) % 32 == 0:
            log("residual_eval_progress", completed=index + 1, total=len(rows), **cell)
    hypotheses = [r["hypothesis"] for r in records]
    references = [[r["reference"] for r in records]]
    summary = dict(examples=len(records),
        bleu=BLEU(lowercase=True, tokenize="13a").corpus_score(hypotheses, references).score,
        chrf=CHRF().corpus_score(hypotheses, references).score,
        seconds=sum(r["seconds"] for r in records), same_state_probe_seconds=0.,
        timing_includes_diagnostics=True, same_state_probability_probes=False)
    return dict(summary=summary, predictions=records)


def run_score(args):
    import torch
    from .generation import ChainDecoder
    from .iwslt_length_refit import load_refit_head
    from .iwslt_run import log, run_lock
    from .translation import MASK_ID, load_conditional
    registered = cells(args.population)
    if not 0 <= args.cell_index < len(registered):
        raise ValueError("Unregistered residual cell index")
    cell = registered[args.cell_index]
    common = common_assets(args)
    head, inherited, saved = load_heads(args.head, args.kind, common, args.global_head)
    candidate = candidate_entry(args.head, saved)
    if candidate["name"] != args.candidate_name:
        raise ValueError("Checkpoint name differs from requested output")
    selection_sha = None
    if args.population == "full":
        if not args.residual_selection:
            raise ValueError("Full development requires a frozen selected checkpoint")
        winner = validate_selected(read(args.residual_selection), common, args.kind)
        if winner != candidate:
            raise ValueError("Full development must use the selected checkpoint")
        selection_sha = file_sha256(args.residual_selection)
    elif args.residual_selection:
        raise ValueError("Checkpoint scoring must not consume a winner")
    rows = development_rows(args.data, args.population)
    identity = build_identity(cell, rows, args.population, args.kind, candidate, common, selection_sha)
    folder = args.output / candidate["name"] / args.population / output_name(cell)
    with run_lock(folder):
        path = folder / "result.json"
        if (folder / "result-identity.json").exists():
            result = verify_output(path, identity)
        else:
            if path.exists():
                raise ValueError("Output without completion receipt")
            model, tokenizer = load_conditional(args.checkpoint, args.tokenizer)
            state = torch.load(args.trained, map_location="cuda", weights_only=False)
            model.load_state_dict(state["model"])
            del state
            model.eval().requires_grad_(False)
            length_head = load_refit_head(args.length_head, common["model_sha256"],
                                          common["data_sha256"], "cuda")
            head = head.cuda()
            if inherited is not None:
                inherited = inherited.cuda()
            decoder = ChainDecoder(mask_id=MASK_ID, device="cuda",
                count_statistics=torch.load(args.data / "counts.pt", weights_only=True),
                count_strength=.2, top_k=500, nucleus_p=.9, fast_candidates=True,
                sampling=cell["sampling"], temperature=.85,
                global_head=head if args.kind == "global" else inherited,
                global_strength=1. if args.kind == "global" or inherited is not None else 0.,
                contextual_head=head if args.kind == "contextual" else None,
                contextual_strength=1. if args.kind == "contextual" else 0.)
            log("residual_eval_started", kind=args.kind, candidate=candidate["name"],
                population=args.population, examples=len(rows), **cell)
            decoder.prepare()
            try:
                result = score_translations(model, tokenizer, length_head, rows, decoder, cell)
            finally:
                decoder.close()
            if (common_assets(args) != common or file_sha256(args.head) != candidate["sha256"] or
                    (selection_sha and file_sha256(args.residual_selection) != selection_sha)):
                raise ValueError("Residual decoding inputs changed")
            atomic_json(path, result)
            atomic_json(folder / "result-identity.json",
                        dict(identity=identity, output_sha256=file_sha256(path)))
        atomic_json(folder / "summary.json",
                    dict(completed=1, expected=1, cell=cell, **result["summary"]))
        log("residual_eval_complete", kind=args.kind, candidate=candidate["name"], **cell,
            **result["summary"])


def verified_scores(root, candidate, kind, common, rows, population, selection_sha=None):
    from sacrebleu.metrics import BLEU, CHRF
    from sacremoses import MosesDetokenizer
    expected = cells(population)
    folder = root / candidate["name"] / population
    observed = {p.name for p in folder.glob("T*") if p.is_dir()}
    if observed != {output_name(c) for c in expected}:
        raise ValueError("Incomplete or unexpected residual evaluation grid")
    detok = MosesDetokenizer(lang="en")
    references = [detok.detokenize(r["target"].split()) for r in rows]
    scores, records_by_steps, outputs, first = [], {}, {}, None
    for cell in expected:
        path = folder / output_name(cell) / "result.json"
        summary = read(path.with_name("summary.json"))
        if summary["cell"] != cell or (summary["completed"], summary["expected"]) != (1, 1):
            raise ValueError("Incomplete residual decoding summary")
        identity = build_identity(cell, rows, population, kind, candidate, common, selection_sha)
        payload = verify_output(path, identity)
        records = payload["predictions"]
        diagnostics = verify_population(records, rows, references, cell["steps"])
        if any(len(r["reveal_counts"]) != cell["steps"] or
               r["length_prediction_forwards"] != 1 for r in records):
            raise ValueError("Unexpected steps or length prediction forwards")
        if first is None:
            first = records
        elif [r["predicted_length"] for r in first] != [r["predicted_length"] for r in records]:
            raise ValueError("Predicted lengths differ across residual cells")
        if cell["steps"] in records_by_steps:
            require_same_schedule(records_by_steps[cell["steps"]], records)
        else:
            records_by_steps[cell["steps"]] = records
        metrics = {}
        for key, metric in (("bleu", BLEU(lowercase=True, tokenize="13a")), ("chrf", CHRF())):
            value = metric.corpus_score([r["hypothesis"] for r in records], [references]).score
            if not math.isfinite(payload["summary"][key]) or abs(value - payload["summary"][key]) > 1e-10:
                raise ValueError("Residual decoding score cannot be reproduced")
            metrics[key] = value
        scores.append(dict(cell=cell, **metrics, diagnostics=diagnostics))
        outputs[str(path.relative_to(root))] = file_sha256(path)
    return scores, outputs, first


def run_select(args):
    common = common_assets(args)
    manifest = read(args.candidates)
    candidates = validate_candidates(manifest)
    if manifest["kind"] != args.kind:
        raise ValueError("Manifest stage differs from requested stage")
    rows = development_rows(args.data, "selection")
    evaluated, outputs, first = {}, {}, None
    manifest_sha = file_sha256(args.candidates)
    for candidate in candidates:
        _, _, saved = load_heads(Path(candidate["head"]), args.kind, common, args.global_head)
        if candidate_entry(candidate["head"], saved) != candidate:
            raise ValueError("Candidate checkpoint metadata or hash changed")
        scored, hashes, records = verified_scores(
            args.input, candidate, args.kind, common, rows, "selection")
        if first is not None and [r["predicted_length"] for r in records] != first:
            raise ValueError("Candidate length predictions differ")
        first = [r["predicted_length"] for r in records]
        evaluated[candidate["name"]] = scored
        outputs.update(hashes)
    winner, ranking = choose_checkpoint(candidates, evaluated)
    result = dict(format=FORMAT, split="dev", kind=args.kind, examples=len(rows),
                  manifest=manifest, manifest_sha256=manifest_sha, common=common,
                  selected=winner, ranking=ranking, evaluated=evaluated, outputs=outputs,
                  policy="Mean BLEU across joint and own-marginal at 2 and 4 steps; ties choose the earlier epoch, then update, then checkpoint name.")
    if (common_assets(args) != common or file_sha256(args.candidates) != manifest_sha or
            any(file_sha256(Path(c["head"])) != c["sha256"] for c in candidates)):
        raise ValueError("Checkpoint selection inputs changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and read(args.output) != result:
        raise ValueError("Refusing to change a frozen residual selection")
    atomic_json(args.output, result)
    print(f"Selected {winner['name']}: mean development BLEU {ranking[0]['mean_bleu']:.6f}", flush=True)


def run_readout(args):
    common = common_assets(args)
    selected = read(args.residual_selection)
    winner = validate_selected(selected, common, args.kind)
    _, _, saved = load_heads(Path(winner["head"]), args.kind, common, args.global_head)
    if candidate_entry(winner["head"], saved) != winner:
        raise ValueError("Selected checkpoint changed")
    selection_sha = file_sha256(args.residual_selection)
    rows = development_rows(args.data, "full")
    scores, hashes, records = verified_scores(
        args.input, winner, args.kind, common, rows, "full", selection_sha)
    steps = {}
    for t in (2, 4, 8, 32):
        methods = {r["cell"]["method"]: r for r in scores if r["cell"]["steps"] == t}
        steps[str(t)] = dict(methods=methods, joint_minus_own_marginal={
            metric: methods["joint"][metric] - methods["own-marginal"][metric]
            for metric in ("bleu", "chrf")})
    result = dict(split="dev", kind=args.kind, examples=len(rows), selected=winner,
                  common=common, residual_selection_sha256=selection_sha,
                  source_sha256=source_hashes(), steps=steps, outputs=hashes,
                  length_errors=length_errors(records, rows))
    if (common_assets(args) != common or file_sha256(args.residual_selection) != selection_sha or
            file_sha256(Path(winner["head"])) != winner["sha256"]):
        raise ValueError("Residual readout inputs changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("score", "select", "readout"):
        sub = subparsers.add_parser(command)
        for name in ("trained", "data", "length-head", "selection", "output"):
            sub.add_argument("--" + name, type=Path, required=True)
        sub.add_argument("--global-head", type=Path)
        sub.add_argument("--kind", choices=("global", "contextual"), required=True)
        if command == "score":
            for name in ("checkpoint", "tokenizer", "head"):
                sub.add_argument("--" + name, type=Path, required=True)
            sub.add_argument("--candidate-name", required=True)
            sub.add_argument("--population", choices=("selection", "full"), required=True)
            sub.add_argument("--cell-index", type=int, required=True)
            sub.add_argument("--residual-selection", type=Path)
        else:
            sub.add_argument("--input", type=Path, required=True)
            sub.add_argument("--candidates" if command == "select" else "--residual-selection",
                             type=Path, required=True)
    args = parser.parse_args()
    {"score": run_score, "select": run_select, "readout": run_readout}[args.command](args)


if __name__ == "__main__":
    main()
