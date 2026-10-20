"""Paired development readout for candidate-support and fitted-length controls."""
import argparse
import math
from pathlib import Path
import statistics

from .backbone import file_sha256
from .iwslt_confirm import atomic_json, ordered_rows, read
from .iwslt_controls import (
    control_cells, control_source_hashes, head_fingerprint, validate_selection)
from .iwslt_followup import METHODS, followup_cells, output_name, stratified_rows, verify_output
from .iwslt_followup_readout import diagnostics


def verify_population(records, rows, references, steps):
    if [r["reference"] for r in records] != references:
        raise ValueError("Reference texts differ from development set")
    for record, row in zip(records, rows):
        if record.get("length_mode", "predicted") != "predicted":
            raise ValueError("Reference length leaked into predicted-length control")
        if record["scheduled_steps"] != steps:
            raise ValueError("Observed step count differs from control")
        if not 1 <= record["predicted_length"] <= 1024 - len(row["source_ids"]) - 2:
            raise ValueError("Predicted length outside source context capacity")
    return diagnostics(records)


def require_same_schedule(left, right):
    if [r["predicted_length"] for r in left] != [r["predicted_length"] for r in right]:
        raise ValueError("Paired methods differ in predicted lengths")
    if [r["reveal_counts"] for r in left] != [r["reveal_counts"] for r in right]:
        raise ValueError("Paired methods differ in reveal schedule")


def summarize(cells):
    """Retain individual fixed-seed scores; seeds are not extra test examples."""
    result = {}
    for method in METHODS:
        group = sorted((c for c in cells if c["method"] == method), key=lambda c: c["seed"])
        result[method] = {
            metric: dict(mean=statistics.mean(c[metric] for c in group),
                         by_seed=[dict(seed=c["seed"], score=c[metric]) for c in group])
            for metric in ("bleu", "chrf")}
    return result


def differences(left, right):
    result = {}
    for metric in ("bleu", "chrf"):
        a, b = left[metric], right[metric]
        if [r["seed"] for r in a["by_seed"]] != [r["seed"] for r in b["by_seed"]]:
            raise ValueError("Paired metric seeds differ")
        result[metric] = dict(mean=a["mean"] - b["mean"],
            by_seed=[dict(seed=x["seed"], delta=x["score"] - y["score"])
                     for x, y in zip(a["by_seed"], b["by_seed"])])
    return result


def contrasts(methods):
    return {a + "_minus_" + b: differences(methods[a], methods[b])
            for a, b in (("joint", "own-marginal"), ("joint", "baseline"),
                         ("own-marginal", "baseline"))}


def length_errors(records, rows):
    errors = [r["predicted_length"] - len(row["target_ids"]) - 1
              for r, row in zip(records, rows)]
    return dict(examples=len(errors), mean_error=statistics.mean(errors),
        mean_absolute_error=statistics.mean(abs(e) for e in errors),
        exact_fraction=statistics.mean(e == 0 for e in errors),
        within_one_fraction=statistics.mean(abs(e) <= 1 for e in errors),
        under_fraction=statistics.mean(e < 0 for e in errors),
        over_fraction=statistics.mean(e > 0 for e in errors))


def verify_grid(folder, cells):
    expected = {output_name(c) + ".json" for c in cells}
    observed = {p.name for p in folder.glob("T*.json") if not p.stem.endswith("-identity")}
    if observed != expected:
        raise ValueError("Incomplete or unexpected control grid")
    progress = read(folder / "summary.json")
    if progress["completed"] != len(cells) or progress["expected"] != len(cells):
        raise ValueError("Incomplete control stage")
    if "cells" in progress:
        saved = [{k: c[k] for k in cells[0]} for c in progress["cells"]]
        if saved != cells:
            raise ValueError("Summary grid differs from expected controls")


def run(args):
    import torch
    from sacrebleu.metrics import BLEU, CHRF
    from sacremoses import MosesDetokenizer
    if args.experiment not in ("p1", "new-length"):
        raise ValueError("Unknown control experiment")
    if (args.experiment == "new-length") != (args.length_head is not None):
        raise ValueError("Only the new-length readout requires a refitted head")
    selection = read(args.selection)
    selection_sha = file_sha256(args.selection)
    model_sha = file_sha256(args.trained)
    data_sha = file_sha256(args.data / "manifest.json")
    validate_selection(selection, model_sha, data_sha)
    state = torch.load(args.trained, map_location="cpu", weights_only=False)
    old_head_sha = head_fingerprint(state["length_head"])
    del state
    head_artifact_sha = file_sha256(args.length_head) if args.length_head else None
    head_sha = old_head_sha
    if args.length_head:
        from .iwslt_length_refit import load_refit_head
        head = load_refit_head(args.length_head, model_sha, data_sha, "cpu")
        head_sha = head_fingerprint(head.state_dict())
        del head
    source = control_source_hashes(args.experiment)
    rows_all = ordered_rows(args.data, "dev")
    detok = MosesDetokenizer(lang="en")
    result = dict(experiment=args.experiment, split="dev",
        model_sha256=model_sha, data_sha256=data_sha, selection_sha256=selection_sha,
        old_head_sha256=old_head_sha, head_sha256=head_sha,
        head_artifact_sha256=head_artifact_sha, source_sha256=source,
        analysis_source_sha256={
            name: file_sha256(Path(__file__).parent / name) for name in
            ("iwslt_controls_readout.py", "iwslt_followup_readout.py", "iwslt_confirm.py")},
        selected=selection["selected"], steps={}, outputs={})
    for steps in (2, 4) if args.experiment == "p1" else (2, 4, 8, 32):
        stage = "low" if steps in (2, 4) else "length"
        cells = control_cells(args.experiment, steps, selection)
        rows = stratified_rows(rows_all, 1024 if stage == "low" else 512)
        ids = [r["id"] for r in rows]
        refs = [detok.detokenize(r["target"].split()) for r in rows]
        arms, grouped_records = {}, {}
        for arm, folder in (
                ("previous", args.previous / stage / f"step-{steps}"),
                ("control", args.controls / args.experiment / f"step-{steps}")):
            verify_grid(folder, followup_cells(stage, steps, selection)
                        if arm == "previous" else cells)
            summaries, records_by_method = [], {m: [] for m in METHODS}
            for cell in cells:
                path = folder / (output_name(cell) + ".json")
                identity = dict(cell=cell, split="dev", ids=ids, model_sha256=model_sha,
                    data_sha256=data_sha, selection_sha256=selection_sha)
                if arm == "previous":
                    identity.update(stage=stage, source_sha256=selection["source_sha256"])
                else:
                    identity.update(experiment=args.experiment, source_sha256=source,
                        nucleus_p=1. if args.experiment == "p1" else .9, top_k=500,
                        head_sha256=head_sha, old_head_sha256=old_head_sha,
                        head_artifact_sha256=head_artifact_sha)
                payload = verify_output(path, identity)
                records = payload["predictions"]
                diag = verify_population(records, rows, refs, steps)
                scores = {}
                for metric, scorer in (("bleu", BLEU(lowercase=True, tokenize="13a")),
                                        ("chrf", CHRF())):
                    value = scorer.corpus_score([r["hypothesis"] for r in records], [refs]).score
                    if not math.isfinite(payload["summary"][metric]) or abs(
                            value - payload["summary"][metric]) > 1e-10:
                        raise ValueError("Saved generation score cannot be reproduced")
                    scores[metric] = value
                summaries.append(dict(cell, **scores, diagnostics=diag))
                records_by_method[cell["method"]].append(records)
                result["outputs"][str(path)] = file_sha256(path)
            for i in range(len(records_by_method["baseline"])):
                for method in METHODS[1:]:
                    require_same_schedule(records_by_method["baseline"][i],
                                          records_by_method[method][i])
            lengths = [r["predicted_length"] for r in records_by_method["baseline"][0]]
            if any([r["predicted_length"] for r in seed_records] != lengths
                   for seed_records in records_by_method["baseline"][1:]):
                raise ValueError("Length predictions changed across decoding seeds")
            means = summarize(summaries)
            arms[arm] = dict(cells=summaries, means=means, contrasts=contrasts(means),
                length_errors=length_errors(records_by_method["baseline"][0], rows))
            grouped_records[arm] = records_by_method
        if args.experiment == "p1":
            for method in METHODS:
                for old, new in zip(grouped_records["previous"][method],
                                    grouped_records["control"][method]):
                    require_same_schedule(old, new)
        old_lengths = [r["predicted_length"] for r in grouped_records["previous"]["baseline"][0]]
        new_lengths = [r["predicted_length"] for r in grouped_records["control"]["baseline"][0]]
        arms["changed_length_fraction"] = statistics.mean(
            a != b for a, b in zip(old_lengths, new_lengths))
        arms["control_minus_previous"] = {
            m: differences(arms["control"]["means"][m], arms["previous"]["means"][m])
            for m in METHODS}
        result["steps"][str(steps)] = arms
        print(f"Verified {args.experiment}, {steps} steps", flush=True)
    if (file_sha256(args.selection) != selection_sha or
            file_sha256(args.trained) != model_sha or
            file_sha256(args.data / "manifest.json") != data_sha or
            (args.length_head and file_sha256(args.length_head) != head_artifact_sha)):
        raise ValueError("Inputs changed during readout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "selection", "trained", "previous", "controls", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--length-head", type=Path)
    parser.add_argument("--experiment", choices=("p1", "new-length"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
