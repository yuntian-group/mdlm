"""Verified full-development BLEU and chrF for a fixed twelve-cell comparison."""
import argparse
import math
from pathlib import Path

from .backbone import file_sha256
from .iwslt_confirm import atomic_json, ordered_rows, read
from .iwslt_controls import head_fingerprint
from .iwslt_controls_readout import length_errors, require_same_schedule, verify_population
from .iwslt_followup import METHODS, output_name, verify_output
from .iwslt_full_dev import build_identity, check_fixed_selection, full_cells, source_hashes


def verify_grid(root, cells):
    expected = {output_name(cell) for cell in cells}
    observed = {path.name for path in root.glob("T*") if path.is_dir()}
    if observed != expected:
        raise ValueError("Incomplete or unexpected full-development grid")
    for cell in cells:
        folder = root / output_name(cell)
        if not (folder / "result.json").is_file() or not (
                folder / "result-identity.json").is_file():
            raise ValueError("Incomplete full-development cell")
        summary = read(folder / "summary.json")
        if summary["completed"] != 1 or summary["expected"] != 1:
            raise ValueError("Incomplete full-development summary")
        if summary["cell"] != cell:
            raise ValueError("Full-development summary cell differs")


def metric_contrasts(methods):
    return {
        left + "_minus_" + right: {
            metric: methods[left][metric] - methods[right][metric]
            for metric in ("bleu", "chrf")}
        for left, right in (("joint", "own-marginal"), ("joint", "baseline"),
                            ("own-marginal", "baseline"))}


def run(args):
    import torch
    from sacrebleu.metrics import BLEU, CHRF
    from sacremoses import MosesDetokenizer
    from .iwslt_length_refit import load_refit_head
    selection_sha = file_sha256(args.selection)
    model_sha = file_sha256(args.trained)
    data_sha = file_sha256(args.data / "manifest.json")
    head_artifact_sha = file_sha256(args.length_head)
    selection = read(args.selection)
    check_fixed_selection(selection, model_sha, data_sha)
    manifest = read(args.data / "manifest.json")
    counts_sha = file_sha256(args.data / "counts.pt")
    if counts_sha != manifest["counts_sha256"]:
        raise ValueError("Training counts changed")
    saved = torch.load(args.trained, map_location="cpu", weights_only=False)
    old_head_sha = head_fingerprint(saved["length_head"])
    del saved
    head = load_refit_head(args.length_head, model_sha, data_sha, "cpu")
    head_sha = head_fingerprint(head.state_dict())
    del head
    rows = ordered_rows(args.data, "dev")
    detok = MosesDetokenizer(lang="en")
    references = [detok.detokenize(row["target"].split()) for row in rows]
    cells = full_cells()
    verify_grid(args.input, cells)
    result = dict(split="dev", examples=len(rows), seed=101,
        model_sha256=model_sha, data_sha256=data_sha, counts_sha256=counts_sha,
        selection_sha256=selection_sha, old_head_sha256=old_head_sha,
        head_sha256=head_sha, head_artifact_sha256=head_artifact_sha,
        source_sha256=source_hashes(),
        analysis_source_sha256={name: file_sha256(Path(__file__).parent / name)
            for name in ("iwslt_full_dev_readout.py", "iwslt_controls_readout.py",
                         "iwslt_followup_readout.py", "iwslt_confirm.py")},
        steps={}, outputs={})
    methods_by_step, common_lengths, first_records = {}, None, None
    for cell in cells:
        path = args.input / output_name(cell) / "result.json"
        identity = build_identity(cell, rows, model_sha, data_sha, selection_sha,
                                  head_sha, head_artifact_sha, old_head_sha)
        payload = verify_output(path, identity)
        records = payload["predictions"]
        diag = verify_population(records, rows, references, cell["steps"])
        if any(len(record["reveal_counts"]) != cell["steps"] or
               record["length_prediction_forwards"] != 1 for record in records):
            raise ValueError("Unexpected number of reveal steps or length forwards")
        lengths = [record["predicted_length"] for record in records]
        if common_lengths is None:
            common_lengths, first_records = lengths, records
        elif lengths != common_lengths:
            raise ValueError("Predicted lengths differ across full-development cells")
        methods = methods_by_step.setdefault(cell["steps"], {})
        if methods:
            require_same_schedule(next(iter(methods.values()))["records"], records)
        scores = {}
        for metric, scorer in (("bleu", BLEU(lowercase=True, tokenize="13a")),
                               ("chrf", CHRF())):
            value = scorer.corpus_score([record["hypothesis"] for record in records],
                                       [references]).score
            saved_score = payload["summary"][metric]
            if not math.isfinite(saved_score) or abs(value - saved_score) > 1e-10:
                raise ValueError("Full-development score cannot be reproduced")
            scores[metric] = value
        methods[cell["method"]] = dict(records=records, score=dict(cell, **scores, diagnostics=diag))
        result["outputs"][str(path.relative_to(args.input))] = file_sha256(path)
        print(f"Verified {cell['method']}, {cell['steps']} steps, {len(rows)} development examples",
              flush=True)
    for steps, group in methods_by_step.items():
        if set(group) != set(METHODS):
            raise ValueError("Missing full-development method")
        methods = {method: group[method]["score"] for method in METHODS}
        result["steps"][str(steps)] = dict(methods=methods, contrasts=metric_contrasts(methods))
    result["length_errors"] = length_errors(first_records, rows)
    if (file_sha256(args.selection) != selection_sha or file_sha256(args.trained) != model_sha
            or file_sha256(args.data / "manifest.json") != data_sha
            or file_sha256(args.length_head) != head_artifact_sha
            or file_sha256(args.data / "counts.pt") != counts_sha
            or source_hashes() != result["source_sha256"]):
        raise ValueError("Full-development inputs changed during readout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "selection", "trained", "length-head", "input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
