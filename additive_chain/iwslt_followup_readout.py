"""Paired development readout for calibration, low-step and oracle diagnostics."""
import argparse
import math
from pathlib import Path

import numpy as np

from .backbone import file_sha256
from .iwslt_confirm import atomic_json, ordered_rows, read
from .iwslt_followup import (
    METHODS, followup_cells, output_name, select_settings, source_hashes,
    stratified_rows, verify_output)
from .iwslt_test_readout import paired_seed_metric


def contrasts(groups, replicates):
    from sacrebleu.metrics import BLEU, CHRF
    result = {}
    for a, b in (("joint", "own-marginal"), ("joint", "baseline"), ("own-marginal", "baseline")):
        result[a + "_minus_" + b] = {
            name: paired_seed_metric(groups[a], groups[b], metric, replicates=replicates)
            for name, metric in (("bleu", BLEU(lowercase=True, tokenize="13a")), ("chrf", CHRF()))}
    return result


def diagnostics(predictions):
    if not predictions:
        raise ValueError("Nonempty decoding diagnostics required")
    for record in predictions:
        states = record["opportunities"]
        counts = record["reveal_counts"]
        remaining = record["predicted_length"]
        if (remaining < 1 or any(n < 0 for n in counts)
                or sum(counts) != remaining
                or [s["revealed"] for s in states] != [n for n in counts if n]
                or len(states) != record["actual_forward_calls"]):
            raise ValueError("Diagnostics differ from the decoding schedule")
        for state in states:
            if (state["masked"] != remaining
                    or any(n < 0 for n in state["component_reveals"])
                    or sum(state["component_reveals"]) != state["revealed"]):
                raise ValueError("Component diagnostics differ from the decoding schedule")
            remaining -= state["revealed"]
    states = [s for r in predictions for s in r["opportunities"]]
    pairs = [p for s in states for p in s.get("same_state_pairs", [])]
    revealed = sum(s["revealed"] for s in states)
    return dict(examples=len(predictions), nonempty_updates=len(states),
        singleton_update_fraction=sum(s["revealed"] == 1 for s in states)/len(states),
        tokens_revealed_with_another_in_same_component_fraction=sum(
            n for s in states for n in s["component_reveals"] if n >= 2)/revealed,
        mean_denoising_model_calls=np.mean([r["actual_forward_calls"] for r in predictions]).item(),
        mean_total_model_calls=np.mean([r["actual_forward_calls"] + r["length_prediction_forwards"]
                                       for r in predictions]).item(),
        mean_length=np.mean([r["predicted_length"] for r in predictions]).item(),
        same_state_pair_count=len(pairs),
        same_state_pair_mean_mi=np.mean([p["mutual_information_nats"] for p in pairs]).item() if pairs else None,
        same_state_pair_mean_tv=np.mean([p["total_variation"] for p in pairs]).item() if pairs else None,
        seconds_including_diagnostics=sum(r["seconds"] for r in predictions),
        same_state_probe_seconds=sum(r["same_state_probe_seconds"] for r in predictions))


def span_summary(default, wide, replicates):
    if not default or replicates < 1:
        raise ValueError("Nonempty spans and positive replicate count required")
    if [(r["id"], r["start"]) for r in default] != [(r["id"], r["start"]) for r in wide]:
        raise ValueError("Span populations differ")
    if any(a["gold_tokens"] != b["gold_tokens"] or
           a["gold_full_vocab_ranks"] != b["gold_full_vocab_ranks"] or
           a["gold_full_vocab_log_probabilities"] != b["gold_full_vocab_log_probabilities"]
           for a, b in zip(default, wide)):
        raise ValueError("Span targets or frozen model outputs differ")
    result = {}
    for name, records in (("default", default), ("wide", wide)):
        for record in records:
            probabilities = [record["joint_gold_probability"], record["independent_gold_probability"]]
            if (any(not math.isfinite(p) or p < 0 or p > 1 + 1e-10 for p in probabilities)
                    or (not record["supported"] and any(p != 0 for p in probabilities))):
                raise ValueError("Invalid span probability or unsupported nonzero probability")
        j = np.array([r["joint_gold_probability"] for r in records])
        m = np.array([r["independent_gold_probability"] for r in records])
        rng = np.random.default_rng(1718)
        means = [float((j-m)[rng.integers(len(j), size=len(j))].mean()) for _ in range(replicates)]
        ranks = [rank for r in records for rank in r["gold_full_vocab_ranks"]]
        result[name] = dict(examples=len(records), supported=sum(r["supported"] for r in records),
            mean_joint_gold_probability=float(j.mean()), mean_independent_gold_probability=float(m.mean()),
            joint_minus_independent=float((j-m).mean()), interval_95=np.quantile(means, [.025, .975]).tolist(),
            token_coverage_top500=sum(r <= 500 for r in ranks)/len(ranks),
            token_coverage_top2048=sum(r <= 2048 for r in ranks)/len(ranks))
    return result


def run(args):
    from sacrebleu.metrics import BLEU, CHRF
    from sacremoses import MosesDetokenizer
    from .iwslt_spans import span_population
    from .iwslt_run import dataset
    if args.replicates < 1:
        raise ValueError("Positive replicate count required")
    selection = read(args.selection)
    if selection["source_sha256"] != source_hashes() or select_settings(selection["cells"]) != selection["selected"]:
        raise ValueError("Changed development selection/code")
    if file_sha256(args.data / "manifest.json") != selection["data_sha256"]:
        raise ValueError("Changed data")
    selection_sha = file_sha256(args.selection)
    all_rows = ordered_rows(args.data, "dev")
    detok = MosesDetokenizer(lang="en")
    result = dict(population="Development-only diagnostics; no test tuning or new test claims.",
        selection_sha256=selection_sha, model_sha256=selection["model_sha256"],
        data_sha256=selection["data_sha256"], calibration=selection["cells"],
        analysis_source_sha256={p: file_sha256(Path(__file__).parent / p) for p in
            ("iwslt_followup_readout.py", "iwslt_test_readout.py", "iwslt_confirm.py")},
        selected=selection["selected"], outputs={}, low={}, length={}, spans={})
    # Recheck archived calibration outputs as well as the newly completed grid.
    for path, expected in selection["development_artifacts"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError("Calibration artifact changed")
    for stage, steps_list, n in (("low", (2, 4), 1024), ("length", (8, 32), 512)):
        rows = stratified_rows(all_rows, n)
        ids = [r["id"] for r in rows]
        refs = [detok.detokenize(r["target"].split()) for r in rows]
        for steps in steps_list:
            groups, summaries = {}, []
            cells = followup_cells(stage, steps, selection)
            root = args.input / stage / f"step-{steps}"
            progress = read(root / "summary.json")
            if progress["completed"] != len(cells) or progress["expected"] != len(cells):
                raise ValueError("Incomplete development stage")
            for cell in cells:
                path = root / (output_name(cell)+".json")
                identity = dict(cell=cell, stage=stage, split="dev", ids=ids,
                    model_sha256=selection["model_sha256"], data_sha256=selection["data_sha256"],
                    source_sha256=selection["source_sha256"], selection_sha256=selection_sha)
                payload = verify_output(path, identity)
                records = payload["predictions"]
                if [r["reference"] for r in records] != refs:
                    raise ValueError("Reference texts differ from development set")
                if cell["length_mode"] == "reference-diagnostic":
                    if any(r.get("length_mode") != "reference-diagnostic" or
                           r["predicted_length"] != len(row["target_ids"])+1
                           for row, r in zip(rows, records)):
                        raise ValueError("Reference length diagnostic was not delivered")
                elif any(r.get("length_mode") == "reference-diagnostic" for r in records):
                    raise ValueError("Reference length leaked into predicted-length decoding")
                for name, metric in (("bleu", BLEU(lowercase=True, tokenize="13a")), ("chrf", CHRF())):
                    value = metric.corpus_score([r["hypothesis"] for r in records], [refs]).score
                    saved = payload["summary"][name]
                    if not math.isfinite(saved) or abs(value-saved) > 1e-10:
                        raise ValueError("Score cannot be reproduced")
                groups.setdefault(cell["length_mode"], {}).setdefault(cell["method"], []).append(records)
                summaries.append(dict(cell, **payload["summary"], diagnostics=diagnostics(records)))
                result["outputs"][str(path.relative_to(args.input))] = file_sha256(path)
            for mode, methods in groups.items():
                for i in range(len(methods["baseline"])):
                    lengths = [[r["predicted_length"] for r in methods[m][i]] for m in METHODS]
                    if any(x != lengths[0] for x in lengths[1:]):
                        raise ValueError("Methods differ in length")
                    masks = [[r["reveal_counts"] for r in methods[m][i]] for m in METHODS]
                    if any(x != masks[0] for x in masks[1:]):
                        raise ValueError("Methods differ in reveal schedule")
            entry = dict(cells=summaries, contrasts={mode: contrasts(g, args.replicates) for mode, g in groups.items()})
            if stage == "length":
                entry["reference_minus_predicted"] = {m: {
                    name: paired_seed_metric(groups["reference-diagnostic"][m], groups["predicted"][m],
                        metric, replicates=args.replicates)
                    for name, metric in (("bleu", BLEU(lowercase=True, tokenize="13a")), ("chrf", CHRF()))
                } for m in METHODS}
            result[stage][str(steps)] = entry
            print(f"Verified {stage}, {steps} steps", flush=True)
    dev = dataset(args.data, "dev")
    span_hashes = {p: file_sha256(Path(__file__).parent / p) for p in
        ("iwslt_spans.py", "translation.py", "generation.py", "chain.py", "counts.py", "candidates.py")}
    for length in (2, 4, 8):
        selected, eligible = span_population(dev, length, 1000, 101)
        expected = [dict(id=r["id"], start=start) for r, start in selected]
        both = []
        for label, k, p in (("default", 500, .9), ("wide", 2048, 1.)):
            path = args.input / ("spans-"+label) / f"span-{length}.json"
            payload = read(path)
            identity = payload["identity"]
            setting = selection["selected"]["8"]
            expected_identity = dict(model_sha256=selection["model_sha256"], data_sha256=selection["data_sha256"],
                length=length, eligible=eligible, selected=expected, strength=setting["strength"],
                temperature=setting["crf_temperature"], seed=101, top_k=k, top_p=p,
                source_sha256=span_hashes,
                protocol="Same masked target span, visible gold context; unchanged model-derived candidates; unsupported gold has zero probability")
            if identity != expected_identity:
                raise ValueError("Span protocol changed")
            records = payload["records"]
            if [dict(id=r["id"], start=r["start"]) for r in records] != expected:
                raise ValueError("Incomplete span outputs")
            both.append(records)
            result["outputs"][str(path.relative_to(args.input))] = file_sha256(path)
        result["spans"][str(length)] = span_summary(*both, args.replicates)
    if file_sha256(args.selection) != selection_sha:
        raise ValueError("Selection changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "data", "selection", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
