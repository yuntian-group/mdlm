"""Paired development-only quality differences for matched CRF decoders."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_difference(joint, independent, scorer, *, replicates=2000, seed=1718):
    """Resample identical sentence indices for both corpus-level scores."""
    keys = [(r["id"], r["reference"]) for r in joint]
    if not keys or keys != [(r["id"], r["reference"]) for r in independent]:
        raise ValueError("Matched IDs, reference texts and ordering are required")
    if replicates < 1:
        raise ValueError("Positive replicate count required")
    a, b = [[r["hypothesis"] for r in rows] for rows in (joint, independent)]
    references = [r["reference"] for r in joint]
    delta = scorer(a, references) - scorer(b, references)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        indices = rng.integers(0, len(keys), len(keys))
        refs = [references[i] for i in indices]
        values.append(scorer([a[i] for i in indices], refs) - scorer([b[i] for i in indices], refs))
    interval = np.quantile(values, [.025, .975]).tolist()
    return dict(joint_minus_independent=delta, interval_95=interval,
                examples=len(keys), replicates=replicates, seed=seed)


def readout(root, *, replicates=2000):
    from sacrebleu.metrics import BLEU, CHRF
    summary = json.loads((root / "summary.json").read_text())
    if summary["completed"] != summary["expected"]:
        raise ValueError("Comparison is incomplete")
    receipts = sorted(root.glob("*-identity.json"))
    if len(receipts) != summary["expected"]:
        raise ValueError("Completion records are missing or duplicated")
    groups = {}
    for path in receipts:
        identity = json.loads(path.read_text())
        if identity["strength"] == 0:
            continue
        if identity["model_sha256"] != summary["model_sha256"] or identity["data_sha256"] != summary["data_sha256"]:
            raise ValueError("Comparison combines different data or model checkpoints")
        key = (identity["steps"], identity["temperature"], identity["strength"])
        output = path.with_name(path.name.replace("-identity.json", ".json"))
        values = groups.setdefault(key, {})
        if identity["sampling"] in values:
            raise ValueError("Duplicate comparison cell")
        predictions = json.loads(output.read_text())["predictions"]
        if [r["id"] for r in predictions] != identity["development_ids"]:
            raise ValueError("Predictions differ from the registered development inputs")
        values[identity["sampling"]] = (identity, predictions)
    result = []
    for (steps, temperature, strength), group in groups.items():
        if set(group) != {"joint", "own-marginal"}:
            raise ValueError("Missing paired sampler")
        ia, a = group["joint"]
        ib, b = group["own-marginal"]
        if {k: v for k, v in ia.items() if k != "sampling"} != {k: v for k, v in ib.items() if k != "sampling"}:
            raise ValueError("Joint and independent cells do not share the same settings")
        metrics = {}
        for name, metric in (("bleu", BLEU(lowercase=True, tokenize="13a")), ("chrf", CHRF())):
            def scorer(hypotheses, references):
                return metric.corpus_score(hypotheses, [references]).score
            metrics[name] = paired_difference(a, b, scorer, replicates=replicates)
            metrics[name]["signature"] = str(metric.get_signature())
        result.append(dict(steps=steps, temperature=temperature, strength=strength, **metrics))
    if not result:
        raise ValueError("No matched comparisons found")
    return dict(model_sha256=summary["model_sha256"], data_sha256=summary["data_sha256"],
                population="development; exploratory intervals, not a test-set claim", comparisons=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    args = p.parse_args()
    result = readout(args.input, replicates=args.replicates)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
