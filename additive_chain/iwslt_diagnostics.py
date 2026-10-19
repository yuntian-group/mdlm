"""Inspect saved development translations without another model evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def diagnostics(predictions, data):
    """Compare predicted token budgets with references and flag output pathologies.

Reference lengths include EOS, just like the predicted length. They are used
only here for analysis, never supplied to the translation model.
"""
    lookup = {row["id"]: row for row in data}
    if len(lookup) != len(data) or len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("Duplicate example IDs")
    if not predictions:
        raise ValueError("No predictions")
    records = []
    for row in predictions:
        original = lookup[row["id"]]
        predicted, reference = row["predicted_length"], len(original["target_ids"]) + 1
        if not 1 <= predicted or len(row["token_ids"]) > predicted:
            raise ValueError("Invalid generated token length")
        words = row["hypothesis"].lower().split()
        ngrams = Counter(tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3)))
        repeated = sum(count - 1 for count in ngrams.values())
        records.append(dict(id=row["id"], source=original["source"], reference=row["reference"],
            hypothesis=row["hypothesis"], predicted_tokens=predicted, reference_tokens=reference,
            length_error=predicted - reference,
            ended_before_budget=len(row["token_ids"]) + 1 < predicted,
            no_eos=len(row["token_ids"]) == predicted,
            repeated_4grams=repeated, total_4grams=sum(ngrams.values()),
            exact_source_copy=row["hypothesis"].strip().lower() == original["source"].strip().lower()))
    summary = dict(examples=len(records),
        mean_predicted_tokens=statistics.mean(r["predicted_tokens"] for r in records),
        mean_reference_tokens=statistics.mean(r["reference_tokens"] for r in records),
        mean_signed_length_error=statistics.mean(r["length_error"] for r in records),
        mean_absolute_length_error=statistics.mean(abs(r["length_error"]) for r in records),
        fraction_within_5_tokens=statistics.mean(abs(r["length_error"]) <= 5 for r in records),
        early_eos=sum(r["ended_before_budget"] for r in records),
        no_eos=sum(r["no_eos"] for r in records),
        repeated_4gram_fraction=sum(r["repeated_4grams"] for r in records) /
            max(1, sum(r["total_4grams"] for r in records)),
        exact_source_copies=sum(r["exact_source_copy"] for r in records))
    return dict(summary=summary, examples=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text())["predictions"]
    with args.data.open() as stream:
        rows = [json.loads(line) for line in stream]
    result = diagnostics(predictions, rows)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
