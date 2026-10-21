"""Audit saved IWSLT development translations without loading model weights.

Each --group supplies a label, a full-development readout, and its output root.
Original result.json files and checksum-verified compact .json.gz packages are
supported. Compact packages may lack token IDs; those measurements stay absent.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


EOS_ID, MASK_ID = 50256, 50257
STEPS = (2, 4, 8, 32)
SHARED_SOURCES = ("translation.py", "generation.py", "chain.py", "counts.py",
                  "candidates.py", "fast_candidates.py", "backbone.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def load_output(root, relative, expected_sha):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "Output paths must stay inside their declared root")
    path = Path(root) / relative
    if path.is_file():
        require(sha(path) == expected_sha, "Changed prediction file")
        payload = read(path)
        receipt = read(path.with_name("result-identity.json"))
    else:
        path = Path(str(path) + ".gz")
        manifest = read(Path(root) / "manifest.json")
        require(manifest.get("split") == "dev", "Compact package is not development data")
        item = manifest["files"][str(relative) + ".gz"]
        require(sha(path) == item["sha256"] and item["raw_sha256"] == expected_sha,
                "Changed compact prediction package")
        payload = read(path)
        require(payload["raw_sha256"] == expected_sha, "Compact original hash differs")
        receipt = payload["receipt"]
    require(receipt["output_sha256"] == expected_sha, "Prediction receipt hash differs")
    return payload, receipt["identity"]


def describe(values):
    values = list(values)
    if not values:
        return None
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values))


def quality(records):
    from sacrebleu.metrics import BLEU, CHRF
    hypotheses = [row["hypothesis"] for row in records]
    references = [[row["reference"] for row in records]]
    bleu = BLEU(lowercase=True, tokenize="13a").corpus_score(hypotheses, references)
    return dict(bleu=bleu.score, brevity_penalty=bleu.bp,
        precisions=bleu.precisions, matched_ngrams=bleu.counts, total_ngrams=bleu.totals,
        system_length=bleu.sys_len, reference_length=bleu.ref_len,
        system_reference_ratio=bleu.sys_len / bleu.ref_len if bleu.ref_len else None,
        bleu_without_brevity_penalty=bleu.score / bleu.bp if bleu.bp else None,
        decomposition_note="BLEU/BP is an algebraic diagnostic, not a replacement metric or a causal effect.",
        chrf=CHRF().corpus_score(hypotheses, references).score,
        bleu_signature="lowercase=True, tokenize=13a; default exponential smoothing",
        length_unit="SacreBLEU 13a tokens")


def token_audit(records, rows):
    """translate_one returns the prefix before the first EOS, or the full window."""
    paired = [(record, row) for record, row in zip(records, rows) if "token_ids" in record]
    if not paired:
        return dict(available_examples=0, unavailable_examples=len(records), measurements=None,
                    unavailable=["Returned token lengths and EOS positions: token_ids were not saved."])
    returned, reference, differences, eos_positions, discarded = [], [], [], [], []
    categories = {key: [] for key in ("early_eos", "final_slot_eos", "no_eos")}
    empty, special = 0, {str(EOS_ID): 0, str(MASK_ID): 0}
    for record, row in paired:
        ids, length = record["token_ids"], record["predicted_length"]
        require(isinstance(ids, list) and all(type(t) is int and 0 <= t <= MASK_ID for t in ids),
                "Invalid saved token IDs")
        require(len(ids) <= length, "Returned tokens exceed the generated target window")
        # These are pre-EOS IDs, not the raw completed denoising window.
        require(EOS_ID not in ids, "Saved EOS violates the verified pre-EOS return contract")
        special[str(MASK_ID)] += ids.count(MASK_ID)
        n = len(ids)
        returned.append(n)
        reference.append(len(row["target_ids"]))
        differences.append(n - len(row["target_ids"]))
        empty += n == 0
        if n < length:
            eos_positions.append(n)
            discarded.append(length - n - 1)
            category = "final_slot_eos" if n == length - 1 else "early_eos"
        else:
            category = "no_eos"
        categories[category].append(record)
    n = len(paired)
    return dict(available_examples=n, unavailable_examples=len(records) - n,
        measurements=dict(returned_lexical_tokens=describe(returned),
            reference_lexical_tokens=describe(reference),
            returned_minus_reference_tokens=describe(differences),
            returned_minus_predicted_lexical_capacity=describe(
                len(r["token_ids"]) - (r["predicted_length"] - 1) for r, _ in paired),
            empty_returned_sequences=empty, returned_special_token_counts=special,
            eos_inferred_from_verified_return_contract=dict(
                examples_with_eos=len(eos_positions), fraction_with_eos=len(eos_positions) / n,
                first_eos_zero_based_position=describe(eos_positions),
                discarded_slots_after_first_eos=describe(discarded),
                categories={key: dict(examples=len(group), fraction=len(group) / n,
                    quality=quality(group) if group else None) for key, group in categories.items()})),
        unavailable=["Token identities after the first EOS and later EOS occurrences were not saved."])


def assets(identity, report):
    value = identity.get("common", identity)
    return dict(model=value["model_sha256"], data=value["data_sha256"],
        counts=value.get("counts_sha256", report.get("counts_sha256")),
        selection=value["selection_sha256"],
        length_head=value.get("length_head_sha256", value.get("head_sha256")),
        length_artifact=value.get("length_head_artifact_sha256", value.get("head_artifact_sha256")),
        nucleus_p=value["nucleus_p"], top_k=value["top_k"])


def validate_records(payload, identity, rows, references):
    records, cell = payload["predictions"], identity["cell"]
    require(identity.get("split") == "dev", "Only development outputs may be audited")
    require(cell["steps"] in STEPS and cell["method"] in ("baseline", "joint", "own-marginal")
            and cell["temperature"] == .85 and cell["seed"] == 101
            and cell["strength"] == (0. if cell["method"] == "baseline" else .2)
            and cell["sampling"] == ("joint" if cell["method"] == "baseline" else cell["method"])
            and cell["length_mode"] == "predicted", "Output does not use the fixed comparison protocol")
    ids = [row["id"] for row in rows]
    require(identity["ids"] == [r["id"] for r in records] == ids
            and payload["summary"]["examples"] == len(rows), "Incomplete or reordered population")
    require([r["reference"] for r in records] == references, "Development references changed")
    signatures = []
    for record, row in zip(records, rows):
        length, reveals = record["predicted_length"], record["reveal_counts"]
        require(type(length) is int and 1 <= length <= 1024 - len(row["source_ids"]) - 2,
                "Invalid predicted target-window length")
        require(record.get("length_mode", "predicted") == "predicted"
                and record["scheduled_steps"] == cell["steps"]
                and len(reveals) == cell["steps"]
                and all(type(x) is int and x >= 0 for x in reveals)
                and sum(reveals) == length
                and record["actual_forward_calls"] == sum(x > 0 for x in reveals)
                and record["length_prediction_forwards"] == 1, "Invalid decoding schedule")
        require(isinstance(record["hypothesis"], str), "Invalid hypothesis text")
        signatures.append((length, tuple(reveals), record["actual_forward_calls"],
                           record["length_prediction_forwards"], record["scheduled_steps"]))
    return records, signatures


def audit(data, groups):
    from sacremoses import MosesDetokenizer
    data = Path(data)
    manifest, data_sha = read(data / "manifest.json"), sha(data / "manifest.json")
    require(sha(data / "dev.jsonl") == manifest["splits"]["dev"]["sha256"], "Development data changed")
    rows = [json.loads(line) for line in (data / "dev.jsonl").read_text().splitlines()]
    require(len(rows) == len({r["id"] for r in rows}) == manifest["splits"]["dev"]["rows"]
            and rows, "Invalid full development population")
    random.Random(31415).shuffle(rows)
    detokenizer = MosesDetokenizer(lang="en")
    references = [detokenizer.detokenize(r["target"].split()) for r in rows]
    labels = [group[0] for group in groups]
    require(groups and len(labels) == len(set(labels)), "Group labels must be nonempty and unique")
    result = dict(format="iwslt_output_length_audit_v1", split="dev", examples=len(rows),
        runner_sha256=sha(__file__),
        data_sha256=data_sha, groups={}, token_return_contract=dict(
            eos_id=EOS_ID, mask_id=MASK_ID, predicted_length="clamped generated target-window slots",
            token_ids="generated prefix strictly before first EOS; full window when no EOS occurred",
            eos_rule="len(token_ids) < predicted_length implies first EOS at len(token_ids)",
            early_eos="first EOS precedes the last generated slot",
            text_rule="Saved text is tokenizer.decode(returned IDs), whitespace-split, then Moses-detokenized.",
            unavailable="Discarded post-EOS identities and pre-clamping length-head argmax were not saved."))
    shared, sources, predicted, anchors = None, None, None, {}
    contract_sha = sha(Path(__file__).with_name("translation.py"))
    for label, readout_path, output_root in groups:
        report, group_rows, observed, candidate = read(readout_path), [], set(), None
        require(report.get("split") == "dev" and report["examples"] == len(rows),
                "Readout is not the full development population")
        require(isinstance(report["outputs"], dict) and report["outputs"], "Empty output manifest")
        for relative, expected_sha in sorted(report["outputs"].items()):
            payload, identity = load_output(output_root, relative, expected_sha)
            current_assets = assets(identity, report)
            require(current_assets["data"] == data_sha
                    and current_assets["counts"] == manifest["counts_sha256"]
                    and current_assets["nucleus_p"] == .9 and current_assets["top_k"] == 500
                    and current_assets["length_head"] and current_assets["length_artifact"],
                    "Output assets or candidate-support settings differ")
            require(shared is None or current_assets == shared, "Cross-method assets differ")
            shared = current_assets
            source = {key: identity["source_sha256"][key] for key in SHARED_SOURCES}
            require(source["translation.py"] == contract_sha,
                    "Translation source differs from the audited token-return contract")
            require(source == {key: report["source_sha256"][key] for key in SHARED_SOURCES}
                    and (sources is None or sources == source), "Shared decoding source changed")
            sources = source
            chosen = identity.get("diagnostic_candidate", identity.get("candidate"))
            require(chosen == report.get("diagnostic_candidate", report.get("selected")),
                    "Prediction checkpoint differs from the readout")
            require(candidate is None or chosen == candidate, "Checkpoint changes between cells")
            candidate = chosen
            records, signature = validate_records(payload, identity, rows, references)
            cell = identity["cell"]
            key = (cell["steps"], cell["method"])
            require(key not in observed, "Duplicate decoding cell")
            observed.add(key)
            require(cell["steps"] not in anchors or signature == anchors[cell["steps"]],
                    "Paired methods have different predicted lengths or reveal schedules")
            anchors[cell["steps"]] = signature
            lengths = [record["predicted_length"] for record in records]
            require(predicted is None or predicted == lengths, "Predicted lengths change across cells")
            predicted = lengths
            scores = quality(records)
            for metric in ("bleu", "chrf"):
                saved = report["steps"][str(cell["steps"])]["methods"][cell["method"]][metric]
                require(math.isfinite(payload["summary"][metric]) and math.isfinite(saved)
                        and abs(scores[metric] - payload["summary"][metric]) <= 1e-10
                        and abs(scores[metric] - saved) <= 1e-10, "Saved quality score cannot be reproduced")
            group_rows.append(dict(cell=cell, quality=scores, tokens=token_audit(records, rows),
                text_lengths=dict(hypothesis_whitespace_words=describe([len(r["hypothesis"].split()) for r in records]),
                    reference_whitespace_words=describe([len(r["reference"].split()) for r in records]),
                    hypothesis_characters=describe([len(r["hypothesis"]) for r in records]),
                    reference_characters=describe([len(r["reference"]) for r in records])),
                prediction_sha256=expected_sha))
        methods = ("baseline", "joint", "own-marginal") if any(m == "baseline" for _, m in observed) else ("joint", "own-marginal")
        require(observed == {(step, method) for step in STEPS for method in methods},
                "Incomplete full-development decoding grid")
        result["groups"][label] = dict(readout_sha256=sha(readout_path), candidate=candidate,
            role=report.get("role", "comparison"), cells=sorted(group_rows, key=lambda x: (x["cell"]["steps"], x["cell"]["method"])))
    result["predicted_lengths"] = dict(window_slots=describe(predicted),
        reference_lexical_tokens=describe([len(r["target_ids"]) for r in rows]),
        predicted_minus_reference_including_eos=describe([n - len(r["target_ids"]) - 1 for n, r in zip(predicted, rows)]))
    result["checks"] = dict(shared_assets=shared, shared_source_sha256=sources,
        exact_ordered_ids=True, exact_references=True, exact_predicted_lengths=True,
        exact_reveal_schedules=True, independently_recomputed_bleu_chrf=True,
        model_weights_loaded=False, test_data_accessed=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--group", nargs=3, action="append", required=True,
                        metavar=("LABEL", "READOUT_JSON", "OUTPUT_ROOT"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data, args.group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output),
                          groups=list(result["groups"]), examples=result["examples"])))


if __name__ == "__main__":
    main()
