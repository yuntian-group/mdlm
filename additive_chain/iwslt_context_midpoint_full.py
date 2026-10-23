"""Fixed two-step contextual midpoint on full development, joint versus marginals.

The contextual coefficient is 0.5 on the first scheduled step and zero on the
second; count/global coefficients and all other decoding settings stay fixed.
Authenticated joint outputs from the earlier 512-example subset are reused at
their original full-development positions. No training or test data are used.
Run by absolute path with the original frozen evaluator on PYTHONPATH.
"""
import argparse
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

from additive_chain import iwslt_residual_eval as evaluation
from additive_chain.backbone import file_sha256
from additive_chain.iwslt_confirm import atomic_json, read
from additive_chain.iwslt_controls_readout import verify_population
from additive_chain.iwslt_followup import output_name, verify_output


_HELPER_PATH = Path(__file__).with_name("iwslt_context_midpoint.py")
_SPEC = importlib.util.spec_from_file_location("_full_midpoint_assets", _HELPER_PATH)
midpoint = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(midpoint)
crossover, replay = midpoint.crossover, midpoint.replay

FORMAT = "iwslt_context_midpoint_full_v1"
SAMPLINGS = ("joint", "own-marginal")
PROTOCOL = {key: value for key, value in midpoint.PROTOCOL.items() if key != "sampling"}
PROTOCOL["samplings"] = list(SAMPLINGS)
PREFLIGHT_EXAMPLES = 4


def load_global_own(args, base, full):
    """Authenticate the own-marginal endpoint without the joint-only helper."""
    from sacremoses import MosesDetokenizer
    path, document = args.global_own_result, read(args.global_readout)
    cell = next(c for c in evaluation.cells("full") if c["steps"] == 2 and c["sampling"] == "own-marginal")
    identity = crossover.endpoint_identity(base, "global-global", document, cell, full)
    candidate = base.identity["selected_global"]
    expected = {f"{candidate['name']}/full/{output_name(c)}/result.json" for c in evaluation.cells("full")}
    relative = f"{candidate['name']}/full/{output_name(cell)}/result.json"
    if set(document["outputs"]) != expected or document["outputs"].get(relative) != file_sha256(path):
        raise ValueError("Global own-marginal endpoint is not authenticated by its readout")
    payload, summary = verify_output(path, identity), read(path.with_name("summary.json"))
    if summary.get("cell") != cell or summary.get("completed") != 1 or summary.get("expected") != 1:
        raise ValueError("Incomplete global own-marginal endpoint")
    detok = MosesDetokenizer(lang="en")
    verify_population(payload["predictions"], full, [detok.detokenize(row["target"].split()) for row in full], 2)
    metrics = crossover.corpus_metrics(payload["predictions"])
    for key in ("bleu", "chrf"):
        values = (payload["summary"][key], document["steps"]["2"]["methods"]["own-marginal"][key])
        if any(not math.isfinite(value) or abs(value - metrics[key]) > 1e-10 for value in values):
            raise ValueError("Global own-marginal metrics cannot be reproduced")
    return payload["predictions"], dict(global_own_result=path,
        global_own_identity=path.with_name("result-identity.json"),
        global_own_summary=path.with_name("summary.json"))


def require_same_schedule(left, right):
    keys = ("id", "reference", "predicted_length", "length_prediction_forwards", "scheduled_steps",
            "executed_steps", "actual_forward_calls", "reveal_counts", "schedule", "schedule_name", "opportunities")
    if len(left) != len(right) or any(a[key] != b[key] for a, b in zip(left, right) for key in keys):
        raise ValueError("Sampling methods do not share references, lengths and full reveal schedules")


def prepare(args):
    for old in (args.midpoint_root, args.crossover_root):
        if args.output.resolve().is_relative_to(old.resolve()) or old.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Full-development outputs must be separate from saved subset outputs")
    saved_args = SimpleNamespace(**dict(vars(args), output=args.midpoint_root))
    previous = midpoint.prepare(saved_args)
    midpoint.parity_receipt(saved_args, previous)
    subset = midpoint.validate_saved(saved_args, previous)
    document = read(args.midpoint_root / "readout.json")
    subset_paths = {str(path.relative_to(args.midpoint_root)): path
                    for path in sorted((args.midpoint_root / midpoint.POLICY / "records").glob("*.json"))}
    if (len(subset) != replay.EXAMPLES or document.get("identity") != previous.identity
            or document.get("complete") is not True or document.get("examples") != replay.EXAMPLES
            or document.get("endpoint_parity_sha256") != file_sha256(args.midpoint_root / "endpoint-parity.json")
            or document.get("outputs") != {key: file_sha256(path) for key, path in subset_paths.items()}
            or document.get("metrics") != {key: crossover.corpus_metrics(rows) for key, rows in {
                **previous.endpoints, "0.5": [subset[row["id"]] for row in previous.base.rows]}.items()}):
        raise ValueError("Saved midpoint subset is incomplete or changed")
    full = evaluation.development_rows(args.data, "full")
    if len(full) != replay.FULL_EXAMPLES or len({row["id"] for row in full}) != replay.FULL_EXAMPLES:
        raise ValueError("Expected all 7283 unique development examples")
    base = SimpleNamespace(**dict(vars(previous.base), rows=full))
    if base.positions != {row["id"]: index for index, row in enumerate(full)}:
        raise ValueError("Original full-development RNG positions changed")
    joint, paths = crossover.load_endpoint(args, base, "global-global", full)
    own, own_paths = load_global_own(args, base, full)
    paths.update(own_paths)
    require_same_schedule(joint, own)
    for path in paths.values():
        if path.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Saved endpoints must remain outside new outputs")
    identity = dict(format=FORMAT, split="dev", role="diagnostic", protocol=PROTOCOL,
        examples=len(full), ids=[row["id"] for row in full], stream_indices=list(range(len(full))),
        population_sha256=replay.digest(full), midpoint_identity=previous.identity,
        midpoint_readout_sha256=file_sha256(args.midpoint_root / "readout.json"),
        midpoint_parity_sha256=file_sha256(args.midpoint_root / "endpoint-parity.json"),
        midpoint_record_sha256={key: file_sha256(path) for key, path in subset_paths.items()},
        endpoint_sha256={key: file_sha256(path) for key, path in paths.items()},
        endpoint_population_sha256={key: replay.digest(rows) for key, rows in {"joint": joint, "own-marginal": own}.items()},
        runner_sha256=file_sha256(Path(__file__)), midpoint_helper_sha256=file_sha256(_HELPER_PATH))
    return SimpleNamespace(previous=previous, saved_args=saved_args, base=base, subset=subset,
        subset_paths=subset_paths, endpoints={"joint": joint, "own-marginal": own}, paths=paths, identity=identity)


def unchanged(args, state):
    midpoint.unchanged(state.saved_args, state.previous)
    if (file_sha256(Path(__file__)) != state.identity["runner_sha256"]
            or file_sha256(_HELPER_PATH) != state.identity["midpoint_helper_sha256"]
            or file_sha256(args.midpoint_root / "readout.json") != state.identity["midpoint_readout_sha256"]
            or file_sha256(args.midpoint_root / "endpoint-parity.json") != state.identity["midpoint_parity_sha256"]
            or {key: file_sha256(path) for key, path in state.subset_paths.items()} != state.identity["midpoint_record_sha256"]
            or {key: file_sha256(path) for key, path in state.paths.items()} != state.identity["endpoint_sha256"]):
        raise ValueError("Full-development midpoint inputs changed")


def make_runtime(args, state):
    return midpoint.make_runtime(args, state.previous)


def decode_one(row, index, model, length, decoder, sampling, strength=.5):
    if sampling not in SAMPLINGS:
        raise ValueError("Unknown sampling method")
    old_sampling = decoder.sampling
    decoder.sampling = sampling
    try:
        result = midpoint.decode_one(row, index, model, length, decoder, strength)
    finally:
        decoder.sampling = old_sampling
    return dict(result, sampling=sampling, reuse_source="new")


def validate_record(record, row, index, sampling, state):
    if record.get("sampling") != sampling or sampling not in SAMPLINGS:
        raise ValueError("Full-development sampling selector changed")
    reused = sampling == "joint" and row["id"] in state.subset
    if record.get("reuse_source") != ("previous-512-midpoint" if reused else "new"):
        raise ValueError("Full-development subset reuse changed")
    midpoint.validate_record(record, row, index, .5, state.endpoints[sampling][index])
    if reused and {key: value for key, value in record.items() if key not in ("sampling", "reuse_source")} != state.subset[row["id"]]:
        raise ValueError("Joint full-development overlap differs from saved midpoint subset")
    return record


def parity_checks(state):
    checks = []
    for offset, row in enumerate(state.previous.base.rows[:PREFLIGHT_EXAMPLES]):
        index = state.base.positions[row["id"]]
        for sampling, strength, endpoint in (("joint", 0., state.endpoints["joint"][index]),
                ("joint", 1., state.previous.endpoints["1"][offset]),
                ("joint", .5, state.subset[row["id"]]),
                ("own-marginal", 0., state.endpoints["own-marginal"][index])):
            checks.append(dict(id=row["id"], sampling=sampling, first_step_strength=strength,
                token_sha256=replay.digest(endpoint["token_ids"]), schedule_sha256=replay.digest(endpoint["schedule"])))
    return checks


def parity_receipt(args, state):
    path = args.output / "endpoint-parity.json"
    if not path.exists():
        raise ValueError("Run the endpoint parity preflight first")
    receipt = read(path)
    if receipt != dict(identity_sha256=replay.digest(state.identity), exact_token_and_schedule_parity=True,
                       checks=parity_checks(state)):
        raise ValueError("Full-development parity preflight changed")
    return receipt


def run_preflight(args, state):
    from additive_chain.iwslt_run import run_lock
    with run_lock(args.output / "preflight-lock"):
        if (args.output / "endpoint-parity.json").exists():
            parity_receipt(args, state)
            return
        model, tokenizer, length, decoder = make_runtime(args, state)
        checks = []
        try:
            for offset, row in enumerate(state.previous.base.rows[:PREFLIGHT_EXAMPLES]):
                index = state.base.positions[row["id"]]
                for sampling, strength, endpoint in (("joint", 0., state.endpoints["joint"][index]),
                        ("joint", 1., state.previous.endpoints["1"][offset]),
                        ("joint", .5, state.subset[row["id"]]),
                        ("own-marginal", 0., state.endpoints["own-marginal"][index])):
                    record = crossover.attach_text(decode_one(row, index, model, length, decoder, sampling, strength), row, tokenizer)
                    midpoint.validate_record(record, row, index, strength, endpoint)
                    if record["token_ids"] != endpoint["token_ids"] or record["hypothesis"] != endpoint["hypothesis"]:
                        raise ValueError("New runner does not reproduce saved endpoint tokens")
                    checks.append(dict(id=row["id"], sampling=sampling, first_step_strength=strength,
                        token_sha256=replay.digest(record["token_ids"]), schedule_sha256=replay.digest(record["schedule"])))
        finally:
            decoder.close()
        unchanged(args, state)
        atomic_json(args.output / "endpoint-parity.json", dict(identity_sha256=replay.digest(state.identity),
                    exact_token_and_schedule_parity=True, checks=checks))
        parity_receipt(args, state)


def validate_saved(args, state, sampling):
    folder = args.output / sampling
    if not (folder / "manifest.json").exists() or read(folder / "manifest.json") != dict(identity=state.identity, sampling=sampling):
        raise ValueError("Missing or changed full-development manifest")
    expected = {f"{index:05d}.json" for index in range(len(state.base.rows))}
    if {path.name for path in (folder / "records").glob("*.json")} - expected:
        raise ValueError("Unexpected full-development records")
    records = {}
    for index, row in enumerate(state.base.rows):
        path = folder / "records" / f"{index:05d}.json"
        if path.exists():
            item = read(path)
            if item.get("identity_sha256") != replay.digest(state.identity) or item.get("record_sha256") != replay.digest(item["record"]):
                raise ValueError("Full-development record hash/identity changed")
            records[row["id"]] = validate_record(item["record"], row, index, sampling, state)
    return records


def save_record(path, state, record):
    atomic_json(path, dict(identity_sha256=replay.digest(state.identity), record_sha256=replay.digest(record), record=record))


def run_policy(args, state):
    from additive_chain.iwslt_run import log, run_lock
    if args.sampling not in SAMPLINGS or args.max_new_examples < 0:
        raise ValueError("Invalid sampler or example limit")
    parity_receipt(args, state)
    folder = args.output / args.sampling
    with run_lock(folder):
        if not (folder / "manifest.json").exists():
            if any(path.name != "run.lock" for path in folder.iterdir()):
                raise ValueError("Nonempty full-development directory without manifest")
            atomic_json(folder / "manifest.json", dict(identity=state.identity, sampling=args.sampling))
        (folder / "records").mkdir(exist_ok=True)
        records = validate_saved(args, state, args.sampling)
        if args.sampling == "joint":
            for row in state.base.rows:
                if row["id"] in state.subset and row["id"] not in records:
                    index = state.base.positions[row["id"]]
                    record = dict(state.subset[row["id"]], sampling="joint", reuse_source="previous-512-midpoint")
                    validate_record(record, row, index, args.sampling, state)
                    save_record(folder / "records" / f"{index:05d}.json", state, record)
                    records[row["id"]] = record
        pending = [(index, row) for index, row in enumerate(state.base.rows) if row["id"] not in records]
        if pending:
            model, tokenizer, length, decoder = make_runtime(args, state)
            try:
                for count, (index, row) in enumerate(pending):
                    if args.max_new_examples and count >= args.max_new_examples:
                        break
                    record = crossover.attach_text(decode_one(row, index, model, length, decoder, args.sampling), row, tokenizer)
                    validate_record(record, row, index, args.sampling, state)
                    save_record(folder / "records" / f"{index:05d}.json", state, record)
                    records[row["id"]] = record
                    if count % 32 == 0:
                        log("context_midpoint_full_progress", sampling=args.sampling, completed=len(records), expected=len(state.base.rows))
            finally:
                decoder.close()
        unchanged(args, state)
        ordered = [records[row["id"]] for row in state.base.rows if row["id"] in records]
        atomic_json(folder / "progress.json", dict(completed=len(ordered), expected=len(state.base.rows),
            complete=len(ordered) == len(state.base.rows), metrics=crossover.corpus_metrics(ordered) if ordered else None))


def run_readout(args, state):
    from additive_chain.iwslt_run import run_lock
    parity_receipt(args, state)
    with run_lock(args.output / "readout-lock"):
        saved = {method: validate_saved(args, state, method) for method in SAMPLINGS}
        if any(len(rows) != len(state.base.rows) for rows in saved.values()):
            raise ValueError("Both samplers must finish every full-development example")
        ordered = {method: [rows[row["id"]] for row in state.base.rows] for method, rows in saved.items()}
        require_same_schedule(ordered["joint"], ordered["own-marginal"])
        metrics = {f"global-{method}": crossover.corpus_metrics(rows) for method, rows in state.endpoints.items()}
        metrics.update({f"half-{method}": crossover.corpus_metrics(rows) for method, rows in ordered.items()})
        contrasts = {f"half_minus_global_{method}": {key: metrics[f"half-{method}"][key] - metrics[f"global-{method}"][key]
                     for key in ("bleu", "chrf")} for method in SAMPLINGS}
        for prefix in ("global", "half"):
            contrasts[f"{prefix}_joint_minus_own"] = {key: metrics[f"{prefix}-joint"][key] - metrics[f"{prefix}-own-marginal"][key]
                                                      for key in ("bleu", "chrf")}
        outputs = {method: {str(path.relative_to(args.output)): file_sha256(path)
                   for path in sorted((args.output / method / "records").glob("*.json"))} for method in SAMPLINGS}
        unchanged(args, state)
        result = dict(identity=state.identity, complete=True, examples=len(state.base.rows), metrics=metrics,
            contrasts=contrasts, outputs=outputs, reused_joint_examples=len(state.subset),
            endpoint_parity_sha256=file_sha256(args.output / "endpoint-parity.json"))
        path = args.output / "readout.json"
        if path.exists() and read(path) != result:
            raise ValueError("Refusing to replace a different complete readout")
        atomic_json(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "readout"))
    for name in ("trained", "data", "length-head", "selection", "global-head", "global-selection",
                 "contextual-selection", "official-evaluation-root", "global-result", "global-readout",
                 "contextual-result", "contextual-readout", "crossover-root", "midpoint-root", "global-own-result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--sampling", choices=SAMPLINGS, default="joint")
    parser.add_argument("--max-new-examples", type=int, default=0)
    args = parser.parse_args()
    if args.command != "readout" and (args.checkpoint is None or args.tokenizer is None):
        parser.error("preflight/run require --checkpoint and --tokenizer")
    state = prepare(args)
    {"preflight": run_preflight, "run": run_policy, "readout": run_readout}[args.command](args, state)


if __name__ == "__main__":
    main()
