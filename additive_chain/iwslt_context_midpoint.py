"""One fixed first-step contextual coefficient between saved zero/one endpoints.

On the existing 512-example development subset, decode with contextual strength
0.5 at time 1 and zero at time 0.5. Count and global residuals remain unchanged.
The saved global/global and contextual/global trajectories supply strengths 0
and 1. A four-example GPU preflight checks both endpoints through this runner.
No training, test evaluation, new candidate extraction, or grid search occurs.

Run by absolute path with the frozen evaluator on PYTHONPATH. The sibling
crossover helper is loaded explicitly and authenticated independently of it.
"""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from additive_chain.backbone import file_sha256
from additive_chain.generation import sample_window, stream_seed
from additive_chain.iwslt_confirm import atomic_json, read
from additive_chain.iwslt_followup import StateObserver
from additive_chain.translation import CONTEXT_LENGTH, EOS_ID, MASK_ID


_HELPER_PATH = Path(__file__).with_name("iwslt_context_crossover.py")
_SPEC = importlib.util.spec_from_file_location("_midpoint_crossover_assets", _HELPER_PATH)
crossover = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(crossover)
replay = crossover.replay

FORMAT = "iwslt_context_midpoint_v1"
POLICY = "half-global"
FIRST_STRENGTH = .5
ENDPOINT_STRENGTHS = (0., 1.)
PREFLIGHT_EXAMPLES = 4
PROTOCOL = dict(steps=2, seed=101, nucleus_p=.9, top_k=500, temperature=.85,
    count_strength=.2, global_strength=1., contextual_strengths=[.5, 0.],
    length_mode="predicted", sampling="joint", stream_index="original_full_development_position",
    switching="scheduled_time_not_forward_call_count")


def prepare(args):
    if (args.output.resolve().is_relative_to(args.crossover_root.resolve())
            or args.crossover_root.resolve().is_relative_to(args.output.resolve())):
        raise ValueError("Midpoint outputs and saved crossover outputs must be separate")
    previous = crossover.prepare(args)
    previous_args = SimpleNamespace(**dict(vars(args), output=args.crossover_root))
    crossover.parity_receipt(previous_args, previous)
    document_path = args.crossover_root / "readout.json"
    document = read(document_path)
    if (document.get("identity") != previous.identity or document.get("complete") is not True
            or document.get("examples") != replay.EXAMPLES
            or set(document.get("metrics", {})) != set(crossover.POLICIES + crossover.ENDPOINTS)
            or set(document.get("outputs", {})) != set(crossover.POLICIES)
            or document.get("endpoint_parity_sha256") != file_sha256(args.crossover_root / "endpoint-parity.json")):
        raise ValueError("Saved crossover readout identity or completeness changed")
    saved = crossover.validate_saved(previous_args, previous, "contextual-global")
    if len(saved) != replay.EXAMPLES:
        raise ValueError("Saved contextual/global endpoint is incomplete")
    endpoints = {"0": previous.endpoints["global-global"],
                 "1": [saved[row["id"]] for row in previous.base.rows]}
    paths = {str(path.relative_to(args.crossover_root)): path for path in
             sorted((args.crossover_root / "contextual-global" / "records").glob("*.json"))}
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    if document["outputs"]["contextual-global"] != hashes:
        raise ValueError("Saved contextual/global record hashes changed")
    for key, policy in (("0", "global-global"), ("1", "contextual-global")):
        if document["metrics"][policy] != crossover.corpus_metrics(endpoints[key]):
            raise ValueError("Saved endpoint metrics cannot be reproduced")
    identity = dict(format=FORMAT, split="dev", role="diagnostic", protocol=PROTOCOL,
        policy=POLICY, examples=replay.EXAMPLES, ids=previous.identity["ids"],
        stream_indices=previous.identity["stream_indices"], population_sha256=previous.identity["population_sha256"],
        crossover_identity=previous.identity, crossover_readout_sha256=file_sha256(document_path),
        crossover_parity_sha256=file_sha256(args.crossover_root / "endpoint-parity.json"),
        endpoint_record_sha256=hashes,
        endpoint_subset_sha256={key: replay.digest(rows) for key, rows in endpoints.items()},
        runner_sha256=file_sha256(Path(__file__)), crossover_helper_sha256=file_sha256(_HELPER_PATH))
    return SimpleNamespace(previous=previous, base=previous.base, endpoints=endpoints,
        paths=paths, identity=identity)


def unchanged(args, state):
    crossover.unchanged(args, state.previous)
    if (file_sha256(Path(__file__)) != state.identity["runner_sha256"]
            or file_sha256(_HELPER_PATH) != state.identity["crossover_helper_sha256"]
            or file_sha256(args.crossover_root / "readout.json") != state.identity["crossover_readout_sha256"]
            or file_sha256(args.crossover_root / "endpoint-parity.json") != state.identity["crossover_parity_sha256"]
            or {key: file_sha256(path) for key, path in state.paths.items()} != state.identity["endpoint_record_sha256"]):
        raise ValueError("Midpoint source or saved endpoint inputs changed")


class ScheduledDecoder:
    """Use one decoder/cache; apply temperature exactly once to each coefficient."""
    def __init__(self, decoder, first_strength):
        if first_strength not in (0., .5, 1.):
            raise ValueError("Only fixed first-step strengths 0, 0.5, and 1 are registered")
        self.decoder, self.first_strength = decoder, first_strength
        self.device, self.mask_id = decoder.device, decoder.mask_id
        self.full_contextual_strength = decoder.contextual_strength
        if self.full_contextual_strength != 1. / .85:
            raise ValueError("Midpoint requires the original full contextual coefficient")
        self.states = []

    def reveal(self, output, tokens, reveal_mask, generator):
        time = output.scheduled_time
        if time not in (1., .5):
            raise ValueError("Midpoint only supports the fixed two-step schedule")
        step = 1 if time == 1. else 2
        strength = self.first_strength if step == 1 else 0.
        before = tokens[0].tolist()
        self.decoder.contextual_strength = strength / .85
        try:
            result = self.decoder.reveal(output, tokens, reveal_mask, generator)
        finally:
            self.decoder.contextual_strength = self.full_contextual_strength
        self.states.append(dict(scheduled_step=step, time=time, contextual_strength=strength,
            applied_contextual_strength=strength / .85, input_tokens=before,
            selected_positions=reveal_mask[0].nonzero().flatten().tolist(), output_tokens=result[0].tolist()))
        return result


@torch.no_grad()
def decode_one(row, stream_index, model, length_head, decoder, first_strength):
    source = {"source_ids": row["source_ids"]}
    device = decoder.device
    features = model.source_features([source], device)
    predicted = int(length_head(features).argmax(-1).item())
    prefix = torch.tensor([[EOS_ID] + list(source["source_ids"]) + [EOS_ID]], device=device)
    length = max(1, min(predicted, CONTEXT_LENGTH - prefix.shape[1]))
    target = torch.full((1, length), MASK_ID, device=device, dtype=torch.long)

    def forward(current, time):
        tokens = torch.cat((prefix, current), -1)
        output = model(tokens, torch.ones_like(tokens, dtype=torch.bool))
        return SimpleNamespace(log_probs=output.log_probs[:, -length:],
                               hidden=output.hidden[:, -length:], scheduled_time=time)

    switching = ScheduledDecoder(decoder, first_strength)
    observer = StateObserver(decoder, probe=False)
    streams = {kind: torch.Generator(device=device).manual_seed(stream_seed("iwslt-101", stream_index, kind))
               for kind in ("reveal", "proposal")}
    result = sample_window(target, forward, switching, steps=2,
        reveal_generator=streams["reveal"], proposal_generator=streams["proposal"], observer=observer)
    raw = result.tokens[0].tolist()
    ids = raw[:raw.index(EOS_ID)] if EOS_ID in raw else raw
    return dict(id=row["id"], stream_index=stream_index, policy=POLICY,
        contextual_strengths=[first_strength, 0.], source_sha256=replay.digest(row["source_ids"]),
        token_ids=ids, raw_target_tokens=raw, predicted_length=length, length_prediction_forwards=1,
        opportunities=observer.steps, intermediate_states=switching.states, **result.stats)


def validate_record(record, row, position, first_strength, endpoint):
    if (record["id"] != row["id"] or record["stream_index"] != position or record["policy"] != POLICY
            or record["contextual_strengths"] != [first_strength, 0.]
            or record["source_sha256"] != replay.digest(row["source_ids"])
            or record["reference"] != endpoint["reference"]
            or record["predicted_length"] != endpoint["predicted_length"]
            or record["length_prediction_forwards"] != 1):
        raise ValueError("Midpoint population, strength, reference, or predicted length changed")
    for key in ("scheduled_steps", "executed_steps", "actual_forward_calls", "reveal_counts", "schedule", "schedule_name", "opportunities"):
        if record[key] != endpoint[key]:
            raise ValueError("Midpoint reveal schedule differs from saved endpoints")
    raw = record["raw_target_tokens"]
    if len(raw) != record["predicted_length"] or any(type(t) is not int or not 0 <= t < MASK_ID for t in raw):
        raise ValueError("Invalid final midpoint tokens")
    returned = raw[:raw.index(EOS_ID)] if EOS_ID in raw else raw
    if record["token_ids"] != returned:
        raise ValueError("Midpoint EOS return contract changed")
    states = record["intermediate_states"]
    if [s["scheduled_step"] for s in states] != [i + 1 for i, count in enumerate(record["reveal_counts"]) if count]:
        raise ValueError("Missing or duplicated scheduled intermediate state")
    if "intermediate_states" in endpoint and [
            (state["scheduled_step"], state["selected_positions"]) for state in states] != [
            (state["scheduled_step"], state["selected_positions"]) for state in endpoint["intermediate_states"]]:
        raise ValueError("Midpoint reveal positions differ from the saved crossover endpoint")
    current = [MASK_ID] * len(raw)
    for state in states:
        step, selected = state["scheduled_step"], state["selected_positions"]
        strength = first_strength if step == 1 else 0.
        if (state["time"] != (1. if step == 1 else .5)
                or state["contextual_strength"] != strength or state["applied_contextual_strength"] != strength / .85
                or state["input_tokens"] != current or len(selected) != record["reveal_counts"][step - 1]
                or selected != sorted(set(selected)) or any(type(p) is not int or not 0 <= p < len(raw) for p in selected)
                or any(current[p] != MASK_ID for p in selected) or len(state["output_tokens"]) != len(raw)):
            raise ValueError("Invalid midpoint coefficient or intermediate reveal")
        output = state["output_tokens"]
        if any((type(value) is not int or not 0 <= value < MASK_ID) if pos in selected else value != current[pos]
               for pos, value in enumerate(output)):
            raise ValueError("Midpoint changed unselected identities")
        current = output
    if current != raw:
        raise ValueError("Saved intermediate trajectory does not produce final tokens")
    return record


def parity_receipt(args, state):
    path = args.output / "endpoint-parity.json"
    if not path.exists():
        raise ValueError("Run the four-example endpoint parity preflight first")
    receipt = read(path)
    rows = state.base.rows[:PREFLIGHT_EXAMPLES]
    checks = [dict(id=row["id"], first_step_strength=strength,
        token_sha256=replay.digest(state.endpoints[str(int(strength))][index]["token_ids"]),
        schedule_sha256=replay.digest(state.endpoints[str(int(strength))][index]["schedule"]))
        for index, row in enumerate(rows) for strength in ENDPOINT_STRENGTHS]
    if (receipt.get("identity_sha256") != replay.digest(state.identity)
            or receipt.get("ids") != [row["id"] for row in rows]
            or receipt.get("first_step_strengths") != list(ENDPOINT_STRENGTHS)
            or receipt.get("exact_token_and_schedule_parity") is not True or receipt.get("checks") != checks):
        raise ValueError("Endpoint parity preflight does not match these inputs")
    return receipt


def validate_saved(args, state):
    folder = args.output / POLICY
    manifest = folder / "manifest.json"
    if not manifest.exists() or read(manifest) != dict(identity=state.identity, policy=POLICY):
        raise ValueError("Missing or changed midpoint policy manifest")
    expected = {f"{state.base.positions[row['id']]:05d}.json" for row in state.base.rows}
    if {path.name for path in (folder / "records").glob("*.json")} - expected:
        raise ValueError("Unexpected midpoint records")
    records = {}
    for row, endpoint in zip(state.base.rows, state.endpoints["1"]):
        position = state.base.positions[row["id"]]
        path = folder / "records" / f"{position:05d}.json"
        if path.exists():
            value = read(path)
            if value["identity_sha256"] != replay.digest(state.identity) or value["record_sha256"] != replay.digest(value["record"]):
                raise ValueError("Midpoint record hash/identity changed")
            records[row["id"]] = validate_record(value["record"], row, position, FIRST_STRENGTH, endpoint)
    return records


def make_runtime(args, state):
    return crossover.make_runtime(args, state.previous)


def run_preflight(args, state):
    from additive_chain.iwslt_run import run_lock
    with run_lock(args.output / "preflight-lock"):
        if (args.output / "endpoint-parity.json").exists():
            parity_receipt(args, state)
            return
        model, tokenizer, length, decoder = make_runtime(args, state)
        checks = []
        try:
            for index, row in enumerate(state.base.rows[:PREFLIGHT_EXAMPLES]):
                position = state.base.positions[row["id"]]
                for strength in ENDPOINT_STRENGTHS:
                    record = crossover.attach_text(decode_one(row, position, model, length, decoder, strength), row, tokenizer)
                    endpoint = state.endpoints[str(int(strength))][index]
                    validate_record(record, row, position, strength, state.endpoints["1"][index])
                    if record["token_ids"] != endpoint["token_ids"] or record["hypothesis"] != endpoint["hypothesis"]:
                        raise ValueError("Endpoint decoding does not reproduce its saved output")
                    checks.append(dict(id=row["id"], first_step_strength=strength,
                        token_sha256=replay.digest(record["token_ids"]), schedule_sha256=replay.digest(record["schedule"])))
        finally:
            decoder.close()
        unchanged(args, state)
        atomic_json(args.output / "endpoint-parity.json", dict(identity_sha256=replay.digest(state.identity),
            ids=[row["id"] for row in state.base.rows[:PREFLIGHT_EXAMPLES]], first_step_strengths=list(ENDPOINT_STRENGTHS),
            exact_token_and_schedule_parity=True, checks=checks))


def run_policy(args, state):
    from additive_chain.iwslt_run import log, run_lock
    if args.max_new_examples < 0:
        raise ValueError("Pass a nonnegative example limit")
    parity_receipt(args, state)
    folder = args.output / POLICY
    with run_lock(folder):
        manifest = folder / "manifest.json"
        if not manifest.exists():
            if any(path.name != "run.lock" for path in folder.iterdir()):
                raise ValueError("Nonempty midpoint directory without manifest")
            atomic_json(manifest, dict(identity=state.identity, policy=POLICY))
        (folder / "records").mkdir(exist_ok=True)
        records = validate_saved(args, state)
        pending = [(index, row) for index, row in enumerate(state.base.rows) if row["id"] not in records]
        if pending:
            model, tokenizer, length, decoder = make_runtime(args, state)
            try:
                for new_index, (index, row) in enumerate(pending):
                    if args.max_new_examples and new_index >= args.max_new_examples:
                        break
                    position = state.base.positions[row["id"]]
                    record = crossover.attach_text(decode_one(row, position, model, length, decoder, FIRST_STRENGTH), row, tokenizer)
                    validate_record(record, row, position, FIRST_STRENGTH, state.endpoints["1"][index])
                    atomic_json(folder / "records" / f"{position:05d}.json",
                        dict(identity_sha256=replay.digest(state.identity), record_sha256=replay.digest(record), record=record))
                    records[row["id"]] = record
                    log("context_midpoint_progress", completed=len(records), expected=replay.EXAMPLES)
            finally:
                decoder.close()
        unchanged(args, state)
        ordered = [records[row["id"]] for row in state.base.rows if row["id"] in records]
        atomic_json(folder / "progress.json", dict(completed=len(ordered), expected=replay.EXAMPLES,
            complete=len(ordered) == replay.EXAMPLES, metrics=crossover.corpus_metrics(ordered) if ordered else None))


def run_readout(args, state):
    from additive_chain.iwslt_run import run_lock
    parity_receipt(args, state)
    with run_lock(args.output / "readout-lock"):
        saved = validate_saved(args, state)
        if len(saved) != replay.EXAMPLES:
            raise ValueError("Midpoint must finish every fixed development example")
        ordered = [saved[row["id"]] for row in state.base.rows]
        metrics = {key: crossover.corpus_metrics(records) for key, records in
                   {**state.endpoints, "0.5": ordered}.items()}
        contrasts = {name: {metric: metrics["0.5"][metric] - metrics[key][metric] for metric in ("bleu", "chrf")}
                     for name, key in (("half_minus_zero", "0"), ("half_minus_one", "1"))}
        outputs = {str(path.relative_to(args.output)): file_sha256(path)
                   for path in sorted((args.output / POLICY / "records").glob("*.json"))}
        unchanged(args, state)
        result = dict(identity=state.identity, complete=True, examples=replay.EXAMPLES,
            metrics=metrics, contrasts=contrasts, outputs=outputs,
            endpoint_parity_sha256=file_sha256(args.output / "endpoint-parity.json"))
        destination = args.output / "readout.json"
        if destination.exists() and read(destination) != result:
            raise ValueError("Refusing to replace a different complete midpoint readout")
        atomic_json(destination, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "readout"))
    for name in ("trained", "data", "length-head", "selection", "global-head", "global-selection",
                 "contextual-selection", "official-evaluation-root", "global-result", "global-readout",
                 "contextual-result", "contextual-readout", "crossover-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-new-examples", type=int, default=0)
    args = parser.parse_args()
    if args.command != "readout" and (args.checkpoint is None or args.tokenizer is None):
        parser.error("preflight/run require --checkpoint and --tokenizer")
    state = prepare(args)
    {"preflight": run_preflight, "run": run_policy, "readout": run_readout}[args.command](args, state)


if __name__ == "__main__":
    main()
