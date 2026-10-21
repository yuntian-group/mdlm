"""Two-step global/contextual head crossovers on a fixed development subset.

The frozen conditional translator, predicted-length head, count potential,
candidate extraction, temperature, and sampling schedule remain unchanged.
Only the contextual residual is enabled or disabled at each scheduled step.
Saved full-development global/global and contextual/contextual outputs supply
the endpoints; they are authenticated and reduced to the same 512 examples.
No training, test evaluation, candidate augmentation, or strength search occurs.

Run by absolute path with the original frozen evaluator on PYTHONPATH. The
published replay helper alongside this file authenticates the original assets
and fixed development population; its separate source hash is recorded.
"""
import argparse
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from additive_chain import iwslt_residual_eval as evaluation
from additive_chain.backbone import file_sha256
from additive_chain.generation import sample_window, stream_seed
from additive_chain.iwslt_confirm import atomic_json, read
from additive_chain.iwslt_controls_readout import require_same_schedule, verify_population
from additive_chain.iwslt_followup import StateObserver, output_name, verify_output
from additive_chain.translation import CONTEXT_LENGTH, EOS_ID, MASK_ID


_HELPER_PATH = Path(__file__).with_name("iwslt_context_replay.py")
_SPEC = importlib.util.spec_from_file_location("_crossover_replay_assets", _HELPER_PATH)
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)

FORMAT = "iwslt_context_crossover_v1"
POLICIES = ("global-contextual", "contextual-global")
ENDPOINTS = ("global-global", "contextual-contextual")
PREFLIGHT_EXAMPLES = 4
PROTOCOL = dict(steps=2, seed=101, nucleus_p=.9, top_k=500, temperature=.85,
                count_strength=.2, global_strength=1., contextual_strength=1.,
                length_mode="predicted", sampling="joint",
                stream_index="original_full_development_position",
                switching="scheduled_time_not_forward_call_count")


def corpus_metrics(records):
    from sacrebleu.metrics import BLEU, CHRF
    if not records:
        raise ValueError("Cannot score an empty crossover population")
    hypotheses, references = [r["hypothesis"] for r in records], [[r["reference"] for r in records]]
    bleu = BLEU(lowercase=True, tokenize="13a").corpus_score(hypotheses, references)
    return dict(examples=len(records), bleu=bleu.score,
        chrf=CHRF().corpus_score(hypotheses, references).score,
        brevity_penalty=bleu.bp, ngram_precisions=bleu.precisions,
        hypothesis_words=bleu.sys_len, reference_words=bleu.ref_len,
        mean_returned_tokens=sum(len(r["token_ids"]) for r in records) / len(records),
        mean_predicted_length=sum(r["predicted_length"] for r in records) / len(records),
        early_eos_examples=sum(len(r["token_ids"]) < r["predicted_length"] - 1 for r in records),
        no_eos_examples=sum(len(r["token_ids"]) == r["predicted_length"] for r in records))


def endpoint_identity(base, policy, document, cell, full):
    identity = base.identity
    if (document.get("split") != "dev" or document.get("examples") != replay.FULL_EXAMPLES
            or document.get("source_sha256") != identity["source_sha256"]
            or set(document.get("steps", {})) != {"2", "4", "8", "32"}
            or len(document.get("outputs", {})) != 8):
        raise ValueError("Endpoint readout population, completeness, or frozen source changed")
    if policy == "global-global":
        common = dict(base.common, inherited_global_sha256=None)
        candidate = identity["selected_global"]
        selection_sha = identity["input_sha256"]["global_selection"]
        if (document.get("kind") != "global" or document.get("common") != common
                or document.get("selected") != candidate
                or document.get("residual_selection_sha256") != selection_sha):
            raise ValueError("Global endpoint does not use the selected global model")
        return evaluation.build_identity(cell, full, "full", "global", candidate, common, selection_sha)
    if policy != "contextual-contextual":
        raise ValueError("Unknown endpoint policy")
    candidate = identity["contextual_diagnostic"]
    reference = dict(relative_path=identity["frozen_reference_relative_path"],
        output_sha256=identity["input_sha256"]["reference_output"],
        identity_sha256=identity["input_sha256"]["reference_identity"])
    fixed = dict(format="iwslt_contextual_nonzero_diagnostic_v1", split="dev", kind="contextual",
        role="diagnostic", policy="highest-ranked-nonzero-from-official-zero-inclusive-ranking",
        official_selected=identity["official_contextual"], diagnostic_candidate=candidate,
        official_selection_sha256=identity["input_sha256"]["contextual_selection"],
        common=base.common, frozen_evaluator_reference=reference,
        source_sha256=identity["source_sha256"],
        training_source_sha256=identity["contextual_training_source_sha256"])
    if any(document.get(key) != value for key, value in fixed.items()):
        raise ValueError("Contextual endpoint diagnostic identity changed")
    runner = document.get("runner_sha256")
    if not isinstance(runner, str) or len(runner) != 64:
        raise ValueError("Missing original contextual diagnostic runner hash")
    return dict(fixed, experiment="iwslt-contextual-nonzero-diagnostic", population="full",
                runner_sha256=runner, cell=cell, ids=[r["id"] for r in full])


def load_endpoint(args, base, policy, full):
    from sacremoses import MosesDetokenizer
    kind = "global" if policy == "global-global" else "contextual"
    path, readout_path = getattr(args, kind + "_result"), getattr(args, kind + "_readout")
    document = read(readout_path)
    cell = next(c for c in evaluation.cells("full") if c["steps"] == 2 and c["sampling"] == "joint")
    expected = endpoint_identity(base, policy, document, cell, full)
    candidate = base.identity["selected_global" if kind == "global" else "contextual_diagnostic"]
    expected_outputs = {f"{candidate['name']}/full/{output_name(item)}/result.json"
                        for item in evaluation.cells("full")}
    if set(document["outputs"]) != expected_outputs:
        raise ValueError("Endpoint readout must contain all eight registered decoding cells")
    relative = f"{candidate['name']}/full/{output_name(cell)}/result.json"
    if document["outputs"].get(relative) != file_sha256(path):
        raise ValueError("Endpoint result is not authenticated by its readout")
    payload = verify_output(path, expected)
    summary = read(path.with_name("summary.json"))
    if (summary.get("cell") != cell or summary.get("completed") != 1 or summary.get("expected") != 1):
        raise ValueError("Incomplete endpoint summary")
    detok = MosesDetokenizer(lang="en")
    refs = [detok.detokenize(row["target"].split()) for row in full]
    verify_population(payload["predictions"], full, refs, 2)
    computed = corpus_metrics(payload["predictions"])
    for metric in ("bleu", "chrf"):
        if (not math.isfinite(payload["summary"][metric])
                or not math.isfinite(document["steps"]["2"]["methods"]["joint"][metric])
                or abs(payload["summary"][metric] - computed[metric]) > 1e-10
                or abs(document["steps"]["2"]["methods"]["joint"][metric] - computed[metric]) > 1e-10):
            raise ValueError("Endpoint corpus metrics cannot be reproduced")
    by_id = {row["id"]: row for row in payload["predictions"]}
    return [by_id[row["id"]] for row in base.rows], {
        f"{kind}_result": path, f"{kind}_identity": path.with_name("result-identity.json"),
        f"{kind}_summary": path.with_name("summary.json"), f"{kind}_readout": readout_path}


def prepare(args):
    base = replay.prepare(args)
    full = evaluation.development_rows(args.data, "full")
    endpoints, paths = {}, {}
    for policy in ENDPOINTS:
        endpoints[policy], inputs = load_endpoint(args, base, policy, full)
        paths.update(inputs)
    require_same_schedule(endpoints[ENDPOINTS[0]], endpoints[ENDPOINTS[1]])
    for global_record, contextual_record in zip(endpoints[ENDPOINTS[0]], endpoints[ENDPOINTS[1]]):
        if any(global_record[key] != contextual_record[key] for key in (
                "id", "reference", "predicted_length", "length_prediction_forwards", "scheduled_steps",
                "executed_steps", "actual_forward_calls", "reveal_counts", "schedule", "schedule_name", "opportunities")):
            raise ValueError("Endpoint references, predicted lengths, or complete schedules differ")
    for path in paths.values():
        if path.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Saved endpoints must remain outside crossover output")
    identity = dict(format=FORMAT, split="dev", role="diagnostic", protocol=PROTOCOL,
        policies=list(POLICIES), examples=replay.EXAMPLES, ids=base.identity["ids"],
        stream_indices=base.identity["stream_indices"], population_sha256=base.identity["population_sha256"],
        replay_asset_identity=base.identity,
        endpoint_sha256={key: file_sha256(path) for key, path in paths.items()},
        endpoint_subset_sha256={key: replay.digest(value) for key, value in endpoints.items()},
        runner_sha256=file_sha256(Path(__file__)), replay_helper_sha256=file_sha256(_HELPER_PATH))
    return SimpleNamespace(base=base, endpoints=endpoints, paths=paths, identity=identity)


def unchanged(args, state):
    replay.unchanged(args, state.base)
    if (file_sha256(Path(__file__)) != state.identity["runner_sha256"]
            or file_sha256(_HELPER_PATH) != state.identity["replay_helper_sha256"]
            or {key: file_sha256(path) for key, path in state.paths.items()} != state.identity["endpoint_sha256"]):
        raise ValueError("Crossover source or endpoint inputs changed")


class ScheduledDecoder:
    """One prepared decoder/cache, with a contextual coefficient set by time."""
    def __init__(self, decoder, policy):
        if policy not in POLICIES + ENDPOINTS:
            raise ValueError("Unregistered crossover policy")
        self.decoder, self.policy = decoder, policy
        self.device, self.mask_id = decoder.device, decoder.mask_id
        self.full_contextual_strength = decoder.contextual_strength
        if self.full_contextual_strength != 1. / .85:
            raise ValueError("Crossover requires the original full contextual coefficient")
        self.states = []

    def reveal(self, output, tokens, reveal_mask, generator):
        time = output.scheduled_time
        if time not in (1., .5):
            raise ValueError("Crossover only supports the fixed two-step schedule")
        step = 0 if time == 1. else 1
        head = self.policy.split("-")[step]
        before = tokens[0].tolist()
        self.decoder.contextual_strength = self.full_contextual_strength if head == "contextual" else 0.
        try:
            result = self.decoder.reveal(output, tokens, reveal_mask, generator)
        finally:
            self.decoder.contextual_strength = self.full_contextual_strength
        self.states.append(dict(scheduled_step=step + 1, time=time, head=head,
            input_tokens=before, selected_positions=reveal_mask[0].nonzero().flatten().tolist(),
            output_tokens=result[0].tolist()))
        return result


@torch.no_grad()
def decode_one(row, stream_index, model, length_head, decoder, policy):
    """Gold target identities never enter a model forward or length prediction."""
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

    switching = ScheduledDecoder(decoder, policy)
    observer = StateObserver(decoder, probe=False)
    streams = {kind: torch.Generator(device=device).manual_seed(stream_seed("iwslt-101", stream_index, kind))
               for kind in ("reveal", "proposal")}
    result = sample_window(target, forward, switching, steps=2,
        reveal_generator=streams["reveal"], proposal_generator=streams["proposal"], observer=observer)
    raw = result.tokens[0].tolist()
    ids = raw[:raw.index(EOS_ID)] if EOS_ID in raw else raw
    return dict(id=row["id"], stream_index=stream_index, policy=policy,
        source_sha256=replay.digest(row["source_ids"]), token_ids=ids, raw_target_tokens=raw,
        predicted_length=length, length_prediction_forwards=1,
        opportunities=observer.steps, intermediate_states=switching.states, **result.stats)


def validate_record(record, row, position, policy, endpoint, first_endpoint=None):
    if (record["id"] != row["id"] or record["stream_index"] != position or record["policy"] != policy
            or record["source_sha256"] != replay.digest(row["source_ids"])
            or record["reference"] != endpoint["reference"]
            or record["predicted_length"] != endpoint["predicted_length"]
            or record["length_prediction_forwards"] != 1):
        raise ValueError("Crossover population, reference, or predicted length changed")
    for key in ("scheduled_steps", "executed_steps", "actual_forward_calls", "reveal_counts", "schedule", "schedule_name", "opportunities"):
        if record[key] != endpoint[key]:
            raise ValueError("Crossover reveal schedule differs from saved endpoints")
    raw = record["raw_target_tokens"]
    if len(raw) != record["predicted_length"] or any(type(token) is not int or not 0 <= token < MASK_ID for token in raw):
        raise ValueError("Invalid final crossover tokens")
    returned = raw[:raw.index(EOS_ID)] if EOS_ID in raw else raw
    if record["token_ids"] != returned:
        raise ValueError("Crossover EOS return contract changed")
    states = record["intermediate_states"]
    expected_steps = [step + 1 for step, count in enumerate(record["reveal_counts"]) if count]
    if [state["scheduled_step"] for state in states] != expected_steps:
        raise ValueError("Missing or duplicated scheduled intermediate state")
    current = [MASK_ID] * len(raw)
    for state in states:
        step = state["scheduled_step"]
        selected = state["selected_positions"]
        if (state["time"] != (1. if step == 1 else .5) or state["head"] != policy.split("-")[step - 1]
                or state["input_tokens"] != current or len(selected) != record["reveal_counts"][step - 1]
                or selected != sorted(set(selected)) or any(type(p) is not int or not 0 <= p < len(raw) for p in selected)
                or any(current[p] != MASK_ID for p in selected)
                or len(state["output_tokens"]) != len(raw)):
            raise ValueError("Invalid crossover head switch or intermediate reveal")
        output = state["output_tokens"]
        if any((type(value) is not int or not 0 <= value < MASK_ID) if pos in selected else value != current[pos]
               for pos, value in enumerate(output)):
            raise ValueError("Crossover changed unselected identities")
        if step == 1:
            original = endpoint if first_endpoint is None else first_endpoint
            known = dict(enumerate(original["token_ids"]))
            if len(original["token_ids"]) < original["predicted_length"]:
                known[len(original["token_ids"])] = EOS_ID
            if any(output[pos] != known[pos] for pos in selected if pos in known):
                raise ValueError("First committed tokens differ from the corresponding constant-policy endpoint")
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
    expected_checks = [dict(id=row["id"], policy=policy,
        token_sha256=replay.digest(state.endpoints[policy][index]["token_ids"]),
        schedule_sha256=replay.digest(state.endpoints[policy][index]["schedule"]))
        for index, row in enumerate(rows) for policy in ENDPOINTS]
    if (receipt.get("identity_sha256") != replay.digest(state.identity)
            or receipt.get("ids") != [row["id"] for row in rows]
            or receipt.get("policies") != list(ENDPOINTS)
            or receipt.get("exact_token_and_schedule_parity") is not True
            or receipt.get("checks") != expected_checks):
        raise ValueError("Endpoint parity preflight does not match these inputs")
    return receipt


def validate_saved(args, state, policy):
    folder = args.output / policy
    manifest = folder / "manifest.json"
    if not manifest.exists() or read(manifest) != dict(identity=state.identity, policy=policy):
        raise ValueError("Missing or changed crossover policy manifest")
    expected = {f"{state.base.positions[row['id']]:05d}.json" for row in state.base.rows}
    if {path.name for path in (folder / "records").glob("*.json")} - expected:
        raise ValueError("Unexpected crossover records")
    records = {}
    first_policy = policy.split("-")[0]
    for index, (row, endpoint) in enumerate(zip(state.base.rows, state.endpoints["global-global"])):
        position = state.base.positions[row["id"]]
        path = folder / "records" / f"{position:05d}.json"
        if path.exists():
            value = read(path)
            if value["identity_sha256"] != replay.digest(state.identity) or value["record_sha256"] != replay.digest(value["record"]):
                raise ValueError("Crossover record hash/identity changed")
            records[row["id"]] = validate_record(value["record"], row, position, policy, endpoint,
                state.endpoints[first_policy + "-" + first_policy][index])
    return records


def make_runtime(args, state):
    from additive_chain.generation import ChainDecoder
    from additive_chain.iwslt_length_refit import load_refit_head
    from additive_chain.translation import load_conditional
    model, tokenizer = load_conditional(args.checkpoint, args.tokenizer)
    saved = torch.load(args.trained, map_location="cuda", weights_only=False)
    model.load_state_dict(saved["model"])
    del saved
    model.eval().requires_grad_(False)
    length = load_refit_head(args.length_head, state.base.common["model_sha256"], state.base.common["data_sha256"], "cuda")
    global_head, contextual = state.base.inherited.cuda(), state.base.head.cuda()
    decoder = ChainDecoder(mask_id=MASK_ID, device="cuda",
        count_statistics=torch.load(args.data / "counts.pt", weights_only=True),
        count_strength=.2, top_k=500, nucleus_p=.9, fast_candidates=True, sampling="joint", temperature=.85,
        global_head=global_head, global_strength=1., contextual_head=contextual, contextual_strength=1.)
    decoder.prepare()
    return model, tokenizer, length, decoder


def attach_text(record, row, tokenizer):
    from sacremoses import MosesDetokenizer
    detok = MosesDetokenizer(lang="en")
    return dict(record, hypothesis=detok.detokenize(tokenizer.decode(record["token_ids"]).split()),
                reference=detok.detokenize(row["target"].split()))


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
                for policy in ENDPOINTS:
                    record = attach_text(decode_one(row, position, model, length, decoder, policy), row, tokenizer)
                    endpoint = state.endpoints[policy][index]
                    validate_record(record, row, position, policy, endpoint)
                    if record["token_ids"] != endpoint["token_ids"] or record["hypothesis"] != endpoint["hypothesis"]:
                        raise ValueError("Constant-policy decoding does not reproduce its saved endpoint")
                    checks.append(dict(id=row["id"], policy=policy,
                        token_sha256=replay.digest(record["token_ids"]), schedule_sha256=replay.digest(record["schedule"])))
        finally:
            decoder.close()
        unchanged(args, state)
        atomic_json(args.output / "endpoint-parity.json", dict(identity_sha256=replay.digest(state.identity),
            ids=[row["id"] for row in state.base.rows[:PREFLIGHT_EXAMPLES]], policies=list(ENDPOINTS),
            exact_token_and_schedule_parity=True, checks=checks))


def run_policy(args, state):
    from additive_chain.iwslt_run import log, run_lock
    if args.policy not in POLICIES or args.max_new_examples < 0:
        raise ValueError("Pass a registered hybrid policy and nonnegative example limit")
    parity_receipt(args, state)
    folder = args.output / args.policy
    with run_lock(folder):
        manifest = folder / "manifest.json"
        expected = dict(identity=state.identity, policy=args.policy)
        if not manifest.exists():
            if any(path.name != "run.lock" for path in folder.iterdir()):
                raise ValueError("Nonempty crossover policy directory without manifest")
            atomic_json(manifest, expected)
        (folder / "records").mkdir(exist_ok=True)
        records = validate_saved(args, state, args.policy)
        pending = [(index, row) for index, row in enumerate(state.base.rows) if row["id"] not in records]
        if pending:
            model, tokenizer, length, decoder = make_runtime(args, state)
            try:
                for new_index, (index, row) in enumerate(pending):
                    if args.max_new_examples and new_index >= args.max_new_examples:
                        break
                    position = state.base.positions[row["id"]]
                    record = attach_text(decode_one(row, position, model, length, decoder, args.policy), row, tokenizer)
                    first = args.policy.split("-")[0]
                    validate_record(record, row, position, args.policy, state.endpoints["global-global"][index],
                                    state.endpoints[first + "-" + first][index])
                    atomic_json(folder / "records" / f"{position:05d}.json",
                        dict(identity_sha256=replay.digest(state.identity), record_sha256=replay.digest(record), record=record))
                    records[row["id"]] = record
                    log("context_crossover_progress", policy=args.policy, completed=len(records), expected=replay.EXAMPLES)
            finally:
                decoder.close()
        unchanged(args, state)
        ordered = [records[row["id"]] for row in state.base.rows if row["id"] in records]
        atomic_json(folder / "progress.json", dict(completed=len(ordered), expected=replay.EXAMPLES,
            complete=len(ordered) == replay.EXAMPLES, metrics=corpus_metrics(ordered) if ordered else None))


def run_readout(args, state):
    from additive_chain.iwslt_run import run_lock
    parity_receipt(args, state)
    with run_lock(args.output / "readout-lock"):
        policies = dict(state.endpoints)
        outputs = {}
        for policy in POLICIES:
            records = validate_saved(args, state, policy)
            if len(records) != replay.EXAMPLES:
                raise ValueError("Both hybrid policies must finish every fixed development example")
            ordered = [records[row["id"]] for row in state.base.rows]
            require_same_schedule(ordered, state.endpoints["global-global"])
            policies[policy] = ordered
            outputs[policy] = {str(path.relative_to(args.output)): file_sha256(path)
                              for path in sorted((args.output / policy / "records").glob("*.json"))}
        metrics = {policy: corpus_metrics(records) for policy, records in policies.items()}
        contrasts = {}
        for first in ("global", "contextual"):
            a, b = metrics[first + "-global"], metrics[first + "-contextual"]
            contrasts["add_context_at_second_given_" + first + "_first"] = {
                metric: b[metric] - a[metric] for metric in ("bleu", "chrf")}
        for second in ("global", "contextual"):
            a, b = metrics["global-" + second], metrics["contextual-" + second]
            contrasts["add_context_at_first_given_" + second + "_second"] = {
                metric: b[metric] - a[metric] for metric in ("bleu", "chrf")}
        contrasts["interaction"] = {
            metric: metrics["contextual-contextual"][metric] - metrics["contextual-global"][metric]
                    - metrics["global-contextual"][metric] + metrics["global-global"][metric]
            for metric in ("bleu", "chrf")}
        unchanged(args, state)
        result = dict(identity=state.identity, complete=True, examples=replay.EXAMPLES,
                      metrics=metrics, contrasts=contrasts, outputs=outputs,
                      endpoint_parity_sha256=file_sha256(args.output / "endpoint-parity.json"))
        destination = args.output / "readout.json"
        if destination.exists() and read(destination) != result:
            raise ValueError("Refusing to replace a different complete crossover readout")
        atomic_json(destination, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "readout"))
    for name in ("trained", "data", "length-head", "selection", "global-head", "global-selection",
                 "contextual-selection", "official-evaluation-root", "global-result", "global-readout",
                 "contextual-result", "contextual-readout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--max-new-examples", type=int, default=0)
    args = parser.parse_args()
    if args.command != "readout" and (args.checkpoint is None or args.tokenizer is None):
        parser.error("preflight/run require --checkpoint and --tokenizer")
    state = prepare(args)
    {"preflight": run_preflight, "run": run_policy, "readout": run_readout}[args.command](args, state)


if __name__ == "__main__":
    main()
