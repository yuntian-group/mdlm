"""Two-step contextual midpoint after the IWSLT backbone continuation.

This diagnostic uses contextual strength 0.5 at scheduled time 1 and zero at
time 0.5, retaining count strength 0.2 and global strength 1 throughout. It does
not change the standard ladder or extend this schedule to other step counts.
The same trained contextual checkpoint is used by both sampling methods. If
zero wins the official selection, the diagnostic uses the highest-ranked
trained checkpoint without replacing the official winner.
"""
import argparse
import math
from pathlib import Path
import time
from types import SimpleNamespace

from . import iwslt_backbone_followup as followup
from . import iwslt_context_midpoint as midpoint
from .backbone import file_sha256
from .iwslt_confirm import atomic_json, ordered_rows, read
from .iwslt_controls_readout import verify_population
from .iwslt_followup import verify_output

FORMAT = "iwslt_continued_backbone_midpoint_v1"
SAMPLINGS = ("joint", "own-marginal")
PROTOCOL = dict(steps=2, seed=101, temperature=.85, nucleus_p=.9, top_k=500,
    count_strength=.2, global_strength=1., contextual_strengths=[.5, 0.],
    length_mode="predicted", switching="scheduled_time_not_forward_call_count",
    stream_index="original_full_development_position")


def cells():
    return [cell for cell in followup.cells("contextual") if cell["steps"] == 2]


def source_hashes():
    result = followup.source_hashes()
    for name in ("iwslt_backbone_midpoint.py", "iwslt_context_midpoint.py",
                 "iwslt_context_crossover.py", "iwslt_context_replay.py"):
        result[name] = file_sha256(Path(__file__).with_name(name))
    return result


def choose_trained_context(document, common):
    official = followup.validate_selected(document, common, "contextual")
    if official["update"] > 0:
        return official, official, "official-selected-trained-contextual"
    trained = next((row["candidate"] for row in document["ranking"]
                    if row["candidate"]["update"] > 0), None)
    if trained is None:
        raise ValueError("The midpoint diagnostic requires a trained contextual checkpoint")
    return official, trained, "highest-ranked-trained-contextual-when-zero-wins"


def metrics(records):
    from sacrebleu.metrics import BLEU, CHRF
    hypotheses = [row["hypothesis"] for row in records]
    references = [[row["reference"] for row in records]]
    return dict(bleu=BLEU(lowercase=True, tokenize="13a").corpus_score(hypotheses, references).score,
                chrf=CHRF().corpus_score(hypotheses, references).score)


def prepare(args):
    from sacremoses import MosesDetokenizer
    global_args = SimpleNamespace(**dict(vars(args), stage="global", global_head=None))
    global_common = followup.common_assets(global_args)
    global_selection = read(args.global_selection)
    global_candidate = followup.validate_selected(global_selection, global_common, "global")
    global_path = Path(global_candidate["head"])
    _, _, global_saved = followup.evaluation.load_heads(global_path, "global", global_common)
    if followup.evaluation.candidate_entry(global_path, global_saved) != global_candidate:
        raise ValueError("Selected global checkpoint changed")
    context_args = SimpleNamespace(**dict(vars(args), stage="contextual", global_head=global_path))
    common = followup.common_assets(context_args)
    official, candidate, policy = choose_trained_context(read(args.contextual_selection), common)
    head, parent, saved = followup.evaluation.load_heads(Path(candidate["head"]), "contextual", common, global_path)
    if followup.evaluation.candidate_entry(Path(candidate["head"]), saved) != candidate:
        raise ValueError("Chosen trained contextual checkpoint changed")
    rows = ordered_rows(args.data, "dev")
    selection_sha = file_sha256(args.global_selection)
    readout = read(args.global_readout)
    if (readout.get("split") != "dev" or readout.get("stage") != "global"
            or readout.get("examples") != len(rows) or readout.get("common") != global_common
            or readout.get("selected") != global_candidate
            or readout.get("residual_selection_sha256") != selection_sha
            or readout.get("source_sha256") != followup.source_hashes()):
        raise ValueError("Global endpoint readout belongs to a different comparison")
    expected = {str((followup.folder_for(args.evaluation_root, "global", global_candidate,
                         "full", cell) / "result.json").relative_to(args.evaluation_root))
                for cell in followup.cells("global")}
    if set(readout["outputs"]) != expected:
        raise ValueError("Incomplete global full-development readout")
    detok = MosesDetokenizer(lang="en")
    references = [detok.detokenize(row["target"].split()) for row in rows]
    endpoints, files = {}, {}
    for cell in cells():
        path = followup.folder_for(args.evaluation_root, "global", global_candidate, "full", cell) / "result.json"
        relative = str(path.relative_to(args.evaluation_root))
        if file_sha256(path) != readout["outputs"][relative]:
            raise ValueError("Global endpoint output differs from its readout")
        identity = followup.identity(cell, rows, "global", "full", global_common, global_candidate, selection_sha)
        payload = verify_output(path, identity)
        summary = read(path.with_name("summary.json"))
        if summary != dict(completed=1, expected=1, cell=cell, **payload["summary"]):
            raise ValueError("Incomplete global endpoint summary")
        records = payload["predictions"]
        verify_population(records, rows, references, 2)
        observed = metrics(records)
        for key, value in observed.items():
            reported = (payload["summary"][key], readout["steps"]["2"]["methods"][cell["sampling"]][key])
            if any(not math.isfinite(score) or abs(score - value) > 1e-10 for score in reported):
                raise ValueError("Global endpoint metrics cannot be reproduced")
        endpoints[cell["sampling"]] = records
        for item in (path, path.with_name("summary.json"), path.with_name("result-identity.json")):
            files[str(item)] = file_sha256(item)
    followup.require_same_schedule(endpoints["joint"], endpoints["own-marginal"])
    for path in (args.trained, args.length_head, args.protocol, args.global_selection,
                 args.contextual_selection, args.global_readout, global_path, Path(candidate["head"])):
        files[str(path)] = file_sha256(path)
    identity = dict(format=FORMAT, split="dev", role="diagnostic", protocol=PROTOCOL,
        common=common, global_candidate=global_candidate, official_contextual=official,
        trained_contextual=candidate, contextual_choice_policy=policy,
        examples=len(rows), ids=[r["id"] for r in rows], source_sha256=source_hashes(),
        input_sha256=files)
    return SimpleNamespace(identity=identity, rows=rows, endpoints=endpoints,
        head=head, global_head=parent, common=common, files=files,
        context_args=context_args)


def unchanged(args, state):
    if (followup.common_assets(state.context_args) != state.common
            or source_hashes() != state.identity["source_sha256"]
            or any(file_sha256(Path(path)) != sha for path, sha in state.files.items())):
        raise ValueError("Midpoint inputs or source changed")


def cell_identity(state, sampling):
    if sampling not in SAMPLINGS:
        raise ValueError("Unknown midpoint sampling method")
    return dict(state.identity, sampling=sampling)


def validate_records(records, state, sampling):
    if len(records) != len(state.rows):
        raise ValueError("Incomplete midpoint development population")
    endpoints = state.endpoints[sampling]
    for index, (record, row, endpoint) in enumerate(zip(records, state.rows, endpoints)):
        if record.get("sampling") != sampling:
            raise ValueError("Midpoint sampling identity changed")
        midpoint.validate_record(record, row, index, .5, endpoint)
        if (not math.isfinite(record.get("seconds", math.nan)) or record["seconds"] < 0
                or record.get("same_state_probe_seconds") != 0.):
            raise ValueError("Invalid midpoint timing/probe record")
    verify_population(records, state.rows, [row["reference"] for row in endpoints], 2)
    return metrics(records)


def parity_receipt(state, sampling):
    endpoints = state.endpoints[sampling][:4]
    return dict(identity=cell_identity(state, sampling), exact_endpoint_parity=True,
        checks=[dict(id=row["id"], token_sha256=midpoint.replay.digest(row["token_ids"]),
                     schedule_sha256=midpoint.replay.digest(row["schedule"])) for row in endpoints])


def check_endpoint_parity(model, length, decoder, state, sampling):
    """The switching adapter at strength zero must reproduce its global endpoint."""
    if decoder.sampling != sampling:
        raise ValueError("Midpoint sampler differs from its endpoint parity control")
    keys = ("token_ids", "predicted_length", "length_prediction_forwards", "scheduled_steps",
            "executed_steps", "actual_forward_calls", "reveal_counts", "schedule", "schedule_name", "opportunities")
    for index, (row, endpoint) in enumerate(zip(state.rows[:4], state.endpoints[sampling][:4])):
        observed = midpoint.decode_one(row, index, model, length, decoder, 0.)
        if any(observed[key] != endpoint[key] for key in keys):
            raise ValueError("Zero-context switching did not reproduce the new global endpoint")
    return parity_receipt(state, sampling)


def validate_output(args, state, sampling):
    path = args.output / sampling / "result.json"
    if read(path.with_name("endpoint-parity.json")) != parity_receipt(state, sampling):
        raise ValueError("Missing or changed midpoint endpoint parity check")
    payload = verify_output(path, cell_identity(state, sampling))
    observed = validate_records(payload["predictions"], state, sampling)
    for metric, value in observed.items():
        score = payload["summary"][metric]
        if not math.isfinite(score) or abs(score - value) > 1e-10:
            raise ValueError("Midpoint quality score cannot be reproduced")
    if (payload["summary"].get("role") != "diagnostic"
            or payload["summary"].get("same_state_probability_probes") is not False
            or payload["summary"].get("same_state_probe_seconds") != 0.
            or payload["summary"].get("seconds") != sum(r["seconds"] for r in payload["predictions"])):
        raise ValueError("Invalid midpoint summary")
    return payload


def run_score(args):
    import torch
    from sacremoses import MosesDetokenizer
    from .generation import ChainDecoder, synchronize
    from .iwslt_length_refit import load_refit_head
    from .iwslt_run import log, run_lock
    from .translation import MASK_ID, load_conditional
    state = prepare(args)
    if type(args.cell_index) is not int or not 0 <= args.cell_index < len(SAMPLINGS):
        raise ValueError("Midpoint has only two registered sampling cells")
    sampling = SAMPLINGS[args.cell_index]
    folder = args.output / sampling
    with run_lock(folder):
        path = folder / "result.json"
        if path.with_name("result-identity.json").exists():
            validate_output(args, state, sampling)
            return
        if path.exists():
            raise ValueError("Midpoint result without completion identity")
        model, tokenizer = load_conditional(args.checkpoint, args.tokenizer)
        saved = torch.load(args.trained, map_location="cuda", weights_only=False)
        model.load_state_dict(saved["model"])
        del saved
        model.eval().requires_grad_(False)
        length = load_refit_head(args.length_head, state.common["model_sha256"], state.common["data_sha256"], "cuda")
        decoder = ChainDecoder(mask_id=MASK_ID, device="cuda", top_k=500, nucleus_p=.9,
            fast_candidates=True, temperature=.85, count_strength=.2, sampling=sampling,
            count_statistics=torch.load(args.data / "counts.pt", weights_only=True),
            global_head=state.global_head.cuda(), global_strength=1.,
            contextual_head=state.head.cuda(), contextual_strength=1.)
        records, detok = [], MosesDetokenizer(lang="en")
        decoder.prepare()
        try:
            parity = check_endpoint_parity(model, length, decoder, state, sampling)
            followup.immutable_json(folder / "endpoint-parity.json", parity)
            for index, row in enumerate(state.rows):
                synchronize(decoder.device)
                start = time.perf_counter()
                record = midpoint.decode_one(row, index, model, length, decoder, .5)
                synchronize(decoder.device)
                record.update(sampling=sampling, seconds=time.perf_counter() - start,
                    same_state_probe_seconds=0., hypothesis=detok.detokenize(tokenizer.decode(record["token_ids"]).split()),
                    reference=detok.detokenize(row["target"].split()))
                records.append(record)
                if (index + 1) % 32 == 0:
                    log("backbone_midpoint_progress", sampling=sampling, completed=index + 1, total=len(state.rows))
        finally:
            decoder.close()
        measured = validate_records(records, state, sampling)
        unchanged(args, state)
        summary = dict(examples=len(records), role="diagnostic", **measured,
            seconds=sum(r["seconds"] for r in records), same_state_probe_seconds=0.,
            same_state_probability_probes=False, timing_includes_diagnostics=True)
        atomic_json(path, dict(summary=summary, predictions=records))
        atomic_json(path.with_name("result-identity.json"),
                    dict(identity=cell_identity(state, sampling), output_sha256=file_sha256(path)))
        log("backbone_midpoint_complete", sampling=sampling, **summary)


def run_readout(args):
    state = prepare(args)
    payloads = {sampling: validate_output(args, state, sampling) for sampling in SAMPLINGS}
    left, right = (payloads[s]["predictions"] for s in SAMPLINGS)
    followup.require_same_schedule(left, right)
    if any([(s["scheduled_step"], s["selected_positions"]) for s in a["intermediate_states"]]
           != [(s["scheduled_step"], s["selected_positions"]) for s in b["intermediate_states"]]
           for a, b in zip(left, right)):
        raise ValueError("Midpoint samplers differ in revealed positions")
    result = dict(identity=state.identity, split="dev", role="diagnostic", complete=True,
        examples=len(state.rows), methods={s: metrics(payloads[s]["predictions"]) for s in SAMPLINGS},
        global_endpoints={s: metrics(state.endpoints[s]) for s in SAMPLINGS},
        outputs={s: file_sha256(args.output / s / "result.json") for s in SAMPLINGS})
    result["joint_minus_own_marginal"] = {key: result["methods"]["joint"][key] - result["methods"]["own-marginal"][key]
                                         for key in ("bleu", "chrf")}
    result["midpoint_minus_global"] = {s: {key: result["methods"][s][key] - result["global_endpoints"][s][key]
                                         for key in ("bleu", "chrf")} for s in SAMPLINGS}
    unchanged(args, state)
    followup.immutable_json(args.output / "readout.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "readout"))
    for name in ("trained", "data", "length-head", "protocol", "global-selection",
                 "contextual-selection", "global-readout", "evaluation-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--cell-index", type=int)
    args = parser.parse_args()
    if args.command == "score" and (args.checkpoint is None or args.tokenizer is None):
        parser.error("score requires --checkpoint and --tokenizer")
    {"score": run_score, "readout": run_readout}[args.command](args)


if __name__ == "__main__":
    main()
