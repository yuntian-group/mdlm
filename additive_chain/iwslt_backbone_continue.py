"""Continue the IWSLT backbone, stopping on full-development decoding BLEU.

The pilot optimizer and random streams are resumed, not restarted. Validation
fits a new source-only length head on the complete training split and decodes
the complete development split. Neither selection nor training reads test data.
"""
import argparse
import ast
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import torch

from .backbone import file_sha256
from .generation import ChainDecoder, synchronize
from .iwslt_confirm import atomic_json, ordered_rows
from . import iwslt_length_refit as refit
from .iwslt_run import atomic_save, log, run_lock
from .translation import LengthPredictor, MASK_ID, collate_pairs, load_conditional, translate_one

FORMAT = "iwslt_backbone_bleu_continuation_v1"
PILOT = dict(step=10000, lr=1e-4, batch_size=128, microbatch=8, warmup=1000, seed=11)
VALIDATION = dict(steps=32, seed=101, temperature=.85, nucleus_p=.9, top_k=500,
                  validate_every=5000, patience=4, min_delta=.1,
                  length_max_epochs=30, length_patience=5)
SOURCE_FILES = ("iwslt_backbone_continue.py", "iwslt_length_refit.py", "iwslt_run.py",
                "iwslt_confirm.py", "translation.py", "generation.py", "backbone.py",
                "chain.py", "counts.py", "candidates.py")


def source_hashes():
    return {name: file_sha256(Path(__file__).parent / name) for name in SOURCE_FILES}


def initial_plateau():
    return dict(best_bleu=None, best_step=None, anchor_bleu=None, bad_checks=0,
                last_validation_step=None, plateau=False, history=[])


def update_plateau(state, step, score):
    """Keep every true best, but reset patience only on cumulative >=0.1 gains."""
    bleu = score["bleu"]
    if not math.isfinite(bleu) or not 0 <= bleu <= 100:
        raise ValueError("Invalid development BLEU")
    if state["last_validation_step"] is not None and step <= state["last_validation_step"]:
        raise ValueError("Development checkpoints must be applied exactly once, in order")
    best = state["best_bleu"] is None or bleu > state["best_bleu"]
    if best:
        state.update(best_bleu=bleu, best_step=step)
    if state["anchor_bleu"] is None or bleu - state["anchor_bleu"] >= VALIDATION["min_delta"] - 1e-12:
        state.update(anchor_bleu=bleu, bad_checks=0)
    else:
        state["bad_checks"] += 1
    state.update(last_validation_step=step, plateau=state["bad_checks"] >= VALIDATION["patience"])
    state["history"].append(dict(step=step, **score))
    return best


def validation_due(step, state):
    """A saved boundary with no committed score must be validated before training."""
    return (step >= PILOT["step"] and (step - PILOT["step"]) % VALIDATION["validate_every"] == 0
            and state["last_validation_step"] != step)


def validate_pilot_identity(identity, data_sha, source_code):
    for key in ("lr", "batch_size", "microbatch", "warmup", "seed"):
        if identity.get(key) != PILOT[key]:
            raise ValueError("Pilot training configuration differs: " + key)
    if identity.get("smoke") is not False or identity.get("dataset_sha256") != data_sha:
        raise ValueError("Pilot dataset or training mode differs")
    expected = identity.get("code_sha256", {})
    if set(expected) != {"translation.py", "iwslt_run.py", "generation.py"}:
        raise ValueError("Missing pilot source identities")
    for name, digest in expected.items():
        if file_sha256(source_code / "additive_chain" / name) != digest:
            raise ValueError("Pilot source snapshot changed: " + name)
    # Generation has evolved since the pilot. Authenticate the original source
    # and separately require that the training adapter itself has not changed.
    names = {"collate_pairs", "corrupt_target", "ConditionalMDLM"}
    def training_ast(path):
        tree = ast.parse(path.read_text())
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}
    old = training_ast(source_code / "additive_chain/translation.py")
    current = training_ast(Path(__file__).parent / "translation.py")
    if set(old) != names or old != current:
        raise ValueError("Training adapter semantics changed since the pilot")


def validate_training_state(state, *, initial=False):
    required = {"model", "optimizer", "step", "corruption_rng", "order_rng", "cpu_rng", "cuda_rng"}
    if not required <= state.keys() or type(state["step"]) is not int:
        raise ValueError("Incomplete resumable training checkpoint")
    if state["step"] < PILOT["step"] or (initial and state["step"] != PILOT["step"]):
        raise ValueError("Unexpected pilot or continuation update")
    groups = state["optimizer"].get("param_groups", [])
    if not groups or not state["optimizer"].get("state"):
        raise ValueError("Missing saved Adam state")
    for group in groups:
        if (group["lr"] != PILOT["lr"] or tuple(group["betas"]) != (.9, .999)
                or group["weight_decay"] != .01):
            raise ValueError("Saved Adam configuration differs")
    for entry in state["optimizer"]["state"].values():
        if not {"step", "exp_avg", "exp_avg_sq"} <= entry.keys():
            raise ValueError("Incomplete saved Adam moments")
        if any(not bool(torch.isfinite(entry[key]).all()) for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Nonfinite saved Adam moments")
    if any(not bool(torch.isfinite(value).all()) for value in state["model"].values()):
        raise ValueError("Nonfinite saved backbone")


def restore_training_state(state, model, optimizer, corruption, order, device):
    """Call after constructing the model/optimizer, which may consume RNG state."""
    validate_training_state(state)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    for parameter, moments in optimizer.state.items():
        if any(moments[key].shape != parameter.shape for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Saved Adam moment shape differs from its parameter")
    corruption.set_state(state["corruption_rng"].cpu())
    order.setstate(state["order_rng"])
    torch.set_rng_state(state["cpu_rng"].cpu())
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device=device)


@contextmanager
def validation_rng(device):
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(2718)
        yield


def _read_progress(path, rows, references):
    """Recover a newline-committed prefix; only a torn final write is discarded."""
    if not path.exists():
        return []
    records, committed = [], 0
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            record = json.loads(line)
            i = len(records)
            if (i >= len(rows) or record.get("id") != rows[i]["id"]
                    or record.get("reference") != references[i]
                    or not isinstance(record.get("hypothesis"), str)
                    or not isinstance(record.get("token_ids"), list)
                    or not math.isfinite(record.get("seconds", math.nan))
                    or record["seconds"] < 0):
                raise ValueError("Development prediction prefix differs")
            records.append(record)
            committed = stream.tell()
    if path.stat().st_size != committed:
        with path.open("r+b") as stream:
            stream.truncate(committed)
    return records


def _score_records(records):
    from sacrebleu.metrics import BLEU, CHRF
    hypotheses, references = [r["hypothesis"] for r in records], [r["reference"] for r in records]
    bleu = BLEU(lowercase=True, tokenize="13a")
    score = bleu.corpus_score(hypotheses, [references]).score
    return dict(bleu=score, bleu_signature=str(bleu.get_signature()),
        chrf=CHRF().corpus_score(hypotheses, [references]).score, examples=len(records),
        seconds=sum(r["seconds"] for r in records),
        empty_outputs=sum(not h.strip() for h in hypotheses),
        mean_predicted_length=sum(r["predicted_length"] for r in records) / len(records))


def score_resumable(model, tokenizer, head, rows, root, identity, device, stopped):
    from sacremoses import MosesDetokenizer
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "decode-identity.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Development decoder identity changed")
    if not marker.exists() and (root / "predictions.jsonl").exists():
        raise ValueError("Unidentified development predictions")
    atomic_json(marker, identity)
    detok = MosesDetokenizer(lang="en")
    references = [detok.detokenize(row["target"].split()) for row in rows]
    records = _read_progress(root / "predictions.jsonl", rows, references)
    model.eval()
    head.eval()
    decoder = ChainDecoder(mask_id=MASK_ID, device=device, fast_candidates=True,
        temperature=VALIDATION["temperature"], top_k=VALIDATION["top_k"], nucleus_p=VALIDATION["nucleus_p"])
    decoder.prepare()
    try:
        with (root / "predictions.jsonl").open("a") as stream:
            for i in range(len(records), len(rows)):
                if stopped():
                    raise refit.Interrupted("Development decoding interrupted after a complete example")
                synchronize(device)
                started = time.perf_counter()
                ids, stats = translate_one(model, rows[i], head, decoder,
                    steps=VALIDATION["steps"], seed=VALIDATION["seed"], example_id=i)
                synchronize(device)
                record = dict(id=rows[i]["id"], hypothesis=detok.detokenize(tokenizer.decode(ids).split()),
                    reference=references[i], token_ids=ids, seconds=time.perf_counter() - started, **stats)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                records.append(record)
                if (i + 1) % 32 == 0:
                    log("backbone_decode_progress", step=identity["step"], completed=i + 1, total=len(rows))
    finally:
        decoder.close()
    summary = _score_records(records)
    atomic_json(root / "dev.json", dict(identity=identity, summary=summary, predictions=records))
    return summary


def run_validation(model, tokenizer, train_rows, dev_rows, decode_rows, old_head_state,
                   step, root, identity, device, stopped):
    root.mkdir(parents=True, exist_ok=True)
    model.eval()
    digest = refit._state_digest(model.state_dict())
    model_identity = dict(backbone_state_sha256=digest, step=step,
        data_sha256=identity["dataset_sha256"], source_sha256=refit.source_hashes(),
        continuation_source_sha256=identity["source_sha256"])
    marker = root / "identity.json"
    if marker.exists() and json.loads(marker.read_text()) != model_identity:
        raise ValueError("Validation checkpoint or feature identity changed")
    atomic_json(marker, model_identity)
    # A stable file identity lets the existing length loader authenticate exports.
    translator = root / "translator.pt"
    if not translator.exists():
        atomic_save(dict(model=model.state_dict(), length_head=old_head_state,
                         step=step, identity=model_identity), translator)
    else:
        saved = torch.load(translator, map_location="cpu", weights_only=False)
        if saved.get("identity") != model_identity or refit._state_digest(saved["model"]) != digest:
            raise ValueError("Validation translator changed")
        del saved
    feature_identity = dict(model_identity, model_sha256=file_sha256(translator),
                            old_length_head_sha256=refit._state_digest(old_head_state))
    hidden = model.pretrained.config.hidden_dim
    old_head = LengthPredictor(hidden).to(device)
    old_head.load_state_dict(old_head_state)
    cache = {split: refit.cache_features(model, rows, split, root / "length/cache",
        feature_identity, hidden_size=hidden, device=device, stopped=stopped)
        for split, rows in (("train", train_rows), ("dev", dev_rows))}
    refit.fit_cached(cache["train"], cache["dev"], old_head, feature_identity, root / "length",
        hidden_size=hidden, device=device, max_epochs=VALIDATION["length_max_epochs"],
        patience=VALIDATION["length_patience"], stopped=stopped)
    del cache, old_head
    length_path = root / "length/best.pt"
    head = refit.load_refit_head(length_path, feature_identity["model_sha256"],
                                identity["dataset_sha256"], device)
    decode_identity = dict(model_identity, length_head_sha256=file_sha256(length_path),
        decoding=VALIDATION, ids=[row["id"] for row in decode_rows])
    score = score_resumable(model, tokenizer, head, decode_rows, root, decode_identity, device, stopped)
    receipt = dict(identity=decode_identity, dev_sha256=file_sha256(root / "dev.json"),
                   length_head_sha256=file_sha256(length_path), summary=score)
    atomic_json(root / "complete.json", receipt)
    return score, refit._cpu_state(head), length_path


def export_best(model, head_state, step, score, identity, length_path, output):
    """Write the selected translator and a length-only artifact bound to its SHA."""
    length = torch.load(length_path, map_location="cpu", weights_only=True)
    if refit._state_digest(length["length_head"]) != refit._state_digest(head_state):
        raise ValueError("Selected length head differs from its fitted artifact")
    atomic_save(dict(model=model.state_dict(), length_head=head_state, step=step,
        dev=score, identity=identity, length_refit=str(length_path)), output / "best.pt")
    length["identity"] = dict(length["identity"], model_sha256=file_sha256(output / "best.pt"))
    atomic_save(length, output / "best-length.pt")
    atomic_json(output / "selected.json", dict(step=step, dev=score,
        backbone_sha256=file_sha256(output / "best.pt"), length_sha256=file_sha256(output / "best-length.pt")))


def build_identity(args):
    pilot = json.loads(args.source_identity.read_text())
    data_sha = file_sha256(args.data / "manifest.json")
    validate_pilot_identity(pilot, data_sha, args.source_code)
    if args.output.resolve() == args.source_latest.parent.resolve():
        raise ValueError("Continuation must not overwrite the pilot directory")
    return dict(format=FORMAT, dataset_sha256=data_sha, source_sha256=source_hashes(),
        source_latest_sha256=file_sha256(args.source_latest),
        source_best_sha256=file_sha256(args.source_best),
        source_identity_sha256=file_sha256(args.source_identity),
        pilot=pilot, training=PILOT, validation=VALIDATION,
        checkpoint=str(args.checkpoint.resolve()), tokenizer=str(args.tokenizer.resolve()),
        data=str(args.data.resolve()), source_code=str(args.source_code.resolve()),
        objective="Continue pilot AdamW and random streams; masked target CE/t over all target tokens")


def run(args):
    with run_lock(args.output):
        _run(args)


def _run(args):
    if not math.isfinite(args.seconds_limit) or args.seconds_limit <= 0 or args.updates_per_chunk < 0 or args.save_every < 1:
        raise ValueError("Time/checkpoint limits must be positive; update cap must be nonnegative")
    device = torch.device(args.device)
    identity = build_identity(args)
    path = args.output / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Continuation source, model, data, or protocol changed")
    atomic_json(path, identity)
    train_rows = refit.registered_rows(args.data, "train")
    dev_rows = refit.registered_rows(args.data, "dev")
    # The same fixed shuffle is essential: proposal RNG is keyed by row index.
    decode_rows = ordered_rows(args.data, "dev")
    model, tokenizer = load_conditional(args.checkpoint, args.tokenizer, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PILOT["lr"], betas=(.9, .999), weight_decay=.01)
    corruption = torch.Generator(device=device)
    order = random.Random()
    latest = args.output / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved.get("format") != FORMAT or saved.get("identity") != identity:
            raise ValueError("Resumed continuation identity differs")
        plateau = saved["plateau"]
        old_head = saved["last_length_head"]
        elapsed = saved["spent_seconds"]
    else:
        saved = torch.load(args.source_latest, map_location="cpu", weights_only=False)
        validate_training_state(saved, initial=True)
        pilot_best = torch.load(args.source_best, map_location="cpu", weights_only=False)
        if (pilot_best.get("step") != saved["step"]
                or pilot_best.get("identity") != identity["pilot"]
                or refit._state_digest(pilot_best["model"]) != refit._state_digest(saved["model"])):
            raise ValueError("Pilot best and latest backbone do not match")
        old_head = pilot_best["length_head"]
        del pilot_best
        plateau, elapsed = initial_plateau(), 0.
    restore_training_state(saved, model, optimizer, corruption, order, device)
    step, first_step = saved["step"], saved["step"]
    del saved
    started, stopped, previous = time.monotonic(), [], {}
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        previous[sig] = signal.signal(sig, lambda signum, frame: stopped.append(signum))

    def should_stop():
        return bool(stopped) or time.monotonic() - started >= args.seconds_limit

    def save():
        atomic_save(dict(format=FORMAT, identity=identity, model=model.state_dict(),
            optimizer=optimizer.state_dict(), step=step, plateau=plateau,
            last_length_head=old_head, spent_seconds=elapsed + time.monotonic() - started,
            corruption_rng=corruption.get_state(), order_rng=order.getstate(),
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None), latest)

    interrupted = False
    try:
        # Save before any validation or update, including the initial pilot check.
        save()
        log("backbone_continuation_started", step=step, training_examples=len(train_rows),
            development_examples=len(dev_rows), job=os.environ.get("SLURM_JOB_ID"),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
        while not plateau["plateau"]:
            if should_stop():
                break
            if validation_due(step, plateau):
                save()
                atomic_json(args.output / "status.json", dict(step=step, phase="validation", plateau=plateau,
                    completed=False, development_examples=len(dev_rows)))
                with validation_rng(device):
                    score, head_state, length_path = run_validation(model, tokenizer, train_rows, dev_rows,
                        decode_rows, old_head, step, args.output / f"validation-{step:07d}",
                        identity, device, should_stop)
                    improved = update_plateau(plateau, step, score)
                    old_head = head_state
                    if improved:
                        export_best(model, head_state, step, score, identity, length_path, args.output)
                save()
                log("backbone_validation", step=step, **score, bad_checks=plateau["bad_checks"],
                    best_step=plateau["best_step"], plateau=plateau["plateau"])
                if plateau["plateau"]:
                    break
            if should_stop() or (args.updates_per_chunk and step - first_step >= args.updates_per_chunk):
                break
            model.train()
            chosen = [train_rows[order.randrange(len(train_rows))] for _ in range(PILOT["batch_size"])]
            chosen.sort(key=lambda row: len(row["source_ids"]) + len(row["target_ids"]))
            denominator = sum(len(row["target_ids"]) + 1 for row in chosen)
            optimizer.zero_grad(set_to_none=True)
            total, update_started = 0., time.perf_counter()
            for offset in range(0, len(chosen), PILOT["microbatch"]):
                batch = collate_pairs(chosen[offset:offset + PILOT["microbatch"]], device)
                loss, _ = model.loss_sum(batch, corruption)
                (loss / denominator).backward()
                total += loss.detach().item() / denominator
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step <= first_step + 3 or step % 10 == 0:
                log("backbone_train", step=step, nll=total, grad_norm=float(norm), lr=PILOT["lr"],
                    seconds=time.perf_counter() - update_started, target_tokens=denominator)
            if step % args.save_every == 0:
                save()
        save()
    except refit.Interrupted as error:
        interrupted = True
        # fork_rng has restored training randomness before this checkpoint.
        save()
        log("backbone_validation_interrupted", step=step, message=str(error))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    reason = ("dev_bleu_plateau" if plateau["plateau"] else "signal" if stopped else
              "time_limit" if time.monotonic() - started >= args.seconds_limit else "chunk_limit")
    summary = dict(format=FORMAT, step=step, completed=plateau["plateau"], stopping_reason=reason,
        validation_pending=validation_due(step, plateau), interrupted_validation=interrupted,
        plateau=plateau, spent_seconds=elapsed + time.monotonic() - started, stopped_by_signal=stopped)
    atomic_json(args.output / "status.json", summary)
    log("backbone_continuation_stopped", **summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "data", "source-latest", "source-best",
                 "source-identity", "source-code", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds-limit", type=float, default=36000)
    parser.add_argument("--max-new-updates", "--updates-per-chunk", dest="updates_per_chunk", type=int,
                        default=0, help="Optional per-invocation cap; 0 continues until plateau or wall time")
    parser.add_argument("--save-every", type=int, default=250)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
