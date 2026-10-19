"""Bounded IWSLT pilots and matched conditional decoding with local assets."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import torch
from torch.nn import functional as F

from .backbone import file_sha256
from .generation import ChainDecoder, synchronize
from .translation import (ConditionalMDLM, LengthPredictor, collate_pairs, corrupt_target,
                          load_conditional, translate_one, MASK_ID, EOS_ID)


def log(event, **fields):
    print(json.dumps(dict(event=event, time=time.time(), **fields), allow_nan=False), flush=True)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@contextmanager
def run_lock(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "run.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def dataset(root, split):
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / (split + ".jsonl")
    if file_sha256(path) != manifest["splits"][split]["sha256"]:
        raise ValueError("Dataset changed: " + split)
    return [json.loads(line) for line in path.open()]


@torch.no_grad()
def full_mask_loss(model, rows, device, microbatch=8):
    model.eval()
    numerator, denominator = 0., 0
    for start in range(0, len(rows), microbatch):
        batch = collate_pairs(rows[start:start + microbatch], device)
        tokens = batch.tokens.masked_fill(batch.target, MASK_ID)
        h, c = model.features(tokens, batch.valid)
        positions = batch.target.nonzero()[:, 0]
        logits = model.project(h[batch.target][:, None], c[positions])[:, 0, :MASK_ID]
        numerator += F.cross_entropy(logits, batch.tokens[batch.target], reduction="sum").item()
        denominator += int(batch.target.sum())
    return numerator / denominator


def fit_length(model, rows, device, *, maximum=8192, iterations=600):
    """Fit on frozen source-only features; never use dev/test target lengths."""
    model.eval()
    chosen = random.Random(2718).sample(rows, min(maximum, len(rows)))
    features, labels = [], []
    with torch.no_grad():
        for start in range(0, len(chosen), 16):
            block = chosen[start:start + 16]
            features.append(model.source_features(block, device).cpu())
            labels.extend(len(r["target_ids"]) + 1 for r in block)
    x = torch.cat(features).to(device)
    y = torch.tensor(labels, device=device)
    head = LengthPredictor(model.pretrained.config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator(device=device).manual_seed(2718)
    head.train()
    for _ in range(iterations):
        indices = torch.randint(len(x), (min(256, len(x)),), device=device, generator=generator)
        loss = F.cross_entropy(head(x[indices]), y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
    return head


def score_translations(model, tokenizer, head, rows, decoder, output, *, steps=32, seed=101):
    from sacrebleu.metrics import BLEU, CHRF
    from sacremoses import MosesDetokenizer
    detok = MosesDetokenizer(lang="en")
    hypotheses, references, records = [], [], []
    model.eval()
    head.eval()
    for i, row in enumerate(rows):
        synchronize(decoder.device)
        start = time.perf_counter()
        ids, stats = translate_one(model, row, head, decoder, steps=steps, seed=seed, example_id=i)
        synchronize(decoder.device)
        seconds = time.perf_counter() - start
        text = detok.detokenize(tokenizer.decode(ids).split())
        reference = detok.detokenize(row["target"].split())
        hypotheses.append(text)
        references.append(reference)
        records.append(dict(id=row["id"], hypothesis=text, reference=reference,
                            token_ids=ids, seconds=seconds, **stats))
        if (i + 1) % 32 == 0:
            log("decode_progress", completed=i + 1, total=len(rows))
    bleu = BLEU(lowercase=True, tokenize="13a")
    value = bleu.corpus_score(hypotheses, [references]).score
    summary = dict(bleu=value, bleu_signature=str(bleu.get_signature()),
        chrf=CHRF().corpus_score(hypotheses, [references]).score,
        examples=len(rows), seconds=sum(r["seconds"] for r in records),
        warmup_included=True, timing_is_pilot_not_final_benchmark=True,
        empty_outputs=sum(not h.strip() for h in hypotheses),
        mean_predicted_length=sum(r["predicted_length"] for r in records) / len(records))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(summary=summary, predictions=records), indent=2) + "\n")
    return summary


def gate(args):
    model, tok = load_conditional(args.checkpoint, args.tokenizer, device=args.device, amp=False)
    model.eval()
    rows = [dict(source_ids=tok.encode("das ist ein kleiner test", add_special_tokens=False),
                 target_ids=tok.encode("this is a small test", add_special_tokens=False)),
            dict(source_ids=tok.encode("hallo", add_special_tokens=False),
                 target_ids=tok.encode("hello", add_special_tokens=False))]
    one, padded = collate_pairs(rows[1:], args.device), collate_pairs(rows, args.device)
    with torch.no_grad():
        native = model.pretrained(input_ids=one.tokens,
            timesteps=torch.zeros(1, device=args.device), sample_mode=True)
        h, c = model.features(one.tokens, one.valid)
        ours = model.project(h, c)
        native_error = (native.float() - ours).abs().max().item()
        hp, cp = model.features(padded.tokens, padded.valid)
        padded_logits = model.project(hp, cp)[1:2, :one.tokens.shape[1]]
        padding_error = (ours - padded_logits).abs().max().item()
    if native_error > .001 or padding_error > .002:
        raise AssertionError(("Native/padding parity failed", native_error, padding_error))
    generator = torch.Generator(device=args.device).manual_seed(3)
    noisy, selected, _ = corrupt_target(padded, generator)
    assert torch.equal(noisy[~padded.target], padded.tokens[~padded.target])
    model.amp = True
    model.train()
    loss, count = model.loss_sum(padded, generator, times=torch.ones(len(rows), device=args.device))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert count > 0 and gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    receipt = dict(native_error=native_error, padding_error=padding_error,
        target_only_mask=True, finite_gradients=True, parameters=sum(p.numel() for p in model.parameters()),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        torch=torch.__version__, maximum_gpu_memory=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "passed.json").write_text(json.dumps(receipt, indent=2) + "\n")
    log("gate_passed", **receipt)


def train(args):
    with run_lock(args.output):
        _train(args)


def _train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model, tokenizer = load_conditional(args.checkpoint, args.tokenizer, device=device)
    train_rows = dataset(args.data, "train")
    dev = dataset(args.data, "dev")
    random.Random(31415).shuffle(dev)
    dev = dev[:args.dev_examples]
    if args.smoke:
        train_rows = [r for r in train_rows if len(r["source_ids"]) + len(r["target_ids"]) <= 100][:32]
        dev = train_rows
    if args.batch_size % args.microbatch:
        raise ValueError("Microbatch must divide effective batch size")
    identity = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    identity.update(dataset_sha256=file_sha256(args.data / "manifest.json"),
        code_sha256={name: file_sha256(Path(__file__).parent / name)
                     for name in ("translation.py", "iwslt_run.py", "generation.py")},
        objective="sum(masked CE/t) / all target tokens in effective batch; t uniform[.001,.999]",
        length_training="Refit classifier at validation using source-only features of training pairs; backbone frozen during fitting")
    identity_path = args.output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Training configuration or source changed; use a new run directory")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    generator = torch.Generator(device=device).manual_seed(args.seed + 100)
    order_rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .999), weight_decay=.01)
    step, best, spent, initial = 0, -math.inf, 0., None
    saved = args.output / "latest.pt"
    if saved.exists():
        state = torch.load(saved, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["corruption_rng"].cpu())
        order_rng.setstate(state["order_rng"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        step, best, spent = state["step"], state["best"], state["spent_seconds"]
        initial = state.get("initial_full_mask_nll")
        del state
    started = time.monotonic()
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda signum, frame: stopped.append(signum))
    def save():
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, best=best,
            initial_full_mask_nll=initial,
            spent_seconds=spent + time.monotonic() - started, corruption_rng=generator.get_state(),
            order_rng=order_rng.getstate(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), saved)
    if initial is None:
        initial = full_mask_loss(model, dev[:32], device)
    log("training_started", step=step, initial_full_mask_nll=initial, samples=len(train_rows),
        lr=args.lr, effective_batch=args.batch_size, gpu=torch.cuda.get_device_name(),
        job=os.environ.get("SLURM_JOB_ID"))
    first_step = step
    while step < args.updates and spent + time.monotonic() - started < args.seconds_limit and not stopped:
        model.train()
        chosen = [train_rows[order_rng.randrange(len(train_rows))] for _ in range(args.batch_size)]
        # Length grouping changes only execution order, not the sampled population or denominator.
        chosen.sort(key=lambda r: len(r["source_ids"]) + len(r["target_ids"]))
        denominator = sum(len(r["target_ids"]) + 1 for r in chosen)
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        start_step = time.perf_counter()
        for offset in range(0, len(chosen), args.microbatch):
            batch = collate_pairs(chosen[offset:offset + args.microbatch], device)
            loss, count = model.loss_sum(batch, generator)
            (loss / denominator).backward()
            total += loss.detach().item() / denominator
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        rate = args.lr * min(1., (step + 1) / args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        step += 1
        if step <= first_step + 3 or step % 10 == 0:
            log("train", step=step, nll=total, grad_norm=float(norm), lr=rate,
                seconds=time.perf_counter() - start_step,
                gpu_memory=torch.cuda.max_memory_allocated(), target_tokens=denominator)
        if step % args.save_every == 0:
            save()
        if step % args.validate_every == 0 or step == args.updates:
            with torch.random.fork_rng(devices=[device.index or 0]):
                torch.manual_seed(2718)
                final_loss = full_mask_loss(model, dev[:32], device)
                log("validation_loss", step=step, full_mask_nll=final_loss)
                if args.smoke:
                    if step == args.updates:
                        if final_loss >= .65 * initial:
                            raise RuntimeError("Tiny-data overfit failed: full-mask loss did not fall by 35%")
                        (args.output / "passed.json").write_text(json.dumps(dict(
                            initial_full_mask_nll=initial, final_full_mask_nll=final_loss, updates=step)) + "\n")
                else:
                    head = fit_length(model, train_rows, device)
                    decoder = ChainDecoder(mask_id=MASK_ID, device=device, fast_candidates=True)
                    decoder.prepare()
                    try:
                        score = score_translations(model, tokenizer, head, dev, decoder,
                            args.output / f"dev-{step}.json", steps=32)
                    finally:
                        decoder.close()
                    log("validation", step=step, **score)
                    if score["bleu"] > best:
                        best = score["bleu"]
                        atomic_save(dict(model=model.state_dict(), length_head=head.state_dict(),
                            step=step, dev=score, identity=identity), args.output / "best.pt")
                    del head
            save()
    save()
    summary = dict(step=step, completed=step == args.updates, best_dev_bleu=None if not math.isfinite(best) else best,
                   spent_seconds=spent + time.monotonic() - started, stopped_by_signal=stopped)
    (args.output / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
    log("training_stopped", **summary)


def evaluate(args):
    model, tokenizer = load_conditional(args.checkpoint, args.tokenizer, device=args.device)
    state = torch.load(args.trained, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"])
    head = LengthPredictor(model.pretrained.config.hidden_dim).to(args.device)
    head.load_state_dict(state["length_head"])
    rows = dataset(args.data, args.split)
    if args.limit:
        random.Random(31415).shuffle(rows)
        rows = rows[:args.limit]
    if args.shuffle_sources:
        donors = list(range(len(rows)))
        random.Random(411).shuffle(donors)
        source_for = dict(zip(donors, donors[1:] + donors[:1]))
        rows = [dict(row, source=rows[source_for[i]]["source"],
                     source_ids=rows[source_for[i]]["source_ids"]) for i, row in enumerate(rows)]
    counts = torch.load(args.data / "counts.pt", weights_only=True)
    if file_sha256(args.data / "counts.pt") != json.loads((args.data / "manifest.json").read_text())["counts_sha256"]:
        raise ValueError("Count data changed")
    decoder = ChainDecoder(mask_id=MASK_ID, device=args.device, count_statistics=counts,
        count_strength=args.strength, sampling=args.sampling, fast_candidates=True,
        temperature=args.temperature)
    decoder.prepare()
    try:
        score = score_translations(model, tokenizer, head, rows, decoder, args.output / "predictions.json",
                                   steps=args.steps, seed=args.seed)
        (args.output / "configuration.json").write_text(json.dumps(dict(
            trained_sha256=file_sha256(args.trained), sampling=args.sampling, strength=args.strength,
            temperature=args.temperature, steps=args.steps, seed=args.seed, split=args.split,
            shuffled_sources=args.shuffle_sources,
            dataset_sha256=file_sha256(args.data / "manifest.json")), indent=2) + "\n")
        log("evaluated", **score)
    finally:
        decoder.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("gate", "train", "evaluate"))
    for name in ("checkpoint", "tokenizer", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--lr", type=float, default=.0001)
    p.add_argument("--updates", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--validate-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--dev-examples", type=int, default=512)
    p.add_argument("--seconds-limit", type=int, default=39600)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--trained", type=Path)
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--strength", type=float, default=0.)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--sampling", choices=("joint", "own-marginal"), default="joint")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--shuffle-sources", action="store_true")
    args = p.parse_args()
    if args.action != "gate" and args.data is None:
        p.error("--data is required")
    {"gate": gate, "train": train, "evaluate": evaluate}[args.action](args)


if __name__ == "__main__":
    main()
