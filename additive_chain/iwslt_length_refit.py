"""Refit the IWSLT length classifier on all training sources, with dev selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.nn import functional as F

from .backbone import file_sha256
from .iwslt_confirm import atomic_json
from .iwslt_run import atomic_save, dataset, log, run_lock
from .translation import CONTEXT_LENGTH, LengthPredictor, load_conditional

SPLIT_COUNTS = {"train": 160239, "dev": 7283}
SOURCE_FILES = ("iwslt_length_refit.py", "translation.py", "iwslt_run.py", "backbone.py")
FORMAT = "iwslt_full_train_length_head_v1"
TRAINING = dict(seed=2718, learning_rate=.001, weight_decay=.01, batch_size=256)


class Interrupted(Exception):
    """The last complete feature shard or training epoch is ready to resume."""


def source_hashes():
    return {name: file_sha256(Path(__file__).parent / name) for name in SOURCE_FILES}


def registered_rows(root, split):
    if split not in SPLIT_COUNTS:
        raise ValueError("Length refitting reads train/dev only")
    rows = dataset(root, split)
    if len(rows) != SPLIT_COUNTS[split] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected the complete registered " + split + " split")
    for row in rows:
        maximum = CONTEXT_LENGTH - len(row["source_ids"]) - 2
        if not row["source_ids"] or not 1 <= len(row["target_ids"]) + 1 <= maximum:
            raise ValueError("Invalid source/target length")
    return rows


def _row_digest(rows):
    value = [(r["id"], r["source_ids"], len(r["target_ids"]) + 1) for r in rows]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps((name, str(value.dtype), list(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def feature_batches(rows, *, batch_size=256, token_budget=16384):
    """Source-length grouping bounded by actual padded source-token count."""
    if batch_size < 1 or token_budget < 1:
        raise ValueError("Invalid feature extraction batch limits")
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["source_ids"]))
    batches, pending = [], []
    for i in order:
        length = len(rows[i]["source_ids"]) + 2
        if length > token_budget:
            raise ValueError("A source exceeds the feature token budget")
        if pending and (len(pending) >= batch_size or (len(pending) + 1) * length > token_budget):
            batches.append(pending)
            pending = []
        pending.append(i)
    if pending:
        batches.append(pending)
    return batches


@torch.no_grad()
def check_feature_batching(model, old_head, rows, device):
    """Measure padding/batch effects on frozen features and old-head predictions."""
    ordered = sorted(rows, key=lambda row: len(row["source_ids"]))
    count = min(8, len(ordered))
    chosen = [ordered[i * (len(ordered) - 1) // max(1, count - 1)] for i in range(count)]
    sources = [dict(source_ids=row["source_ids"]) for row in chosen]
    model.eval()
    old_head.eval()
    batched = model.source_features(sources, device)
    single = torch.cat([model.source_features([source], device) for source in sources])
    if not bool(torch.isfinite(batched).all() and torch.isfinite(single).all()):
        raise ValueError("Nonfinite frozen source features")
    batch_predictions, single_predictions = old_head(batched).argmax(-1), old_head(single).argmax(-1)
    return dict(ids=[row["id"] for row in chosen], source_lengths=[len(row["source_ids"]) for row in chosen],
        feature_max_abs_difference=(batched - single).abs().max().item(),
        feature_mean_abs_difference=(batched - single).abs().mean().item(),
        old_head_batched_predictions=batch_predictions.cpu().tolist(),
        old_head_single_predictions=single_predictions.cpu().tolist(),
        old_head_prediction_agreement=(batch_predictions == single_predictions).float().mean().item())


def _validate_features(payload, identity, rows, hidden_size):
    if payload.get("identity") != identity:
        raise ValueError("Feature cache identity changed")
    x, y, maximum = payload["features"], payload["labels"], payload["maximum"]
    labels = torch.tensor([len(r["target_ids"]) + 1 for r in rows])
    limits = torch.tensor([CONTEXT_LENGTH - len(r["source_ids"]) - 2 for r in rows])
    if (x.shape != (len(rows), hidden_size + 1) or x.dtype != torch.float32
            or not bool(torch.isfinite(x).all()) or y.dtype != torch.long
            or maximum.dtype != torch.long or not torch.equal(y, labels)
            or not torch.equal(maximum, limits)):
        raise ValueError("Feature cache values or row order changed")


@torch.no_grad()
def cache_features(model, rows, split, root, identity, *, hidden_size,
                   device, batch_size=256, token_budget=16384, shard_size=4096, stopped=lambda: False):
    """Cache frozen source-only features in atomically written, resumable shards."""
    if split not in SPLIT_COUNTS or batch_size < 1 or shard_size < 1:
        raise ValueError("Invalid feature extraction settings")
    root.mkdir(parents=True, exist_ok=True)
    model.eval()
    parts = []
    for start in range(0, len(rows), shard_size):
        block = rows[start:start + shard_size]
        shard_identity = dict(identity, split=split, start=start, examples=len(block),
            rows_sha256=_row_digest(block), hidden_size=hidden_size,
            feature_batch_size=batch_size, feature_token_budget=token_budget)
        path = root / f"{split}-{start:06d}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            _validate_features(payload, shard_identity, block, hidden_size)
        else:
            # Group execution by source length, then restore the original row order.
            x = torch.empty((len(block), hidden_size + 1), dtype=torch.float32)
            for indices in feature_batches(block, batch_size=batch_size, token_budget=token_budget):
                sources = [dict(source_ids=block[i]["source_ids"]) for i in indices]
                features = model.source_features(sources, device).float().cpu()
                x[indices] = features
            payload = dict(identity=shard_identity, features=x,
                labels=torch.tensor([len(r["target_ids"]) + 1 for r in block]),
                maximum=torch.tensor([CONTEXT_LENGTH - len(r["source_ids"]) - 2 for r in block]))
            _validate_features(payload, shard_identity, block, hidden_size)
            atomic_save(payload, path)
        parts.append(payload)
        log("length_features", split=split, completed=start + len(block), total=len(rows))
        if stopped():
            raise Interrupted("Feature extraction interrupted after a complete shard")
    return {key: torch.cat([part[key] for part in parts])
            for key in ("features", "labels", "maximum")}


@torch.no_grad()
def length_metrics(head, cached, device, *, batch_size=1024):
    """CE selects the head; length errors use the same clipping as generation."""
    head.eval()
    x, y, maximum = (cached[key] for key in ("features", "labels", "maximum"))
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty split")
    loss, errors, clipped = 0., [], 0
    for start in range(0, len(y), batch_size):
        labels = y[start:start + batch_size].to(device)
        logits = head(x[start:start + batch_size].to(device))
        loss += F.cross_entropy(logits, labels, reduction="sum").item()
        raw = logits.argmax(-1).cpu()
        pred = raw.clamp(min=1).minimum(maximum[start:start + batch_size])
        clipped += int((pred != raw).sum())
        errors.append(pred - labels.cpu())
    error = torch.cat(errors)
    return dict(examples=len(y), length_ce=loss / len(y),
        mae=error.float().abs().mean().item(), mean_error=error.float().mean().item(),
        exact=(error == 0).float().mean().item(),
        within1=(error.abs() <= 1).float().mean().item(),
        under_fraction=(error < 0).float().mean().item(),
        over_fraction=(error > 0).float().mean().item(), clipped_fraction=clipped / len(y))


def epoch_batches(examples, epoch, *, batch_size=256, seed=2718):
    """Every training example appears exactly once, including the final batch."""
    if examples < 1 or epoch < 1 or batch_size < 1:
        raise ValueError("Invalid epoch settings")
    order = torch.randperm(examples, generator=torch.Generator().manual_seed(seed + epoch))
    return list(order.split(batch_size))


def _cpu_state(head):
    return {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}


def _best_checkpoint(state):
    return dict(format=FORMAT, identity=state["identity"], hidden_size=state["hidden_size"],
                length_head=state["best_head"], epoch=state["best_epoch"],
                dev_metrics=state["best_dev"])


def load_refit_head(path, model_sha, data_sha, device="cuda"):
    """Load only a fitted length head, checking the frozen translator and data."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    identity = saved.get("identity", {})
    if (saved.get("format") != FORMAT
            or identity.get("model_sha256") != model_sha
            or identity.get("data_sha256") != data_sha
            or identity.get("source_sha256") != source_hashes()):
        raise ValueError("Refit head model/data/source identity changed")
    head = LengthPredictor(saved["hidden_size"])
    head.load_state_dict(saved["length_head"], strict=True)
    if not all(bool(torch.isfinite(v).all()) for v in head.state_dict().values()):
        raise ValueError("Nonfinite refit head")
    return head.to(device).eval()


def fit_cached(train, dev, old_head, identity, output, *, hidden_size, device,
               max_epochs=30, patience=5, stopped=lambda: False):
    """Fit one fresh head; select by full-dev CE and resume completed epochs."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("Epoch limit and patience must be positive")
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(identity, training=dict(TRAINING, max_epochs=max_epochs, patience=patience),
                    train_examples=len(train["labels"]), dev_examples=len(dev["labels"]))
    latest, best_path = output / "latest.pt", output / "best.pt"
    torch.manual_seed(TRAINING["seed"])
    head = LengthPredictor(hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=TRAINING["learning_rate"],
                                  weight_decay=TRAINING["weight_decay"])
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if (state.get("identity") != identity or state.get("format") != FORMAT
                or state["hidden_size"] != hidden_size):
            raise ValueError("Training checkpoint identity changed")
        head.load_state_dict(state["length_head"])
        optimizer.load_state_dict(state["optimizer"])
        # An interruption between latest.pt and best.pt is repaired from latest.
        atomic_save(_best_checkpoint(state), best_path)
    else:
        state = dict(format=FORMAT, identity=identity, hidden_size=hidden_size,
            epoch=0, best_epoch=0, best_dev=None, best_head=None, bad_epochs=0, history=[],
            old_dev=length_metrics(old_head, dev, device), elapsed_seconds=0.)
    start_time, previous_time = time.monotonic(), state["elapsed_seconds"]
    x, y = train["features"].to(device), train["labels"].to(device)
    for epoch in range(state["epoch"] + 1, max_epochs + 1):
        if state["bad_epochs"] >= patience:
            break
        head.train()
        loss_sum, seen = 0., 0
        for indices in epoch_batches(len(y), epoch, batch_size=TRAINING["batch_size"],
                                     seed=TRAINING["seed"]):
            indices = indices.to(device)
            loss = F.cross_entropy(head(x[indices]), y[indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite length training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
            seen += len(indices)
        if seen != len(y):
            raise AssertionError("An epoch did not cover the full training split")
        metrics = length_metrics(head, dev, device)
        if not math.isfinite(metrics["length_ce"]):
            raise ValueError("Nonfinite development length CE")
        improved = state["best_dev"] is None or metrics["length_ce"] < state["best_dev"]["length_ce"]
        if improved:
            state.update(best_epoch=epoch, best_dev=metrics, best_head=_cpu_state(head), bad_epochs=0)
        else:
            state["bad_epochs"] += 1
        record = dict(epoch=epoch, train_examples=seen, train_ce=loss_sum / seen, dev=metrics)
        state["history"].append(record)
        state.update(epoch=epoch, length_head=_cpu_state(head), optimizer=optimizer.state_dict(),
                     elapsed_seconds=previous_time + time.monotonic() - start_time)
        atomic_save(state, latest)
        atomic_save(_best_checkpoint(state), best_path)
        log("length_refit_epoch", **record, best_epoch=state["best_epoch"])
        if stopped():
            raise Interrupted("Length training interrupted after a complete epoch")
    result = dict(identity=identity, completed=True, epochs=state["epoch"],
        selected_epoch=state["best_epoch"],
        stopping_reason="dev_patience" if state["bad_epochs"] >= patience else "epoch_limit",
        old_dev=state["old_dev"], refit_dev=state["best_dev"], history=state["history"],
        best_sha256=file_sha256(best_path), elapsed_seconds=state["elapsed_seconds"])
    atomic_json(output / "summary.json", result)
    return result


def run(args):
    with run_lock(args.output):
        stopped = []
        previous = {}
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            previous[sig] = signal.signal(sig, lambda signum, frame: stopped.append(signum))
        try:
            rows = {split: registered_rows(args.data, split) for split in ("train", "dev")}
            identity = dict(model_sha256=file_sha256(args.trained),
                data_sha256=file_sha256(args.data / "manifest.json"), source_sha256=source_hashes())
            model, _ = load_conditional(args.checkpoint, args.tokenizer, device=args.device)
            saved = torch.load(args.trained, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
            model.eval().requires_grad_(False)
            hidden_size = model.pretrained.config.hidden_dim
            old_head = LengthPredictor(hidden_size).to(args.device)
            old_head.load_state_dict(saved["length_head"])
            identity["old_length_head_sha256"] = _state_digest(saved["length_head"])
            del saved
            check = check_feature_batching(model, old_head, rows["train"], args.device)
            atomic_json(args.output / "feature-batch-check.json", dict(identity=identity, **check))
            log("length_feature_batch_check", **check)
            cached = {split: cache_features(model, rows[split], split, args.output / "cache",
                identity, hidden_size=hidden_size, device=args.device,
                batch_size=args.feature_batch_size, token_budget=args.feature_token_budget,
                stopped=lambda: bool(stopped))
                for split in ("train", "dev")}
            del model, rows
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
            result = fit_cached(cached["train"], cached["dev"], old_head, identity, args.output,
                hidden_size=hidden_size, device=args.device, max_epochs=args.max_epochs,
                patience=args.patience, stopped=lambda: bool(stopped))
            if (file_sha256(args.trained) != identity["model_sha256"]
                    or file_sha256(args.data / "manifest.json") != identity["data_sha256"]):
                raise ValueError("Training inputs changed during fitting")
            log("length_refit_complete", epochs=result["epochs"], selected_epoch=result["selected_epoch"],
                old_dev=result["old_dev"], refit_dev=result["refit_dev"])
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "trained", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--feature-token-budget", type=int, default=16384)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    try:
        run(args)
    except Interrupted as error:
        log("length_refit_interrupted", message=str(error))
        raise SystemExit(75) from error


if __name__ == "__main__":
    main()
