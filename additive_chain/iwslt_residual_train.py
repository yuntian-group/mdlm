"""Resumable target-only CRF adaptation with a frozen conditional translator."""
import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import signal
import time

import torch

from .backbone import file_sha256
from .counts import SparseCountPotential
from .iwslt_confirm import atomic_json, ordered_rows, read
from .iwslt_followup import stratified_rows
from .iwslt_length_refit import registered_rows
from .iwslt_residual_assets import (PROTOCOL, dense_count_scores, head_checkpoint,
    load_residual_head, make_residual_head, residual_source_hashes)
from .iwslt_residual_objective import residual_objective
from .iwslt_run import atomic_save, log, run_lock
from .translation import collate_pairs, load_conditional


@dataclass(frozen=True)
class TrainingConfig:
    kind: str = "global"
    seed: int = 2718
    rank: int = 64
    context_size: int = 128
    learning_rate: float = .001
    weight_decay: float = .01
    microbatch: int = 8
    accumulate: int = 4
    target_token_budget: int = 1024
    max_epochs: int = 5
    patience: int = 4
    dev_examples: int = 512
    checkpoint_every: int = 50
    dense_counts: bool = True

    def validate(self):
        if self.kind not in ("global", "contextual"):
            raise ValueError("Unknown stage")
        for name in ("rank", "context_size", "microbatch", "accumulate",
                     "target_token_budget", "max_epochs", "patience",
                     "dev_examples", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer required: " + name)
        if self.rank > 64 or self.dev_examples > 7283:
            raise ValueError("Stage rank must leave room for the inherited global head")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")


def epoch_groups(rows, epoch, config):
    """Every row once: shuffle, length-sort local pools, shuffle bounded batches."""
    rng = random.Random(config.seed + epoch)
    order = list(range(len(rows)))
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), 512):
        pool = sorted(order[start:start + 512],
                      key=lambda i: (len(rows[i]["target_ids"]), len(rows[i]["source_ids"])))
        pending, maximum = [], 0
        for index in pool:
            size = len(rows[index]["target_ids"]) + 1
            width = max(maximum, size)
            if pending and (len(pending) >= config.microbatch
                            or (len(pending) + 1) * width > config.target_token_budget):
                batches.append(pending)
                pending, maximum = [], 0
            pending.append(index)
            maximum = max(maximum, size)
        if pending:
            batches.append(pending)
    rng.shuffle(batches)
    return [batches[i:i + config.accumulate] for i in range(0, len(batches), config.accumulate)]


@torch.no_grad()
def evaluate_block_loss(rows, model, potential, head, config, device,
                        global_head=None, count_table=None):
    """The same corruption bank for every checkpoint; no decoding or test access."""
    generator = torch.Generator(device=device).manual_seed(314159)
    head.eval()
    total, selected, missing = 0., 0, 0
    for group in epoch_groups(rows, 0, config):
        for indices in group:
            batch = collate_pairs([rows[i] for i in indices], device)
            value = residual_objective(batch, generator=generator, frozen=model,
                count_potential=potential, head=head, global_head=global_head,
                global_strength=1. if global_head is not None else 0.,
                count_score_table=count_table, temperature=PROTOCOL["temperature"],
                nucleus_p=PROTOCOL["nucleus_p"], top_k=PROTOCOL["top_k"])
            if value.loss is not None:
                total += float(value.total_nll)
                selected += value.selected_tokens
                missing += value.metrics["original_out_of_support_tokens"]
    if not selected or not math.isfinite(total):
        raise ValueError("Invalid development block loss")
    return dict(examples=len(rows), selected_tokens=selected, total_nll=total,
                nll_per_selected_token=total / selected,
                original_out_of_support_fraction=missing / selected)


def update_plateau(state, metrics, epoch):
    value = metrics["nll_per_selected_token"]
    improved = state["best_dev"] is None or value < state["best_dev"]["nll_per_selected_token"]
    if improved:
        state["best_dev"], state["bad_checks"] = metrics, 0
    else:
        state["bad_checks"] += 1
    state["history"].append(dict(epoch=epoch, update=state["update"], dev=metrics,
                                improved=improved))
    return improved


def same_payload(first, second):
    """Compare a replayed immutable snapshot without rewriting its archive."""
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(
            same_payload(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(
            same_payload(a, b) for a, b in zip(first, second))
    return first == second


def publish_completion(output, kind, state):
    """Reconstruct downstream inputs even after a crash immediately at completion."""
    for candidate in state["candidates"]:
        if file_sha256(Path(candidate["head"])) != candidate["sha256"]:
            raise ValueError("Completed training candidate changed")
    atomic_json(output / "candidates.json",
                dict(kind=kind, candidates=state["candidates"], complete=True))
    atomic_json(output / "summary.json", dict(kind=kind, **state))


def run(args):
    config = TrainingConfig(**{name: getattr(args, name) for name in TrainingConfig.__dataclass_fields__})
    config.validate()
    if config.kind == "global" and args.global_head is not None:
        raise ValueError("Global stage has no inherited global head")
    with run_lock(args.output):
        train = registered_rows(args.data, "train")
        dev = stratified_rows(ordered_rows(args.data, "dev"), config.dev_examples)
        model_sha = file_sha256(args.trained)
        data_sha = file_sha256(args.data / "manifest.json")
        counts_sha = file_sha256(args.data / "counts.pt")
        if counts_sha != read(args.data / "manifest.json")["counts_sha256"]:
            raise ValueError("Counts changed")
        global_sha = None if args.global_head is None else file_sha256(args.global_head)
        identity = dict(PROTOCOL, model_sha256=model_sha, data_sha256=data_sha,
            counts_sha256=counts_sha, inherited_global_sha256=global_sha,
            training_config=asdict(config), source_sha256=residual_source_hashes(),
            train_examples=len(train), dev_examples=7283,
            objective="selected gold block logZ-logZ_clamped on gold-augmented candidates",
            schedules=[2, 4, 8], test_access=False)
        saved = None
        if (args.output / "latest.pt").exists():
            saved = torch.load(args.output / "latest.pt", map_location="cpu", weights_only=True)
            if saved["identity"] != identity:
                raise ValueError("Training continuation identity changed")
            if saved["state"]["complete"]:
                publish_completion(args.output, config.kind, saved["state"])
                log("residual_training_already_complete", kind=config.kind)
                return
        elif any(p.name not in ("run.lock", "latest.pt.tmp") for p in args.output.iterdir()):
            raise ValueError("Nonempty new training directory")
        model, _ = load_conditional(args.checkpoint, args.tokenizer, device=args.device)
        trained = torch.load(args.trained, map_location=args.device, weights_only=False)
        model.load_state_dict(trained["model"])
        del trained
        model.eval().requires_grad_(False)
        global_head = None
        if args.global_head is not None:
            global_head, metadata = load_residual_head(
                args.global_head, model_sha, data_sha, args.device)
            if metadata["kind"] != "global" or metadata["identity"]["counts_sha256"] != counts_sha:
                raise ValueError("Wrong inherited head")
        torch.manual_seed(config.seed)
        head = make_residual_head(config.kind, model, rank=config.rank,
                                  context_size=config.context_size)
        optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        generator = torch.Generator(device=args.device).manual_seed(config.seed)
        counts = torch.load(args.data / "counts.pt", weights_only=True)
        potential = SparseCountPotential.from_counts(
            *(counts[k].to(args.device) for k in
              ("left_counts", "right_counts", "pair_keys", "pair_counts")),
            strength=PROTOCOL["count_strength"], smoothing=.1)
        count_table = None
        if config.dense_counts:
            if args.device.startswith("cuda"):
                free, _ = torch.cuda.mem_get_info()
                required = potential.vocab_size ** 2 * 4 + 12 * 1024 ** 3
                if free < required:
                    raise ValueError("Insufficient GPU memory for requested dense count lookup")
            count_table = dense_count_scores(potential)
        stop = []
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, lambda *_: stop.append(True))
        state = dict(epoch=1, next_group=0, update=0, seen_epoch=0,
            best_dev=None, bad_checks=0, history=[], candidates=[], complete=False,
            elapsed_seconds=0., best_snapshot=None, last_dev_group=0, initialized=False,
            stop_training=False)
        if saved is not None:
            head.load_state_dict(saved["head_state"])
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator"])
            state = saved["state"]
        start, previous_time, initial_updates = time.monotonic(), state["elapsed_seconds"], state["update"]

        def persist():
            state["elapsed_seconds"] = previous_time + time.monotonic() - start
            atomic_save(dict(identity=identity, state=state,
                head_state={k: v.detach().cpu() for k, v in head.state_dict().items()},
                optimizer=optimizer.state_dict(), generator=generator.get_state().cpu()),
                args.output / "latest.pt")

        def snapshot(name, epoch, metrics):
            path = args.output / (name + ".pt")
            payload = head_checkpoint(head, config.kind, identity, checkpoint_id=name,
                epoch=epoch, update=state["update"], metrics=metrics)
            if path.exists():
                existing = torch.load(path, map_location="cpu", weights_only=True)
                if not same_payload(payload, existing):
                    raise ValueError("Immutable candidate differs from replayed snapshot")
            else:
                atomic_save(payload, path)
            return dict(name=name, head=str(path.resolve()), sha256=file_sha256(path),
                        epoch=epoch, update=state["update"])

        def validate(epoch):
            # Save the exact weights before publishing immutable candidates. A
            # GPU backward pass need not reproduce atomic sums bit-for-bit.
            persist()
            metrics = evaluate_block_loss(dev, model, potential, head, config, args.device,
                                          global_head, count_table)
            improved = update_plateau(state, metrics, epoch)
            log("residual_dev", kind=config.kind, epoch=epoch, update=state["update"],
                bad_checks=state["bad_checks"], **metrics)
            if improved:
                state["best_snapshot"] = snapshot(
                    f"best-loss-{state['update']:07d}", epoch, metrics)
            return metrics

        if saved is None:
            persist()
        if not state["initialized"]:
            metrics = validate(0)
            state["candidates"].append(snapshot("epoch-0000", 0, metrics))
            state["initialized"] = True
            persist()
        log("residual_training_started", kind=config.kind, resumed=saved is not None,
            gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
            train_examples=len(train), config=asdict(config), count_cache_bytes=(
                count_table.numel() * count_table.element_size() if count_table is not None else 0))
        while state["epoch"] <= config.max_epochs and not state["stop_training"]:
            epoch = state["epoch"]
            groups = epoch_groups(train, epoch, config)
            validate_every = max(1, math.ceil(len(groups) / 4))
            for index in range(state["next_group"], len(groups)):
                head.train()
                optimizer.zero_grad(set_to_none=True)
                total, selected, examples, absent = 0., 0, 0, 0
                for indices in groups[index]:
                    batch = collate_pairs([train[i] for i in indices], args.device)
                    value = residual_objective(batch, generator=generator, frozen=model,
                        count_potential=potential, head=head, global_head=global_head,
                        global_strength=1. if global_head is not None else 0.,
                        count_score_table=count_table, temperature=PROTOCOL["temperature"],
                        nucleus_p=PROTOCOL["nucleus_p"], top_k=PROTOCOL["top_k"])
                    examples += len(indices)
                    if value.loss is not None:
                        value.total_nll.backward()
                        total += float(value.total_nll.detach())
                        selected += value.selected_tokens
                        absent += value.metrics["original_out_of_support_tokens"]
                if selected:
                    for parameter in head.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(selected)
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    state["update"] += 1
                state["seen_epoch"] += examples
                state["next_group"] = index + 1
                if state["update"] % 25 == 0:
                    log("residual_train", kind=config.kind, epoch=epoch, update=state["update"],
                        examples_epoch=state["seen_epoch"], groups=len(groups),
                        nll_per_selected_token=total / max(1, selected),
                        selected_tokens=selected, missing_targets=absent,
                        elapsed_seconds=previous_time + time.monotonic() - start,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated()
                            if args.device.startswith("cuda") else 0)
                if ((index + 1) % validate_every == 0 and index + 1 < len(groups)
                        and state["last_dev_group"] != index + 1):
                    validate(epoch)
                    state["last_dev_group"] = index + 1
                    persist()
                if state["update"] % config.checkpoint_every == 0:
                    persist()
                if stop or (args.max_new_updates and state["update"] - initial_updates >= args.max_new_updates):
                    persist()
                    atomic_json(args.output / "progress.json", dict(kind=config.kind, **state))
                    log("residual_training_checkpointed", kind=config.kind, update=state["update"],
                        signal_requested=bool(stop), complete=False)
                    return
            if state["seen_epoch"] != len(train):
                raise AssertionError("Training epoch did not cover every example once")
            metrics = validate(epoch)
            name = f"epoch-{epoch:04d}"
            state["candidates"].append(snapshot(name, epoch, metrics))
            state.update(epoch=epoch + 1, next_group=0, seen_epoch=0, last_dev_group=0,
                         stop_training=state["bad_checks"] >= config.patience)
            persist()
            atomic_json(args.output / "candidates-progress.json",
                        dict(kind=config.kind, candidates=state["candidates"], complete=False))
            if state["bad_checks"] >= config.patience:
                break
        # Include the best block-loss checkpoint even when it occurred between epoch ends.
        best = state["best_snapshot"]
        if best["update"] > 0 and not any(
                candidate["name"] == best["name"] for candidate in state["candidates"]):
            state["candidates"].append(best)
        state["complete"] = True
        state["termination"] = ("development_block_loss_plateau" if state["bad_checks"] >= config.patience
                                else "epoch_limit")
        persist()
        publish_completion(args.output, config.kind, state)
        log("residual_training_complete", kind=config.kind, updates=state["update"],
            epochs=state["epoch"] - 1, elapsed_seconds=state["elapsed_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "trained", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--global-head", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-updates", type=int, default=0)
    for name, field in TrainingConfig.__dataclass_fields__.items():
        if name == "dense_counts":
            parser.add_argument("--no-dense-counts", dest=name, action="store_false")
        else:
            parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
