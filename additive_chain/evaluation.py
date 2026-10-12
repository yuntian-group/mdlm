"""Saved-generation metrics; no generation, filtering, or model downloads.

Scoring is adapted from BD3-LM's Apache-2.0 ``metrics.py`` at commit
1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7 (Cornell University, 2024).
Changes: functions instead of TorchMetrics state, explicit batch-one scoring,
FP64 pooled accounting, input validation, and separate timing/diagnostics.
See EVALUATION.md for the exact protocol and the limits of its CPU tests.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F


SCORER_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonnegative(value, name):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             name + " must be a finite nonnegative number")
    return float(value)


def _tokens(values):
    _require(isinstance(values, list) and bool(values)
             and all(type(v) is int and v >= 0 for v in values),
             "token_ids must be a nonempty flat list of nonnegative integers")
    return values


def token_diagnostics(token_ids, eos_token_id):
    """Empirical token entropy and repeated-4gram fraction on RAW token IDs.

    Includes initial context/BOS and every EOS, without decode/retokenization.
    Entropy is not predictive entropy of the language model.
    """
    values = _tokens(token_ids)
    counts = Counter(values)
    n = len(values)
    grams = [tuple(values[i:i + 4]) for i in range(max(0, n - 3))]
    return {
        "token_entropy_nats": -math.fsum((c / n) * math.log(c / n)
                                        for c in counts.values()),
        "unique_token_fraction": len(counts) / n,
        "repeated_4gram_fraction": 1 - len(set(grams)) / len(grams) if grams else 0.,
        "eos_tokens_including_initial_context": counts.get(eos_token_id, 0),
    }


def last256_entropy(token_ids):
    """The upstream FP32 empirical entropy statistic (not a new filter).

    Its fixed-length generation policy rejected entropy <4 nats. This function
    reports the statistic only; saved acceptance decisions remain authoritative.
    """
    values = torch.tensor(_tokens(token_ids)[-256:], dtype=torch.long)
    counts = torch.unique(values, return_counts=True, sorted=False)[1]
    return torch.special.entr(counts.float() / counts.sum()).sum().item()


def decode_retaining_eos(tokenizer, token_ids):
    """Decode a single saved RAW sequence, keeping all special tokens/context."""
    return tokenizer.batch_decode([_tokens(token_ids)], skip_special_tokens=False)[0]


def scoring_windows(length, context_size=1024, stride=512):
    """Yield (start, end, first_target) with every target position1..L-1 once.

    The first window scores all its next-token targets. Later windows score
    only newly reached targets, using the overlapping prefix as context.
    """
    _require(type(length) is int and length >= 2, "Need at least two token positions")
    _require(type(context_size) is int and type(stride) is int
             and 1 <= stride < context_size, "Require 1 <= stride < context_size")
    windows = max(1, math.ceil((length - context_size + stride) / stride))
    for index in range(windows):
        start = index * stride
        end = min(start + context_size, length)
        first = 1 if index == 0 else start + context_size - stride
        yield start, end, first


@torch.no_grad()
def score_token_batch(model, input_ids, attention_mask, *, eos_token_id,
                      context_size=1024, stride=512, undefined="error"):
    """Score right-padded retokenized samples, one sequence per model call.

    Matches the GPT-2 branch of the pinned upstream scorer: FP32 logits/CE,
    EOS IDs excluded as TARGETS but retained as CONTEXT. Position0 is unscored.
    Padding must use EOS IDs, as in the original scorer. All samples share the
    tokenizer's padded length. ``model`` must already be in eval mode/device.
    """
    _require(undefined in ("error", "record"), "Unknown undefined-score policy")
    _require(type(context_size) is int and type(stride) is int
             and 1 <= stride < context_size, "Require 1 <= stride < context_size")
    _require(input_ids.ndim == 2 and input_ids.shape == attention_mask.shape
             and input_ids.shape[0] > 0 and input_ids.shape[1] >= (1 if undefined == "record" else 2),
             "Expected nonempty [samples, length] IDs and matching attention mask")
    _require(input_ids.dtype == torch.long and not model.training,
             "Integer IDs and an eval-mode model are required")
    _require(bool(((attention_mask == 0) | (attention_mask == 1)).all()),
             "Attention mask must be binary")
    _require(bool((attention_mask[:, 1:] <= attention_mask[:, :-1]).all())
             and bool(attention_mask[:, 0].all()), "Only right padding is supported")
    _require(bool(input_ids[attention_mask == 0].eq(eos_token_id).all()),
             "Padding IDs must be EOS, matching the upstream scoring policy")
    rows = []
    for index in range(input_ids.shape[0]):
        ids = input_ids[index:index + 1]
        mask = attention_mask[index:index + 1]
        if not bool(ids[:, 1:].ne(eos_token_id).any()):
            _require(undefined == "record", "Sample has no non-EOS next-token targets")
            rows.append({"sample_id": index, "status": "undefined",
                "reason": "no_non_EOS_targets", "scored_tokens": 0,
                "total_nll_float64_sum": 0., "legacy_sequence_ppl": None,
                "scored_target_mask": [False] * ids.shape[1]})
            continue
        nll = torch.zeros_like(ids, dtype=torch.float32)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for start, end, first in scoring_windows(ids.shape[1], context_size, stride):
            chunk = ids[:, start:end]
            output = model(chunk, attention_mask=mask[:, start:end])
            logits = output.logits if hasattr(output, "logits") else output[0]
            _require(logits.dtype == torch.float32 and logits.shape[:2] == chunk.shape,
                     "Pinned scoring requires FP32 logits with [1, length, vocab] shape")
            losses = F.cross_entropy(logits.transpose(-1, -2)[..., :-1],
                                     chunk[..., 1:], reduction="none")
            offset = first - start - 1
            nll[:, first:end] = losses[:, offset:]
            valid[:, first:end] = chunk[:, offset + 1:].ne(eos_token_id)
        scored_tokens = int(valid.sum().item())
        _require(scored_tokens > 0, "Sample has no non-EOS next-token targets")
        weighted = nll * valid.float()
        # Preserve the legacy FP32 sequence PPL before its arithmetic mean.
        legacy_ppl = (weighted.sum() / valid.float().sum()).exp().item()
        total_nll = (nll.cpu().double() * valid.cpu().double()).sum().item()
        _require(math.isfinite(total_nll) and math.isfinite(legacy_ppl),
                 "Nonfinite model score")
        rows.append({
            "sample_id": index,
            "total_nll_float64_sum": total_nll,
            "scored_tokens": scored_tokens,
            "legacy_sequence_ppl": legacy_ppl,
            "scored_target_mask": valid[0].cpu().tolist(),
        })
        if undefined == "record":
            rows[-1]["status"] = "defined"
    return rows


def aggregate_scores(rows):
    """Pool token NLLs/counts, separately retaining mean(sequence PPL)."""
    _require(bool(rows), "No score rows")
    for row in rows:
        _nonnegative(row["total_nll_float64_sum"], "token NLL")
        _require(type(row["scored_tokens"]) is int and row["scored_tokens"] > 0,
                 "Each sample must contain scored targets")
        _require(math.isfinite(row["legacy_sequence_ppl"])
                 and row["legacy_sequence_ppl"] >= 1, "Invalid sequence PPL")
    total = math.fsum(row["total_nll_float64_sum"] for row in rows)
    count = sum(row["scored_tokens"] for row in rows)
    return {"samples": len(rows), "total_nll_float64_sum": total,
            "scored_tokens": count, "pooled_token_ppl": math.exp(total / count),
            "legacy_mean_sequence_ppl": statistics.fmean(
                row["legacy_sequence_ppl"] for row in rows)}


def aggregate_population_scores(rows):
    """Keep undefined members visible; never relabel their omission as a full score."""
    defined, undefined = [], []
    for row in rows:
        if row.get("status") == "undefined":
            _require(row.get("reason") == "no_non_EOS_targets"
                and type(row["scored_tokens"]) is int and row["scored_tokens"] == 0
                and _nonnegative(row["total_nll_float64_sum"], "undefined NLL") == 0
                and row["legacy_sequence_ppl"] is None, "Invalid undefined score record")
            undefined.append(row)
        else:
            _require(row.get("status") == "defined", "Explicit score status required")
            defined.append(row)
    subset = aggregate_scores(defined) if defined else None
    return {"samples": len(rows), "defined": len(defined), "undefined": len(undefined),
        "total_nll_float64_sum": 0. if subset is None else subset["total_nll_float64_sum"],
        "scored_tokens": 0 if subset is None else subset["scored_tokens"],
        "pooled_token_ppl": subset["pooled_token_ppl"] if subset is not None and not undefined else None,
        "legacy_mean_sequence_ppl": subset["legacy_mean_sequence_ppl"]
            if subset is not None and not undefined else None,
        "defined_subset": subset}


def summarize_attempts(attempts, *, eos_token_id, samples_requested,
                       timing_complete, setup_seconds=None, warmup_seconds=None,
                       population="accepted"):
    """Summarize saved single-sequence attempts; never drop rejected work.

    ``generation_seconds`` includes complete generation, preparation, inference,
    decoding and checks, but excludes durable-file I/O/loading/scoring/setup.
    Missing timing is unknown, never zero; incomplete timing disables rates.
    ``new_tokens`` is explicit, excluding supplied prompt/BOS, not inferred.
    """
    _require(type(samples_requested) is int and samples_requested > 0,
             "samples_requested must be positive")
    _require(type(timing_complete) is bool, "timing_complete must be explicit")
    _require(population in ("accepted", "raw"), "Unknown generation population")
    _require([r.get("attempt_id") for r in attempts] == list(range(len(attempts))),
             "Preserve contiguous raw attempt IDs, including rejections")
    accepted, durations, diagnostic_rows = [], [], []
    tokens_known = True
    for row in attempts:
        _require(type(row.get("accepted")) is bool, "Acceptance must be a saved boolean")
        values = _tokens(row["token_ids"])
        diagnostic_rows.append(token_diagnostics(values, eos_token_id))
        if row["accepted"]:
            accepted.append(len(diagnostic_rows) - 1)
        elapsed = row.get("generation_seconds")
        durations.append(None if elapsed is None else _nonnegative(elapsed, "generation_seconds"))
        new = row.get("new_tokens")
        if new is None:
            tokens_known = False
        else:
            _require(type(new) is int and 0 <= new <= len(values), "Invalid new_tokens")
    _require(len(accepted) <= samples_requested, "More accepted samples than declared")
    _require(population != "raw" or len(attempts) <= samples_requested,
             "Raw population exceeds its declared fixed size")
    measured = timing_complete and bool(attempts) and all(d is not None for d in durations)
    seconds = math.fsum(d for d in durations if d is not None)
    if measured:
        _require(seconds > 0, "Completed timing must be positive")
    setup = None if setup_seconds is None else _nonnegative(setup_seconds, "setup_seconds")
    warmup = None if warmup_seconds is None else _nonnegative(warmup_seconds, "warmup_seconds")
    count = len(accepted)
    def means(indices):
        return {key: statistics.fmean(diagnostic_rows[i][key] for i in indices)
             if indices else None for key in (
                 "token_entropy_nats", "unique_token_fraction", "repeated_4gram_fraction",
                 "eos_tokens_including_initial_context")}
    return {"attempts": len(attempts), "accepted": count,
            "rejected": len(attempts) - count, "samples_requested": samples_requested,
            "population": population,
            "population_complete": (len(attempts) if population == "raw" else count) == samples_requested,
            "rejection_rate": (len(attempts) - count) / len(attempts) if attempts else None,
            "timing_complete": measured, "generation_seconds_recorded": seconds,
            "seconds_per_accepted_sequence": seconds / count if measured and count else None,
            "seconds_per_raw_sequence": seconds / len(attempts) if measured else None,
            "accepted_new_tokens_per_second": sum(attempts[i]["new_tokens"] for i in accepted) / seconds
                if measured and tokens_known else None,
            "raw_new_tokens_per_second": sum(r["new_tokens"] for r in attempts) / seconds
                if measured and tokens_known else None,
            "setup_seconds": setup, "warmup_seconds": warmup,
            "setup_warmup_amortized_seconds_per_accepted_sequence": (seconds + setup + warmup) / count
                if measured and count and setup is not None and warmup is not None else None,
            "accepted_diagnostics_mean": means(accepted),
            "raw_diagnostics_mean": means(list(range(len(attempts)))),
            "rejected_diagnostics_mean": means([i for i, r in enumerate(attempts) if not r["accepted"]]),
            "last256_entropy_nats_all_attempts": [last256_entropy(r["token_ids"]) for r in attempts]}


def evaluate_saved(data, generator_tokenizer, scorer_tokenizer, scorer, *, device="cpu",
                   max_length=1024):
    """Evaluate a portable saved-attempt dictionary with caller-loaded models.

    Decode generator IDs with special tokens retained, then retokenize the
    declared population. Raw-schema inputs score all saved attempts, including
    rejections, and retain undefined rows. No new samples or rejection decisions.
    """
    schemas = {"additive_chain_saved_generations_v1": "accepted",
               "additive_chain_saved_raw_generations_v1": "raw"}
    _require(data.get("schema") in schemas, "Unknown saved-sample schema")
    population = schemas[data["schema"]]
    _require(data.get("population", population) == population,
             "Saved schema and population disagree")
    _require(data["eos_token_id"] == generator_tokenizer.eos_token_id,
             "Saved token IDs and generator tokenizer EOS differ")
    _require(type(max_length) is int and max_length >= 2, "Invalid scoring maximum length")
    summary = summarize_attempts(data["attempts"], eos_token_id=data["eos_token_id"],
        samples_requested=data["samples_requested"], timing_complete=data["generation_timing_complete"],
        setup_seconds=data.get("setup_seconds"), warmup_seconds=data.get("warmup_seconds"),
        population=population)
    chosen = [r for r in data["attempts"] if population == "raw" or r["accepted"]]
    texts = [decode_retaining_eos(generator_tokenizer, r["token_ids"]) for r in chosen]
    for row, text in zip(chosen, texts):
        _require("text" not in row or row["text"] == text or (not row["accepted"] and row["text"] is None),
                 "Saved text differs from retained-special-token decoding")
    rows = []
    if texts:
        _require(scorer_tokenizer.eos_token_id is not None, "Scorer tokenizer needs EOS")
        if scorer_tokenizer.pad_token_id is None:
            scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
        _require(scorer_tokenizer.pad_token_id == scorer_tokenizer.eos_token_id
                 and scorer_tokenizer.padding_side == "right", "Scorer requires right EOS padding")
        batch = scorer_tokenizer(texts, return_tensors="pt", return_token_type_ids=False,
            return_attention_mask=True, truncation=True, padding=True, max_length=max_length)
        rows = score_token_batch(scorer, batch["input_ids"].to(device),
            batch["attention_mask"].to(device), eos_token_id=scorer_tokenizer.eos_token_id,
            undefined="record" if population == "raw" else "error")
        for row, attempt, text in zip(rows, chosen, texts):
            row.update(attempt_id=attempt["attempt_id"], accepted=attempt["accepted"],
                       text_sha256=hashlib.sha256(text.encode()).hexdigest())
    quality = aggregate_population_scores(rows) if population == "raw" else aggregate_scores(rows) if rows else None
    return {"schema": "additive_chain_raw_generation_metrics_v1" if population == "raw"
                else "additive_chain_generation_metrics_v1", "generation": summary,
            "quality": quality, "population": population,
            "quality_subsets": {key: aggregate_population_scores([r for r in rows if r["accepted"] == flag])
                for key, flag in (("accepted", True), ("rejected", False))} if population == "raw" else None,
            "quality_population": "complete" if summary["population_complete"] else "partial",
            "score_rows": rows,
            "scoring_protocol": {"retokenize": True, "decode_skip_special_tokens": False,
                "context_size": 1024, "stride": 512, "max_length": max_length,
                "batch_size": 1, "logits_dtype": "float32",
                "eos_context": "retained", "eos_targets": "excluded",
                "position_zero_target": "excluded"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scorer", default="gpt2-large")
    parser.add_argument("--scorer-revision", default=SCORER_REVISION)
    parser.add_argument("--generator-tokenizer", default="gpt2")
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args(argv)
    _require(not args.output.exists(), "Output exists; refusing to replace an evaluation")
    # Optional dependency: only the explicit CLI loads cached Hugging Face assets.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_tokenizer,
        revision=args.tokenizer_revision, local_files_only=True, trust_remote_code=False)
    scorer_tokenizer = AutoTokenizer.from_pretrained(args.scorer,
        revision=args.scorer_revision, local_files_only=True, trust_remote_code=False)
    scorer = AutoModelForCausalLM.from_pretrained(args.scorer, revision=args.scorer_revision,
        local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32).eval().to(args.device)
    source = args.samples.read_bytes()
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    result = evaluate_saved(json.loads(source), generator_tokenizer, scorer_tokenizer, scorer,
                            device=args.device, max_length=args.max_length)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(args.device)
    result.update(scoring_seconds=time.perf_counter() - started,
        input_sha256=hashlib.sha256(source).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scorer={"name_or_path": args.scorer, "revision": args.scorer_revision},
        generator_tokenizer={"name_or_path": args.generator_tokenizer, "revision": args.tokenizer_revision},
        environment={"torch": str(torch.__version__), "device": args.device})
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
