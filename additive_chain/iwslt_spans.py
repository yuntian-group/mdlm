"""Compare joint and independent CRF probabilities at identical masked inputs.

This is a held-out reconstruction diagnostic, not free translation: the
unmasked target words are visible. Gold words never alter candidate support.
"""
import argparse
import json
import random
from pathlib import Path

import torch

from .backbone import file_sha256
from .chain import marginals, path_score, sample_ffbs
from .generation import ChainDecoder
from .iwslt_run import dataset, log, run_lock
from .translation import MASK_ID, collate_pairs, load_conditional


@torch.no_grad()
def compare_span(candidates, potential, gold, *, draws=4, seed=101):
    """One span plus any singleton boundary neighbors, using exact FP64 DP."""
    ids, unary = candidates.candidate_ids, candidates.log_unary
    if ids.shape[0] != 1 or gold.shape != ids.shape[:2] or draws < 1:
        raise ValueError("Expected one chain and matching gold IDs")
    # Padded IDs have -inf unaries. Their pair entries cannot contribute.
    safe = ids.clamp_max(potential.vocab_size - 1)
    pair = potential.log_scores(safe[:, :-1, :, None], safe[:, 1:, None, :])
    posterior = marginals(unary, pair, return_edges=False)
    match = ids.eq(gold[..., None]) & torch.isfinite(unary)
    supported = bool(match.any(-1).all())
    result = {"supported": supported, "gold_tokens": gold[0].tolist(),
              "support_sizes": torch.isfinite(unary[0]).sum(-1).tolist(),
              "joint_gold_log_probability": None, "independent_gold_log_probability": None,
              "joint_gold_probability": 0., "independent_gold_probability": 0.}
    if supported:
        slots = match.long().argmax(-1)
        joint = (path_score(unary, pair, slots) - posterior.log_partition).item()
        independent = posterior.node.gather(-1, slots[..., None]).log().sum().item()
        result.update(joint_gold_log_probability=joint,
                      independent_gold_log_probability=independent,
                      joint_gold_probability=float(torch.exp(torch.tensor(joint, dtype=torch.float64))),
                      independent_gold_probability=float(torch.exp(torch.tensor(independent, dtype=torch.float64))))
    generators = [torch.Generator(device=ids.device).manual_seed(seed) for _ in range(2)]
    joint_slots = sample_ffbs(unary, pair, num_samples=draws, generator=generators[0])[:, 0]
    independent_slots = torch.multinomial(posterior.node[0], draws, replacement=True,
                                        generator=generators[1]).T
    result["joint_draws"] = ids[0].expand(draws, -1, -1).gather(-1, joint_slots[..., None]).squeeze(-1).tolist()
    result["independent_draws"] = ids[0].expand(draws, -1, -1).gather(-1, independent_slots[..., None]).squeeze(-1).tolist()
    return result


def span_population(rows, length, limit, seed):
    """Choose by length eligibility only, before any model scoring."""
    eligible = [r for r in rows if len(r["target_ids"]) >= length]
    random.Random(seed).shuffle(eligible)
    chosen = []
    for row in eligible[:limit]:
        rng = random.Random(f"{seed}:{length}:{row['id']}")
        start = rng.randrange(len(row["target_ids"]) - length + 1)
        chosen.append((row, start))
    return chosen, len(eligible)


@torch.no_grad()
def run(args):
    model_sha = file_sha256(args.trained)
    model, _ = load_conditional(args.checkpoint, args.tokenizer)
    state = torch.load(args.trained, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"])
    del state
    model.eval()
    rows = dataset(args.data, "dev")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if file_sha256(args.data / "counts.pt") != manifest["counts_sha256"]:
        raise ValueError("Training counts changed")
    counts = torch.load(args.data / "counts.pt", weights_only=True)
    decoder = ChainDecoder(mask_id=MASK_ID, device="cuda", count_statistics=counts,
                           count_strength=args.strength, temperature=args.temperature,
                           top_k=args.top_k, nucleus_p=args.top_p)
    for length in args.span_lengths:
        selected, eligible = span_population(rows, length, args.limit, args.seed)
        identity = dict(model_sha256=model_sha, data_sha256=file_sha256(args.data / "manifest.json"),
            length=length, eligible=eligible, selected=[dict(id=r["id"], start=s) for r, s in selected],
            strength=args.strength, temperature=args.temperature, seed=args.seed,
            top_k=args.top_k, top_p=args.top_p,
            source_sha256={p: file_sha256(Path(__file__).parent / p) for p in
                ("iwslt_spans.py", "translation.py", "generation.py", "chain.py", "counts.py", "candidates.py")},
            protocol="Same masked target span, visible gold context; unchanged model-derived candidates; unsupported gold has zero probability")
        path = args.output / f"span-{length}.json"
        if path.exists():
            if json.loads(path.read_text())["identity"] != identity:
                raise ValueError("Existing span comparison differs")
            continue
        records = []
        for i, (row, start) in enumerate(selected):
            batch = collate_pairs([row], "cuda")
            tokens = batch.tokens.clone()
            target_start = batch.starts[0]
            tokens[:, target_start + start:target_start + start + length] = MASK_ID
            output = model(tokens, batch.valid)
            # Visible target tokens outside the immediate neighbors contribute
            # only constants. Never add a source-to-target CRF boundary edge.
            lo = max(0, start - 1)
            hi = min(len(row["target_ids"]) + 1, start + length + 1)
            indices = slice(target_start + lo, target_start + hi)
            candidates = decoder._candidates(output.log_probs[:, indices], tokens[:, indices])
            result = compare_span(candidates, decoder.potential, batch.tokens[:, indices], seed=args.seed + i)
            gold = batch.tokens[:, target_start + start:target_start + start + length]
            full = output.log_probs[:, target_start + start:target_start + start + length, :MASK_ID]
            gold_logp = full.gather(-1, gold[..., None])
            ranks = (full > gold_logp).sum(-1) + 1
            records.append(dict(id=row["id"], start=start, span_length=length,
                                left_boundary=start - lo, gold_full_vocab_ranks=ranks[0].tolist(),
                                gold_full_vocab_log_probabilities=gold_logp[0, :, 0].tolist(), **result))
            if (i + 1) % 50 == 0:
                log("span_progress", length=length, completed=i + 1, total=len(selected))
        if file_sha256(args.trained) != model_sha:
            raise ValueError("Checkpoint changed during span analysis")
        summary = dict(examples=len(records), supported=sum(r["supported"] for r in records),
            mean_joint_gold_probability=sum(r["joint_gold_probability"] for r in records) / len(records),
            mean_independent_gold_probability=sum(r["independent_gold_probability"] for r in records) / len(records))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(identity=identity, summary=summary, records=records),
                                        indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        log("span_complete", length=length, **summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "data", "trained", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--span-lengths", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--strength", type=float, required=True)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--top-p", type=float, default=.9)
    args = p.parse_args()
    if args.limit < 1 or any(x < 1 for x in args.span_lengths):
        p.error("Positive population size and span lengths required")
    with run_lock(args.output):
        run(args)


if __name__ == "__main__":
    main()
