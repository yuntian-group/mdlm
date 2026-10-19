"""Matched IWSLT development decoding from one immutable trained checkpoint.

Resume skips only finished configurations with matching model/data hashes.
The screen does not choose hyperparameters on test examples or alter outputs.
"""
import argparse
import json
from pathlib import Path
import random

import torch

from .backbone import file_sha256
from .generation import ChainDecoder
from .iwslt_run import log, run_lock, score_translations, dataset
from .translation import MASK_ID, LengthPredictor, load_conditional


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "data", "trained", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--strengths", type=float, nargs="+", default=[.05, .1, .2])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.])
    parser.add_argument("--baseline-temperatures", type=float, nargs="+", default=[.85, 1.])
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if args.limit <= 0 or any(s <= 0 for s in args.steps) or any(s <= 0 for s in args.strengths):
        parser.error("Require a positive development size, steps, and pair strengths")
    with run_lock(args.output):
        run(args)


def run(args):
    model_sha = file_sha256(args.trained)
    data_sha = file_sha256(args.data / "manifest.json")
    model, tokenizer = load_conditional(args.checkpoint, args.tokenizer)
    saved = torch.load(args.trained, map_location="cuda", weights_only=False)
    model.load_state_dict(saved["model"])
    head = LengthPredictor(model.pretrained.config.hidden_dim).cuda()
    head.load_state_dict(saved["length_head"])
    del saved
    model.eval()
    head.eval()
    rows = dataset(args.data, "dev")
    random.Random(31415).shuffle(rows)
    rows = rows[:args.limit]
    manifest = json.loads((args.data / "manifest.json").read_text())
    if file_sha256(args.data / "counts.pt") != manifest["counts_sha256"]:
        raise ValueError("Count artifact changed")
    counts = torch.load(args.data / "counts.pt", weights_only=True)
    cells = []
    for steps in args.steps:
        for temperature in args.baseline_temperatures:
            cells.append(dict(steps=steps, temperature=temperature, strength=0., sampling="joint"))
        for temperature in args.temperatures:
            for strength in args.strengths:
                for sampling in ("joint", "own-marginal"):
                    cells.append(dict(steps=steps, temperature=temperature, strength=strength, sampling=sampling))
    summaries = []
    for cell in cells:
        name = f"T{cell['steps']}-tau{cell['temperature']}-lambda{cell['strength']}-{cell['sampling']}"
        output = args.output / (name + ".json")
        receipt = args.output / (name + "-identity.json")
        identity = dict(cell, model_sha256=model_sha, data_sha256=data_sha,
                        development_ids=[r["id"] for r in rows], seed=args.seed,
                        source_sha256={p: file_sha256(Path(__file__).parent / p) for p in
                            ("iwslt_compare.py", "translation.py", "iwslt_run.py", "generation.py")})
        if receipt.exists():
            if json.loads(receipt.read_text()) != identity or not output.exists():
                raise ValueError("Existing comparison identity differs: " + name)
            summary = json.loads(output.read_text())["summary"]
        else:
            if output.exists():
                raise ValueError("Unverified output exists without completion identity: " + name)
            decoder = ChainDecoder(mask_id=MASK_ID, device="cuda", fast_candidates=True,
                count_statistics=counts, count_strength=cell["strength"],
                sampling=cell["sampling"], temperature=cell["temperature"])
            decoder.prepare()
            try:
                summary = score_translations(model, tokenizer, head, rows, decoder, output,
                    steps=cell["steps"], seed=args.seed)
            finally:
                decoder.close()
            if file_sha256(args.trained) != model_sha:
                raise ValueError("Trained checkpoint changed during comparison")
            receipt.write_text(json.dumps(identity, indent=2) + "\n")
        summaries.append(dict(cell, **summary))
        log("comparison_cell", **cell, **summary)
        (args.output / "summary.json").write_text(json.dumps(dict(
            completed=len(summaries), expected=len(cells), cells=summaries,
            model_sha256=model_sha, data_sha256=data_sha), indent=2) + "\n")


if __name__ == "__main__":
    main()
