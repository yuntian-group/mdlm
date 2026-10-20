"""Development-only temperature, reveal-schedule and length diagnostics.

The existing test selection is never changed. All subsets are fixed before
scoring. Joint and own-marginal sampling share a temperature and pair strength.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time

from .backbone import file_sha256
from .iwslt_confirm import atomic_json, cell_name, dev_cells, ordered_rows, read

STEPS = (8, 32)
SEEDS = (101, 202, 303)
METHODS = ("baseline", "joint", "own-marginal")
FILES = ("iwslt_followup.py", "iwslt_spans.py", "iwslt_run.py", "translation.py",
         "generation.py", "chain.py", "counts.py", "candidates.py", "fast_candidates.py",
         "backbone.py")


def source_hashes():
    return {p: file_sha256(Path(__file__).parent / p) for p in FILES}


def calibration_cells(steps):
    if steps not in STEPS:
        raise ValueError("Unregistered calibration step count")
    return [dict(steps=steps, temperature=.85, strength=s, sampling=m, seed=101)
            for s in (.1, .2) for m in ("joint", "own-marginal")]


def select_settings(cells):
    expected = [c for t in STEPS for c in dev_cells(t) + calibration_cells(t)]
    if (len(cells) != len(expected) or
            {cell_name(c) for c in cells} != {cell_name(c) for c in expected}):
        raise ValueError("Incomplete or duplicate calibration grid")
    if any(not math.isfinite(c["bleu"]) for c in cells):
        raise ValueError("Nonfinite development score")
    selected = {}
    for t in STEPS:
        group = [c for c in cells if c["steps"] == t]
        baseline = max((c for c in group if not c["strength"]),
                       key=lambda c: (c["bleu"], -c["temperature"]))
        scores = {(tau, s): sum(c["bleu"] for c in group
                  if c["temperature"] == tau and c["strength"] == s) / 2
                  for tau in (.85, 1.) for s in (.1, .2)}
        tau, strength = max(scores, key=lambda k: (scores[k], -k[0], -k[1]))
        selected[str(t)] = dict(baseline_temperature=baseline["temperature"],
            crf_temperature=tau, strength=strength, mean_crf_dev_bleu=scores[tau, strength])
    return selected


def stratified_rows(rows, limit=1024):
    """Proportional allocation in reference-length bins, without model scores."""
    if limit < 1 or limit > len(rows):
        raise ValueError("Invalid development subset size")
    groups = [[r for r in rows if (0 if len(r["target_ids"]) + 1 <= 15 else
               1 if len(r["target_ids"]) + 1 <= 30 else 2) == b] for b in range(3)]
    exact = [limit * len(g) / len(rows) for g in groups]
    sizes = [int(x) for x in exact]
    for b in sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), i))[:limit-sum(sizes)]:
        sizes[b] += 1
    chosen = []
    for b, (group, size) in enumerate(zip(groups, sizes)):
        group = sorted(group, key=lambda r: r["id"])
        random.Random(20261020 + b).shuffle(group)
        chosen.extend(group[:size])
    random.Random(20261020).shuffle(chosen)
    return chosen


def followup_cells(stage, steps, selection):
    if stage not in ("low", "length") or steps not in ((2, 4) if stage == "low" else STEPS):
        raise ValueError("Unregistered development diagnostic")
    # No new tuning at 2/4 steps: carry over the 8-step choice.
    setting = selection["selected"][str(8 if stage == "low" else steps)]
    cells = []
    for seed in SEEDS if stage == "low" else (101,):
        for method in METHODS:
            for mode in ("predicted",) if stage == "low" else ("predicted", "reference-diagnostic"):
                cells.append(dict(steps=steps, temperature=setting["baseline_temperature"]
                    if method == "baseline" else setting["crf_temperature"],
                    strength=0. if method == "baseline" else setting["strength"],
                    sampling="joint" if method == "baseline" else method,
                    seed=seed, method=method, length_mode=mode))
    return cells


def output_name(cell):
    return cell_name(cell) + "-" + cell.get("length_mode", "predicted")


def opportunity(masked, selected):
    if len(masked) != len(selected) or any(s and not m for m, s in zip(masked, selected)):
        raise ValueError("Selected positions must be masked")
    counts, count = [], None
    for m, s in zip(masked + [False], selected + [False]):
        if m:
            count = (0 if count is None else count) + int(s)
        elif count is not None:
            counts.append(count)
            count = None
    return dict(revealed=sum(selected), masked=sum(masked),
                component_reveals=counts,
                adjacent_co_reveals=sum(a and b for a, b in zip(selected, selected[1:])))


def pair_information(node, edge, selected):
    """Exact dependence between the first two revealed sites in each component.

    Multiplying posterior transitions includes intervening latent sites. Thus
    separated selected sites are not incorrectly assumed to be independent.
    """
    import torch
    if len(selected) < 2:
        return None
    left, right = selected[:2]
    joint = torch.diag(node[left])
    for i in range(left, right):
        transition = edge[i] / node[i, :, None].clamp_min(1e-300)
        joint = joint @ transition
    independent = node[left, :, None] * node[right, None, :]
    live = joint > 0
    mi = (joint[live] * (joint[live].log() - independent[live].clamp_min(1e-300).log())).sum()
    return dict(left=left, right=right, distance=right-left,
                mutual_information_nats=float(mi),
                total_variation=float((joint-independent).abs().sum()/2))


class StateObserver:
    def __init__(self, decoder, probe=False):
        self.decoder, self.probe = decoder, probe
        self.steps = []
        self.probe_seconds = 0.

    def __call__(self, tokens, output, reveal):
        import torch
        from .chain import marginals
        masked = tokens[0].eq(self.decoder.mask_id).tolist()
        selected = reveal[0].tolist()
        record = opportunity(masked, selected)
        # One same-state probe per sentence; separate from sampling RNG streams.
        if self.probe and not self.steps:
            from .generation import synchronize
            synchronize(self.decoder.device)
            probe_start = time.perf_counter()
            pairs = []
            start = None
            for end, m in enumerate(masked + [False]):
                if m and start is None:
                    start = end
                if not m and start is not None:
                    positions = [i for i in range(start, end) if selected[i]]
                    if len(positions) >= 2:
                        lo, hi = max(0, start-1), min(len(masked), end+1)
                        c = self.decoder._candidates(output.log_probs[:, lo:hi], tokens[:, lo:hi])
                        safe = c.candidate_ids.clamp_max(self.decoder.potential.vocab_size-1)
                        pair = self.decoder.potential.log_scores(safe[:, :-1, :, None], safe[:, 1:, None, :])
                        posterior = marginals(c.log_unary, pair, return_edges=True)
                        value = pair_information(posterior.node[0], posterior.edge[0],
                                                 [i-lo for i in positions])
                        pairs.append(dict(component_start=start, component_end=end, **value))
                    start = None
            record["same_state_pairs"] = pairs
            synchronize(self.decoder.device)
            self.probe_seconds += time.perf_counter() - probe_start
        self.steps.append(record)


def verify_output(path, identity):
    receipt = read(path.with_name(path.stem + "-identity.json"))
    if receipt["identity"] != identity or receipt["output_sha256"] != file_sha256(path):
        raise ValueError("Output identity/hash changed")
    result = read(path)
    if [r["id"] for r in result["predictions"]] != identity["ids"]:
        raise ValueError("Incomplete or reordered output")
    if result["summary"]["examples"] != len(identity["ids"]):
        raise ValueError("Incomplete output summary")
    return result


def score(model, tokenizer, head, rows, decoder, cell):
    from sacrebleu.metrics import BLEU, CHRF
    from sacremoses import MosesDetokenizer
    from .generation import synchronize
    from .iwslt_run import log
    from .translation import translate_one
    detok, records = MosesDetokenizer(lang="en"), []
    for i, row in enumerate(rows):
        observer = StateObserver(decoder, probe=cell["method"] == "joint" and i < 32
                                 and cell["seed"] == 101)
        length = len(row["target_ids"]) + 1 if cell["length_mode"] == "reference-diagnostic" else None
        synchronize(decoder.device)
        start = time.perf_counter()
        ids, stats = translate_one(model, row, head, decoder, steps=cell["steps"],
            seed=cell["seed"], example_id=i, length_override=length, observer=observer)
        synchronize(decoder.device)
        records.append(dict(id=row["id"], hypothesis=detok.detokenize(tokenizer.decode(ids).split()),
            reference=detok.detokenize(row["target"].split()), token_ids=ids,
            seconds=time.perf_counter()-start, same_state_probe_seconds=observer.probe_seconds,
            **stats, opportunities=observer.steps))
        if (i+1) % 32 == 0:
            log("followup_progress", completed=i+1, total=len(rows), **cell)
    hypotheses, references = [r["hypothesis"] for r in records], [r["reference"] for r in records]
    bleu, chrf = BLEU(lowercase=True, tokenize="13a"), CHRF()
    summary = dict(examples=len(records), bleu=bleu.corpus_score(hypotheses, [references]).score,
        chrf=chrf.corpus_score(hypotheses, [references]).score,
        seconds=sum(r["seconds"] for r in records),
        same_state_probe_seconds=sum(r["same_state_probe_seconds"] for r in records),
        timing_includes_diagnostics=True)
    return dict(summary=summary, predictions=records)


def decode(args):
    import torch
    from .generation import ChainDecoder
    from .iwslt_run import log, run_lock, score_translations
    from .translation import LengthPredictor, load_conditional, MASK_ID
    with run_lock(args.output):
        model_sha, data_sha = file_sha256(args.trained), file_sha256(args.data / "manifest.json")
        rows = ordered_rows(args.data, "dev")
        selection_sha = None
        if args.stage == "calibration":
            if args.selection:
                raise ValueError("Calibration must not consume a selection")
            cells = calibration_cells(args.steps)
        else:
            if not args.selection:
                raise ValueError("Diagnostics require completed development calibration")
            selection = read(args.selection)
            if (selection["model_sha256"], selection["data_sha256"], selection["source_sha256"]) != (
                    model_sha, data_sha, source_hashes()):
                raise ValueError("Selection model/data/code changed")
            if select_settings(selection["cells"]) != selection["selected"]:
                raise ValueError("Selection does not follow development scores")
            selection_sha = file_sha256(args.selection)
            cells = followup_cells(args.stage, args.steps, selection)
            rows = stratified_rows(rows, 1024 if args.stage == "low" else 512)
        counts_path = args.data / "counts.pt"
        if file_sha256(counts_path) != read(args.data / "manifest.json")["counts_sha256"]:
            raise ValueError("Training counts changed")
        model, tok = load_conditional(args.checkpoint, args.tokenizer)
        state = torch.load(args.trained, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        head = LengthPredictor(model.pretrained.config.hidden_dim).cuda()
        head.load_state_dict(state["length_head"])
        del state
        model.eval()
        head.eval()
        counts = torch.load(counts_path, weights_only=True)
        done = []
        log("followup_started", stage=args.stage, steps=args.steps, examples=len(rows),
            cells=len(cells), gpu=torch.cuda.get_device_name())
        for cell in cells:
            path = args.output / (output_name(cell) + ".json")
            identity = dict(cell=cell, stage=args.stage, split="dev", ids=[r["id"] for r in rows],
                model_sha256=model_sha, data_sha256=data_sha, source_sha256=source_hashes(),
                selection_sha256=selection_sha)
            if path.with_name(path.stem + "-identity.json").exists():
                result = verify_output(path, identity)
            else:
                if path.exists():
                    raise ValueError("Output without completion receipt")
                decoder = ChainDecoder(mask_id=MASK_ID, device="cuda", fast_candidates=True,
                    count_statistics=counts, count_strength=cell["strength"],
                    sampling=cell["sampling"], temperature=cell["temperature"])
                decoder.prepare()
                try:
                    if args.stage == "calibration":
                        pending = path.with_suffix(".pending.json")
                        score_translations(model, tok, head, rows, decoder, pending,
                                           steps=cell["steps"], seed=cell["seed"])
                        result = read(pending)
                        pending.unlink()
                    else:
                        result = score(model, tok, head, rows, decoder, cell)
                finally:
                    decoder.close()
                if file_sha256(args.trained) != model_sha:
                    raise ValueError("Model changed during decoding")
                if args.selection and file_sha256(args.selection) != selection_sha:
                    raise ValueError("Selection changed during decoding")
                atomic_json(path, result)
                atomic_json(path.with_name(path.stem + "-identity.json"),
                            dict(identity=identity, output_sha256=file_sha256(path)))
            done.append(dict(cell, **result["summary"]))
            atomic_json(args.output / "summary.json", dict(completed=len(done), expected=len(cells), cells=done))
            log("followup_cell", stage=args.stage, **done[-1])


def freeze(args):
    from sacrebleu.metrics import BLEU
    from .iwslt_run import run_lock
    with run_lock(args.output):
        model_sha, data_sha = file_sha256(args.trained), file_sha256(args.data / "manifest.json")
        ids = [r["id"] for r in ordered_rows(args.data, "dev")]
        old = read(args.previous_selection)
        if old["model_sha256"] != model_sha or old["data_sha256"] != data_sha:
            raise ValueError("Previous model/data differ")
        cells, artifacts = [], {}
        for t in STEPS:
            for cell in dev_cells(t):
                path = args.previous_development / f"step-{t}" / (cell_name(cell)+".json")
                key = str(path.relative_to(args.previous_development))
                if file_sha256(path) != old["development_artifacts"][key]:
                    raise ValueError("Previous development output changed")
                receipt = path.with_name(path.stem + "-identity.json")
                receipt_key = str(receipt.relative_to(args.previous_development))
                if file_sha256(receipt) != old["development_artifacts"][receipt_key]:
                    raise ValueError("Previous receipt changed")
                prior_identity = dict(cell, split="dev", ids=ids, model_sha256=model_sha,
                    data_sha256=data_sha, source_sha256=old["source_sha256"], selection_sha256=None)
                payload = read(path)
                r = read(receipt)
                if r["identity"] != prior_identity or r["output_sha256"] != file_sha256(path):
                    raise ValueError("Previous cell identity differs")
                cells.append(_scored_cell(cell, payload, ids, BLEU))
                artifacts[str(path)] = file_sha256(path)
            for cell in calibration_cells(t):
                path = args.development / f"step-{t}" / (output_name(cell)+".json")
                identity = dict(cell=cell, stage="calibration", split="dev", ids=ids,
                    model_sha256=model_sha, data_sha256=data_sha, source_sha256=source_hashes(),
                    selection_sha256=None)
                payload = verify_output(path, identity)
                cells.append(_scored_cell(cell, payload, ids, BLEU))
                artifacts[str(path)] = file_sha256(path)
        selection = dict(selected=select_settings(cells), cells=cells, development_artifacts=artifacts,
            model_sha256=model_sha, data_sha256=data_sha, source_sha256=source_hashes(),
            policy="Development only; common CRF temperature/strength maximizes mean joint and own-marginal BLEU. Low-step runs carry the 8-step choice. Original test selection remains unchanged.")
        path = args.output / "selection.json"
        if path.exists() and read(path) != selection:
            raise ValueError("Refusing to overwrite selection")
        atomic_json(path, selection)
        print(json.dumps(selection["selected"]), flush=True)


def _scored_cell(cell, payload, ids, metric):
    predictions = payload["predictions"]
    if [r["id"] for r in predictions] != ids:
        raise ValueError("Incomplete or reordered development predictions")
    score_value = metric(lowercase=True, tokenize="13a").corpus_score(
        [r["hypothesis"] for r in predictions], [[r["reference"] for r in predictions]]).score
    if abs(score_value-payload["summary"]["bleu"]) > 1e-10:
        raise ValueError("Development BLEU mismatch")
    return dict(cell, bleu=score_value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("decode")
    for name in ("checkpoint", "tokenizer", "trained", "data", "output"):
        d.add_argument("--"+name, type=Path, required=True)
    d.add_argument("--stage", choices=("calibration", "low", "length"), required=True)
    d.add_argument("--steps", type=int, required=True)
    d.add_argument("--selection", type=Path)
    f = sub.add_parser("freeze")
    for name in ("trained", "data", "output", "previous-selection", "previous-development", "development"):
        f.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    (decode if args.command == "decode" else freeze)(args)


if __name__ == "__main__":
    main()
