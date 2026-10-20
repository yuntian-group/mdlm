"""Paired development controls for candidate support and source length prediction."""
import argparse
import hashlib
import json
from pathlib import Path

from .backbone import file_sha256
from .iwslt_confirm import atomic_json, ordered_rows, read
from .iwslt_followup import (followup_cells, output_name, score, select_settings,
                             source_hashes, stratified_rows, verify_output)


def control_cells(experiment, steps, selection):
    if experiment not in ("p1", "new-length"):
        raise ValueError("Unknown control experiment")
    if steps not in ((2, 4) if experiment == "p1" else (2, 4, 8, 32)):
        raise ValueError("Unregistered control step count")
    stage = "low" if steps in (2, 4) else "length"
    return [c for c in followup_cells(stage, steps, selection)
            if c["length_mode"] == "predicted"]


def head_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps((name, str(value.dtype), list(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def control_source_hashes(experiment):
    result = source_hashes()
    result["iwslt_controls.py"] = file_sha256(Path(__file__))
    if experiment == "new-length":
        result["iwslt_length_refit.py"] = file_sha256(
            Path(__file__).with_name("iwslt_length_refit.py"))
    return result


def validate_selection(selection, model_sha, data_sha):
    if (selection["model_sha256"], selection["data_sha256"],
            selection["source_sha256"]) != (model_sha, data_sha, source_hashes()):
        raise ValueError("Selection model/data/code changed")
    if select_settings(selection["cells"]) != selection["selected"]:
        raise ValueError("Selection does not follow development scores")


def run(args):
    import torch
    from .generation import ChainDecoder
    from .iwslt_run import log, run_lock
    from .translation import LengthPredictor, load_conditional, MASK_ID
    if (args.experiment == "new-length") != (args.length_head is not None):
        raise ValueError("Only the new-length control requires a refitted head")
    with run_lock(args.output):
        model_sha = file_sha256(args.trained)
        data_sha = file_sha256(args.data / "manifest.json")
        selection_sha = file_sha256(args.selection)
        selection = read(args.selection)
        validate_selection(selection, model_sha, data_sha)
        cells = control_cells(args.experiment, args.steps, selection)
        rows = stratified_rows(ordered_rows(args.data, "dev"),
                               1024 if args.steps in (2, 4) else 512)
        counts_path = args.data / "counts.pt"
        if file_sha256(counts_path) != read(args.data / "manifest.json")["counts_sha256"]:
            raise ValueError("Training counts changed")
        model, tokenizer = load_conditional(args.checkpoint, args.tokenizer)
        state = torch.load(args.trained, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        old_head_sha = head_fingerprint(state["length_head"])
        head_artifact_sha = file_sha256(args.length_head) if args.length_head else None
        if args.length_head:
            from .iwslt_length_refit import load_refit_head
            head = load_refit_head(args.length_head, model_sha, data_sha, "cuda")
        else:
            head = LengthPredictor(model.pretrained.config.hidden_dim).cuda()
            head.load_state_dict(state["length_head"])
        del state
        head.eval()
        model.eval()
        head_sha = head_fingerprint(head.state_dict())
        counts = torch.load(counts_path, weights_only=True)
        nucleus_p = 1. if args.experiment == "p1" else .9
        source = control_source_hashes(args.experiment)
        done = []
        log("control_started", experiment=args.experiment, steps=args.steps,
            examples=len(rows), cells=len(cells), nucleus_p=nucleus_p, top_k=500,
            gpu=torch.cuda.get_device_name(), head_sha256=head_sha)
        for cell in cells:
            path = args.output / (output_name(cell) + ".json")
            identity = dict(cell=cell, experiment=args.experiment, split="dev",
                ids=[r["id"] for r in rows], model_sha256=model_sha,
                data_sha256=data_sha, source_sha256=source,
                selection_sha256=selection_sha, nucleus_p=nucleus_p, top_k=500,
                head_sha256=head_sha, old_head_sha256=old_head_sha,
                head_artifact_sha256=head_artifact_sha)
            if path.with_name(path.stem + "-identity.json").exists():
                result = verify_output(path, identity)
            else:
                if path.exists():
                    raise ValueError("Output without completion receipt")
                decoder = ChainDecoder(mask_id=MASK_ID, device="cuda", top_k=500,
                    nucleus_p=nucleus_p, fast_candidates=True, count_statistics=counts,
                    count_strength=cell["strength"], sampling=cell["sampling"],
                    temperature=cell["temperature"])
                decoder.prepare()
                try:
                    result = score(model, tokenizer, head, rows, decoder, cell)
                finally:
                    decoder.close()
                if (file_sha256(args.trained) != model_sha or
                        file_sha256(args.selection) != selection_sha or
                        (args.length_head and file_sha256(args.length_head) != head_artifact_sha)):
                    raise ValueError("Input changed during decoding")
                atomic_json(path, result)
                atomic_json(path.with_name(path.stem + "-identity.json"),
                            dict(identity=identity, output_sha256=file_sha256(path)))
            done.append(dict(cell, **result["summary"]))
            atomic_json(args.output / "summary.json",
                        dict(completed=len(done), expected=len(cells), cells=done))
            log("control_cell", experiment=args.experiment, **done[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "trained", "data", "output", "selection"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--length-head", type=Path)
    parser.add_argument("--experiment", choices=("p1", "new-length"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
