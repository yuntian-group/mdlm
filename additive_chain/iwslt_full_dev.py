"""Fixed-setting full-development comparison with the refitted length head."""
import argparse
from pathlib import Path

from .backbone import file_sha256
from .iwslt_confirm import atomic_json, ordered_rows, read
from .iwslt_controls import control_source_hashes, head_fingerprint, validate_selection
from .iwslt_followup import METHODS, output_name, score, verify_output

STEPS = (32, 8, 4, 2)  # Submit the longest jobs first.


def full_cells():
    return [dict(steps=steps, temperature=.85,
                 strength=0. if method == "baseline" else .2,
                 sampling="joint" if method == "baseline" else method,
                 seed=101, method=method, length_mode="predicted")
            for steps in STEPS for method in METHODS]


def source_hashes():
    result = control_source_hashes("new-length")
    for name in ("iwslt_full_dev.py", "iwslt_confirm.py"):
        result[name] = file_sha256(Path(__file__).parent / name)
    return result


def build_identity(cell, rows, model_sha, data_sha, selection_sha,
                   head_sha, head_artifact_sha, old_head_sha):
    if cell not in full_cells():
        raise ValueError("Unregistered full-development cell")
    return dict(cell=cell, experiment="full-dev-refit", split="dev",
        ids=[r["id"] for r in rows], model_sha256=model_sha,
        data_sha256=data_sha, selection_sha256=selection_sha,
        source_sha256=source_hashes(), nucleus_p=.9, top_k=500,
        head_sha256=head_sha, head_artifact_sha256=head_artifact_sha,
        old_head_sha256=old_head_sha)


def check_fixed_selection(selection, model_sha, data_sha):
    validate_selection(selection, model_sha, data_sha)
    for setting in selection["selected"].values():
        if (setting["baseline_temperature"], setting["crf_temperature"],
                setting["strength"]) != (.85, .85, .2):
            raise ValueError("The registered settings are fixed, not retuned")


def run(args):
    import torch
    from .generation import ChainDecoder
    from .iwslt_run import log, run_lock
    from .iwslt_length_refit import load_refit_head
    from .translation import load_conditional, MASK_ID
    cells = full_cells()
    if not 0 <= args.cell_index < len(cells):
        raise ValueError("Unregistered full-development cell index")
    cell = cells[args.cell_index]
    folder = args.output / output_name(cell)
    with run_lock(folder):
        model_sha = file_sha256(args.trained)
        data_sha = file_sha256(args.data / "manifest.json")
        selection_sha = file_sha256(args.selection)
        selection = read(args.selection)
        check_fixed_selection(selection, model_sha, data_sha)
        rows = ordered_rows(args.data, "dev")
        counts_path = args.data / "counts.pt"
        if file_sha256(counts_path) != read(args.data / "manifest.json")["counts_sha256"]:
            raise ValueError("Training counts changed")
        head_artifact_sha = file_sha256(args.length_head)
        model, tokenizer = load_conditional(args.checkpoint, args.tokenizer)
        state = torch.load(args.trained, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        old_head_sha = head_fingerprint(state["length_head"])
        del state
        model.eval()
        head = load_refit_head(args.length_head, model_sha, data_sha, "cuda")
        head_sha = head_fingerprint(head.state_dict())
        identity = build_identity(cell, rows, model_sha, data_sha, selection_sha,
                                  head_sha, head_artifact_sha, old_head_sha)
        path = folder / "result.json"
        if (folder / "result-identity.json").exists():
            result = verify_output(path, identity)
        else:
            if path.exists():
                raise ValueError("Output without completion receipt")
            counts = torch.load(counts_path, weights_only=True)
            decoder = ChainDecoder(mask_id=MASK_ID, device="cuda", top_k=500,
                nucleus_p=.9, fast_candidates=True, count_statistics=counts,
                count_strength=cell["strength"], sampling=cell["sampling"],
                temperature=cell["temperature"])
            log("full_dev_started", **cell, examples=len(rows),
                gpu=torch.cuda.get_device_name(), head_sha256=head_sha)
            decoder.prepare()
            try:
                result = score(model, tokenizer, head, rows, decoder, cell)
            finally:
                decoder.close()
            if (file_sha256(args.trained) != model_sha or
                    file_sha256(args.selection) != selection_sha or
                    file_sha256(args.length_head) != head_artifact_sha or
                    file_sha256(args.data / "manifest.json") != data_sha):
                raise ValueError("Input changed during decoding")
            atomic_json(path, result)
            atomic_json(folder / "result-identity.json",
                        dict(identity=identity, output_sha256=file_sha256(path)))
        atomic_json(folder / "summary.json",
                    dict(completed=1, expected=1, cell=cell, **result["summary"]))
        log("full_dev_complete", **cell, **result["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "trained", "data", "output",
                 "selection", "length-head"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
