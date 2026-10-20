"""Local learned-head assets for the development-only IWSLT experiment."""
from pathlib import Path

import torch

from .backbone import file_sha256
from .heads import ContextualAdditiveHead, GlobalAdditiveHead

FORMAT = "iwslt_additive_residual_v1"
PROTOCOL = dict(temperature=.85, count_strength=.2, residual_strength=1.,
                nucleus_p=.9, top_k=500)
SOURCE_FILES = ("iwslt_residual_assets.py", "iwslt_residual_objective.py",
    "iwslt_residual_train.py", "translation.py", "heads.py", "adaptation.py",
    "training.py", "chain.py", "composition.py", "counts.py", "fast_candidates.py",
    "candidates.py", "backbone.py")


def residual_source_hashes():
    return {name: file_sha256(Path(__file__).parent / name) for name in SOURCE_FILES}


def make_residual_head(kind, model, *, rank=64, context_size=128):
    table = model.backbone.vocab_embed.embedding.detach()
    if kind == "global":
        return GlobalAdditiveHead(table.shape[0], rank).to(table.device)
    if kind == "contextual":
        return ContextualAdditiveHead(table, hidden_size=table.shape[1],
            rank=rank, context_size=context_size).to(table.device)
    raise ValueError("Unknown residual kind")


def head_checkpoint(head, kind, identity, *, checkpoint_id, epoch, update, metrics):
    return dict(format=FORMAT, kind=kind, rank=head.rank,
        vocab_size=head.vocab_size,
        hidden_size=getattr(head, "hidden_size", 768),
        context_size=getattr(head, "context_size", 128),
        identity=identity, checkpoint_id=checkpoint_id, epoch=epoch, update=update,
        metrics=metrics,
        head_state={name: value.detach().cpu().clone()
                    for name, value in head.state_dict().items()})


def load_residual_head(path, model_sha, data_sha, device="cuda"):
    path = Path(path)
    before = file_sha256(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    identity = saved.get("identity", {})
    if (saved.get("format") != FORMAT or saved.get("kind") not in ("global", "contextual")
            or identity.get("model_sha256") != model_sha
            or identity.get("data_sha256") != data_sha
            or any(identity.get(key) != value for key, value in PROTOCOL.items())):
        raise ValueError("Residual model/data/protocol identity changed")
    if (type(saved.get("rank")) is not int or not 1 <= saved["rank"] <= 128
            or type(saved.get("vocab_size")) is not int or saved["vocab_size"] < 2
            or any(type(saved.get(k)) is not int or saved[k] < 0 for k in ("epoch", "update"))
            or not isinstance(saved.get("checkpoint_id"), str)
            or not isinstance(saved.get("head_state"), dict)):
        raise ValueError("Invalid residual checkpoint metadata")
    state = saved["head_state"]
    if any(not isinstance(t, torch.Tensor) or t.dtype != torch.float32
           or not bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError("Residual tensors must be finite FP32")
    if saved["kind"] == "global":
        if identity.get("inherited_global_sha256") is not None:
            raise ValueError("Global head cannot inherit another global head")
        head = GlobalAdditiveHead(saved["vocab_size"], saved["rank"])
    else:
        table = state.get("token_embeddings")
        if (table is None or table.shape != (saved["vocab_size"], saved["hidden_size"])
                or type(saved.get("context_size")) is not int or saved["context_size"] < 1):
            raise ValueError("Invalid contextual embedding table")
        head = ContextualAdditiveHead(table, hidden_size=saved["hidden_size"],
            rank=saved["rank"], context_size=saved["context_size"])
    head.load_state_dict(state, strict=True)
    if file_sha256(path) != before:
        raise ValueError("Residual checkpoint changed while loading")
    head.to(device).eval().requires_grad_(False)
    return head, {key: value for key, value in saved.items() if key != "head_state"}


@torch.no_grad()
def dense_count_scores(potential, *, row_chunk=256):
    """One frozen FP32 V² log-score table; a GPU-memory optimization, not a model change."""
    if row_chunk < 1:
        raise ValueError("Positive row chunk required")
    v = potential.vocab_size
    result = torch.empty((v, v), dtype=torch.float32, device=potential.left.device)
    left, right = potential.left.log(), potential.right.log()
    for start in range(0, v, row_chunk):
        result[start:start + row_chunk] = (
            left[start:start + row_chunk, None] + right[None]).float()
    rows, columns = potential.pair_keys // v, potential.pair_keys % v
    result[rows, columns] = (
        potential.left[rows] * potential.right[columns] + potential.row_values).log().float()
    return result
