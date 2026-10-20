import copy

import pytest
import torch

from additive_chain.counts import SparseCountPotential
from additive_chain.heads import ContextualAdditiveHead, GlobalAdditiveHead
from additive_chain.iwslt_residual_assets import (
    PROTOCOL, dense_count_scores, head_checkpoint, load_residual_head)


@pytest.mark.parametrize("kind", ["global", "contextual"])
def test_residual_checkpoint_roundtrip(tmp_path, kind):
    head = (GlobalAdditiveHead(7, 2) if kind == "global" else
            ContextualAdditiveHead(torch.randn(7, 4), rank=2, context_size=3))
    identity = dict(PROTOCOL, model_sha256="model", data_sha256="data",
                    inherited_global_sha256=None)
    value = head_checkpoint(head, kind, identity, checkpoint_id="epoch-0000",
                            epoch=0, update=0, metrics={})
    path = tmp_path / "head.pt"
    torch.save(value, path)
    loaded, metadata = load_residual_head(path, "model", "data", "cpu")
    assert metadata["checkpoint_id"] == "epoch-0000" and "head_state" not in metadata
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert all(torch.equal(t, loaded.state_dict()[k]) for k, t in head.state_dict().items())
    with pytest.raises(ValueError, match="identity"):
        load_residual_head(path, "different-model", "data", "cpu")
    corrupt = copy.deepcopy(value)
    corrupt["identity"]["temperature"] = 1.
    torch.save(corrupt, path)
    with pytest.raises(ValueError, match="identity"):
        load_residual_head(path, "model", "data", "cpu")
    corrupt = copy.deepcopy(value)
    next(iter(corrupt["head_state"].values())).flatten()[0] = torch.nan
    torch.save(corrupt, path)
    with pytest.raises(ValueError, match="finite"):
        load_residual_head(path, "model", "data", "cpu")


def test_dense_count_table_matches_sparse_entries():
    counts = torch.tensor([[0., 3., 1.], [2., 0., 1.], [0., 1., 0.]], dtype=torch.float64)
    keys = counts.flatten().nonzero().flatten()
    potential = SparseCountPotential.from_counts(counts.sum(1), counts.sum(0), keys,
        counts.flatten()[keys], strength=.2)
    table = dense_count_scores(potential, row_chunk=2)
    rows, cols = torch.arange(3)[:, None], torch.arange(3)[None]
    assert table.dtype == torch.float32 and not table.requires_grad
    torch.testing.assert_close(table, potential.entries(rows, cols).log().float())
