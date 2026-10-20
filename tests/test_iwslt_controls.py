import pytest
import torch

from additive_chain.iwslt_confirm import dev_cells
from additive_chain.iwslt_followup import calibration_cells, followup_cells, select_settings
from additive_chain.iwslt_controls import control_cells, head_fingerprint, validate_selection


def selection():
    cells = [dict(c, bleu=10 + (c["temperature"] == .85) + c["strength"])
             for t in (8, 32) for c in dev_cells(t) + calibration_cells(t)]
    return dict(cells=cells, selected=select_settings(cells))


def test_p1_preserves_old_low_grid():
    frozen = selection()
    for steps in (2, 4):
        assert control_cells("p1", steps, frozen) == followup_cells("low", steps, frozen)
        assert len(control_cells("p1", steps, frozen)) == 9
    with pytest.raises(ValueError):
        control_cells("p1", 8, frozen)


def test_new_head_is_predicted_only_and_preserves_settings():
    frozen = selection()
    assert sum(len(control_cells("new-length", t, frozen)) for t in (2, 4, 8, 32)) == 24
    for steps in (2, 4, 8, 32):
        cells = control_cells("new-length", steps, frozen)
        assert {c["length_mode"] for c in cells} == {"predicted"}
        assert {c["temperature"] for c in cells} == {.85}
        assert {c["strength"] for c in cells} == {0., .2}
        for method in ("baseline", "joint", "own-marginal"):
            assert any(c["method"] == method for c in cells)
    with pytest.raises(ValueError):
        control_cells("test", 2, frozen)
    with pytest.raises(ValueError):
        control_cells("new-length", 16, frozen)


def test_head_fingerprint_checks_values_shapes_and_names():
    x = torch.arange(6).float()
    assert head_fingerprint({"a": x}) == head_fingerprint({"a": x.clone()})
    assert head_fingerprint({"a": x}) != head_fingerprint({"a": x.reshape(2, 3)})
    assert head_fingerprint({"a": x}) != head_fingerprint({"a": x + 1})
    assert head_fingerprint({"a": x}) != head_fingerprint({"b": x})
    assert head_fingerprint({"a": x, "b": x}) == head_fingerprint({"b": x, "a": x})


def test_selection_cannot_be_retuned_or_reassigned(monkeypatch):
    from additive_chain import iwslt_controls as controls
    monkeypatch.setattr(controls, "source_hashes", lambda: {"source": "sha"})
    frozen = dict(selection(), model_sha256="model", data_sha256="data",
                  source_sha256={"source": "sha"})
    validate_selection(frozen, "model", "data")
    with pytest.raises(ValueError, match="changed"):
        validate_selection(frozen, "other", "data")
    frozen["selected"]["8"]["strength"] = .1
    with pytest.raises(ValueError, match="scores"):
        validate_selection(frozen, "model", "data")
