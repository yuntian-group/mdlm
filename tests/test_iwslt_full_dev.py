import pytest

from additive_chain import iwslt_full_dev as full


def test_fixed_full_grid_one_seed_no_retuning():
    cells = full.full_cells()
    assert len(cells) == 12
    assert [c["steps"] for c in cells] == [32]*3 + [8]*3 + [4]*3 + [2]*3
    assert {c["seed"] for c in cells} == {101}
    assert {c["temperature"] for c in cells} == {.85}
    assert {c["length_mode"] for c in cells} == {"predicted"}
    for c in cells:
        assert c["strength"] == (0. if c["method"] == "baseline" else .2)
        assert c["sampling"] == ("joint" if c["method"] == "baseline" else c["method"])


def test_identity_preserves_full_order_and_all_head_identities():
    rows = [dict(id=f"dev-{i}") for i in range(7283)]
    identity = full.build_identity(full.full_cells()[0], rows,
                                  "model", "data", "selection", "head", "artifact", "old")
    assert identity["ids"] == [r["id"] for r in rows]
    assert identity["split"] == "dev"
    assert identity["nucleus_p"] == .9 and identity["top_k"] == 500
    assert identity["head_sha256"] == "head"
    assert identity["head_artifact_sha256"] == "artifact"
    assert identity["old_head_sha256"] == "old"
    assert "iwslt_full_dev.py" in identity["source_sha256"]
    assert "iwslt_confirm.py" in identity["source_sha256"]
    with pytest.raises(ValueError, match="Unregistered"):
        full.build_identity(dict(full.full_cells()[0], seed=202), rows,
                            "model", "data", "selection", "head", "artifact", "old")


def test_changed_selection_rejected_even_if_valid_elsewhere(monkeypatch):
    monkeypatch.setattr(full, "validate_selection", lambda *a: None)
    setting = dict(baseline_temperature=.85, crf_temperature=.85, strength=.2)
    full.check_fixed_selection(dict(selected={"8": setting, "32": setting}), "m", "d")
    for key, value in (("strength", .1), ("baseline_temperature", 1.),
                       ("crf_temperature", 1.)):
        with pytest.raises(ValueError, match="fixed"):
            full.check_fixed_selection(dict(selected={"8": dict(setting, **{key: value})}), "m", "d")
