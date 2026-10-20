import math
from types import SimpleNamespace

import pytest
import torch

from additive_chain.iwslt_confirm import dev_cells
from additive_chain.iwslt_followup import (
    calibration_cells, followup_cells, opportunity, pair_information,
    select_settings, stratified_rows, StateObserver, verify_output, atomic_json)
from additive_chain.backbone import file_sha256


def grid():
    return [dict(c, bleu=10 + (c["temperature"] == .85) + c["strength"])
            for t in (8, 32) for c in dev_cells(t) + calibration_cells(t)]


def test_calibration_complete_and_shared_selection():
    rows = grid()
    assert len(rows) == 20
    selected = select_settings(rows)
    for value in selected.values():
        assert value["crf_temperature"] == .85
        assert value["strength"] == .2
        assert value["baseline_temperature"] == .85
    for bad in (rows[:-1], rows + rows[:1], rows[:-1] + rows[:1]):
        with pytest.raises(ValueError, match="grid"):
            select_settings(bad)
    rows[0]["bleu"] = math.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        select_settings(rows)


def test_shared_selection_not_cherry_picked_joint_score():
    rows = grid()
    for r in rows:
        if r["strength"] == .1 and r["temperature"] == 1:
            r["bleu"] = 30 if r["sampling"] == "joint" else 0
        elif r["strength"] > 0:
            r["bleu"] = 20
    selected = select_settings(rows)
    assert selected["8"]["crf_temperature"] == .85
    assert selected["8"]["strength"] == .1


def test_registered_grids_and_oracle_counterparts():
    frozen = dict(selected=select_settings(grid()))
    low = [c for t in (2, 4) for c in followup_cells("low", t, frozen)]
    length = [c for t in (8, 32) for c in followup_cells("length", t, frozen)]
    assert len(low) == 18 and len(length) == 12
    assert {c["seed"] for c in low} == {101, 202, 303}
    assert {c["length_mode"] for c in low} == {"predicted"}
    assert {c["seed"] for c in length} == {101}
    for c in length:
        counterpart = dict(c, length_mode="predicted" if c["length_mode"] == "reference-diagnostic"
                           else "reference-diagnostic")
        assert counterpart in length
    with pytest.raises(ValueError, match="Unregistered"):
        followup_cells("test", 8, frozen)
    with pytest.raises(ValueError, match="Unregistered"):
        followup_cells("low", 32, frozen)


def test_subset_proportional_unique_and_order_independent():
    rows = [dict(id=f"{i:04}", target_ids=[0] * (10 if i < 50 else 20 if i < 80 else 40))
            for i in range(100)]
    selected = stratified_rows(rows, 20)
    assert selected == stratified_rows(rows[::-1], 20)
    assert len({r["id"] for r in selected}) == 20
    assert [sum(len(r["target_ids"]) == n for r in selected) for n in (10, 20, 40)] == [10, 6, 4]
    with pytest.raises(ValueError):
        stratified_rows(rows, 101)


def test_reveal_components_include_latent_intermediate_sites():
    result = opportunity([True, True, True, False, True], [True, False, True, False, True])
    assert result["component_reveals"] == [2, 1]
    assert result["adjacent_co_reveals"] == 0
    assert result["revealed"] == 3
    with pytest.raises(ValueError):
        opportunity([False], [True])


def test_same_state_probe_recovers_nonadjacent_dependence():
    node = torch.full((3, 2), .5, dtype=torch.float64)
    edge = torch.tensor([[[.45, .05], [.05, .45]]] * 2, dtype=torch.float64)
    result = pair_information(node, edge, [0, 2])
    # Two transitions: agreement .82, versus .5 under product marginals.
    expected = .82 * math.log(.82/.5) + .18 * math.log(.18/.5)
    assert result["mutual_information_nats"] == pytest.approx(expected)
    assert result["total_variation"] == pytest.approx(.32)
    assert pair_information(node, edge, [1]) is None
    independent = torch.full((2, 2, 2), .25, dtype=torch.float64)
    assert pair_information(node, independent, [0, 2])["mutual_information_nats"] == pytest.approx(0.)


def test_observer_does_not_change_sampling_or_rng():
    from additive_chain.generation import ChainDecoder, sample_window
    mask = 4
    decoder = ChainDecoder(mask_id=mask, device="cpu", top_k=4, nucleus_p=1.)
    decoder.prepare()
    tokens = torch.full((1, 9), mask)
    logp = torch.full((1, 9, 5), -math.log(4), dtype=torch.float32)
    logp[..., mask] = -torch.inf
    forward = lambda current, t: SimpleNamespace(log_probs=logp, hidden=None)
    def run(observer):
        return sample_window(tokens, forward, decoder, steps=4,
            reveal_generator=torch.Generator().manual_seed(10),
            proposal_generator=torch.Generator().manual_seed(11), observer=observer)
    plain = run(None)
    observer = StateObserver(decoder)
    traced = run(observer)
    assert torch.equal(plain.tokens, traced.tokens)
    assert plain.stats == traced.stats
    assert sum(s["revealed"] for s in observer.steps) == tokens.numel()
    decoder.close()


def test_receipt_checks_output_hash_and_rows(tmp_path):
    path = tmp_path / "cell.json"
    identity = dict(ids=["a"])
    payload = dict(summary=dict(examples=1), predictions=[dict(id="a")])
    atomic_json(path, payload)
    atomic_json(tmp_path / "cell-identity.json",
                dict(identity=identity, output_sha256=file_sha256(path)))
    assert verify_output(path, identity) == payload
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        verify_output(path, identity)
