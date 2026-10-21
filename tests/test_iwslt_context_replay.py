from copy import deepcopy
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from additive_chain import iwslt_context_replay as replay
from additive_chain import iwslt_residual_eval as evaluation
from additive_chain.adaptation import count_pair_scores
from additive_chain.backbone import file_sha256
from additive_chain.composition import compose_pair_scores
from additive_chain.counts import SparseCountPotential
from additive_chain.fast_candidates import extract_compact_candidates
from additive_chain.generation import ChainDecoder, sample_window, stream_seed
from additive_chain.heads import GlobalAdditiveHead, ContextualAdditiveHead
from additive_chain.iwslt_confirm import atomic_json, read
from additive_chain.iwslt_followup import output_name
from additive_chain.iwslt_residual_assets import dense_count_scores


def brute_scores(unary, pair, slots, selected):
    paths = list(itertools.product(range(unary.shape[-1]), repeat=unary.shape[1]))
    weights = torch.stack([sum(unary[0, i, v] for i, v in enumerate(path))
                          + sum(pair[0, i, a, b] for i, (a, b) in enumerate(zip(path, path[1:])))
                          for path in paths]).double()
    logz = torch.logsumexp(weights, 0)
    positions = selected[0].nonzero().flatten().tolist()
    individual = []
    for pos in positions:
        if slots[0, pos] < 0:
            individual.append(None)
        else:
            chosen = [i for i, path in enumerate(paths) if path[pos] == int(slots[0, pos])]
            individual.append(float(logz - torch.logsumexp(weights[chosen], 0)))
    joint = None
    if all(v is not None for v in individual):
        chosen = [i for i, path in enumerate(paths) if all(path[p] == int(slots[0, p]) for p in positions)]
        joint = float(logz - torch.logsumexp(weights[chosen], 0))
    return joint, individual


def toy_heads():
    torch.manual_seed(91)
    global_head = GlobalAdditiveHead(5, 2)
    context = ContextualAdditiveHead(torch.randn(5, 3), rank=2, context_size=4)
    zero = ContextualAdditiveHead(context.token_embeddings, rank=2, context_size=4)
    with torch.no_grad():
        global_head.right_embedding.weight.normal_(std=.2)
        context.right_token.weight.normal_(std=.2)
        context.right_shift.weight.normal_(std=.1)
    for head in (global_head, context, zero):
        head.eval().requires_grad_(False)
    counts = dict(left_counts=torch.tensor([2., 3., 0., 0., 0.]), right_counts=torch.tensor([0., 2., 0., 3., 0.]),
                  pair_keys=torch.tensor([1, 8]), pair_counts=torch.tensor([2., 3.]))
    potential = SparseCountPotential.from_counts(*counts.values(), strength=.2, smoothing=.1)
    return global_head, context, zero, counts, potential


@pytest.mark.parametrize("selected", [[[True, False, True]], [[False, False, True]]])
def test_log_gold_scores_match_exhaustive_with_hidden_intermediate_and_visible_clamp(selected):
    torch.manual_seed(7)
    unary = torch.randn(1, 3, 3, dtype=torch.float64)
    unary[0, 1, 1:] = -torch.inf
    pair = torch.randn(1, 2, 3, 3, dtype=torch.float64)
    selected = torch.tensor(selected)
    slots = torch.tensor([[2, 0, 1]])
    expected_joint, expected_single = brute_scores(unary, pair, slots, selected)
    actual = replay.gold_scores(unary, pair, slots, selected)
    assert actual["joint_gold_nll"] == pytest.approx(expected_joint, abs=1e-12)
    assert actual["singleton_gold_nll"] == pytest.approx(expected_single, abs=1e-12)
    if int(selected.sum()) == 1:
        assert actual["joint_gold_nll"] == pytest.approx(actual["sum_singleton_gold_nll"], abs=1e-12)


def test_gold_scores_keep_unsupported_nll_infinite_and_do_not_epsilon_clamp():
    unary = torch.tensor([[[0., -1000.], [0., -2.]]])
    pair = torch.zeros(1, 1, 2, 2)
    value = replay.gold_scores(unary, pair, torch.tensor([[1, -1]]), torch.ones(1, 2, dtype=torch.bool))
    assert value["joint_gold_nll"] is None
    assert value["joint_nll_status"] == "infinite_gold_out_of_support"
    assert value["singleton_gold_nll"] == [1000., None]
    assert value["sum_singleton_gold_nll"] is None
    assert value["supported_singleton_nll_sum"] == 1000.
    with pytest.raises(ValueError, match="nonempty"):
        replay.gold_scores(unary, pair, torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.bool))


def test_state_composition_matches_enumeration_zero_nesting_and_dense_counts():
    global_head, context, zero, counts, potential = toy_heads()
    output = SimpleNamespace(log_probs=torch.tensor([[[.25, .25, .25, .25, 0.]] * 3]).log(),
                             hidden=torch.randn(1, 3, 3))
    # Visible token 2 is deliberately different from aligned reference 0.
    current, gold = torch.tensor([[2, 4, 4]]), torch.tensor([[0, 1, 2]])
    selected = torch.tensor([[False, True, True]])
    sparse = replay.score_state(output, current, selected, gold, potential,
                                global_head, context, zero_head=zero, mask_id=4)
    dense = replay.score_state(output, current, selected, gold, potential, global_head,
                               context, zero_head=zero, count_table=dense_count_scores(potential), mask_id=4)
    assert sparse == dense
    assert sparse["zero_context_nesting_checked"] is True
    assert sparse["visible_reference_mismatches"] == 1
    assert sparse["tokens"][0] == 2
    candidates = extract_compact_candidates(output.log_probs, current, mask_id=4, nucleus_p=.9, top_k=500)
    slots = candidates.candidate_ids.eq(gold[..., None]).long().argmax(-1)
    assert bool(candidates.candidate_ids.eq(gold[..., None]).any(-1)[selected].all())
    count = count_pair_scores(candidates, potential)
    for name, contextual in (("global", None), ("contextual", context)):
        pair = compose_pair_scores(count, global_factors=global_head(candidates.candidate_ids), global_strength=1.,
            contextual_factors=None if contextual is None else contextual(candidates.candidate_ids, output.hidden),
            contextual_strength=0. if contextual is None else 1.) / .85
        expected_joint, expected_single = brute_scores(candidates.log_unary.float() / .85, pair, slots, selected)
        assert sparse["models"][name]["joint_gold_nll"] == pytest.approx(expected_joint, abs=2e-6)
        assert sparse["models"][name]["singleton_gold_nll"] == pytest.approx(expected_single, abs=2e-6)
    zero_result = replay.score_state(output, current, selected, gold, potential, global_head, zero, mask_id=4)
    assert zero_result["models"]["global"] == zero_result["models"]["contextual"]


def test_state_energy_matches_decoder_temperature_convention():
    global_head, context, _, counts, potential = toy_heads()
    current, gold = torch.tensor([[4, 1, 4]]), torch.tensor([[2, 1, 3]])
    selected = current.eq(4)
    output = SimpleNamespace(log_probs=torch.tensor([[[.3, .2, .25, .25, 0.]] * 3]).log(),
                             hidden=torch.randn(1, 3, 3))
    result = replay.score_state(output, current, selected, gold, potential, global_head, context, mask_id=4)
    decoder = ChainDecoder(mask_id=4, device="cpu", count_statistics=counts,
        count_strength=.2, top_k=500, nucleus_p=.9, fast_candidates=True,
        temperature=.85, global_head=global_head, global_strength=1.,
        contextual_head=context, contextual_strength=1.)
    candidates = decoder._candidates(output.log_probs, current)
    pair = compose_pair_scores(count_pair_scores(candidates, decoder.potential),
        global_factors=global_head(candidates.candidate_ids), global_strength=decoder.global_strength,
        contextual_factors=context(candidates.candidate_ids, output.hidden), contextual_strength=decoder.contextual_strength)
    slots = candidates.candidate_ids.eq(gold[..., None]).long().argmax(-1)
    assert bool(candidates.candidate_ids.eq(gold[..., None]).any(-1)[selected].all())
    expected = replay.gold_scores(candidates.log_unary.float(), pair, slots, selected)
    assert result["models"]["contextual"]["joint_gold_nll"] == pytest.approx(expected["joint_gold_nll"], abs=3e-6)
    assert result["models"]["contextual"]["singleton_gold_nll"] == pytest.approx(expected["singleton_gold_nll"], abs=3e-6)


def test_state_never_injects_unsupported_gold():
    global_head, context, zero, _, potential = toy_heads()
    output = SimpleNamespace(log_probs=torch.tensor([[[.99, .003, .003, .004, 0.]] * 2]).log(),
                             hidden=torch.randn(1, 2, 3))
    value = replay.score_state(output, torch.tensor([[4, 4]]), torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0, 3]]), potential, global_head, context, zero_head=zero, mask_id=4)
    assert value["candidate_counts"] == [1, 1]
    assert value["supported_positions"] == [0]
    assert value["gold_injected_tokens"] == 0
    for model in value["models"].values():
        assert model["joint_gold_nll"] is None
        assert model["singleton_gold_nll"] == [0., None]


class TinyModel:
    def __init__(self):
        self.calls = []

    def __call__(self, tokens, valid):
        self.calls.append(tokens.clone())
        assert bool(valid.all())
        raw = torch.full((*tokens.shape, 5), .25)
        raw[..., 4] = 0.
        hidden = tokens.float()[..., None].expand(*tokens.shape, 3).clone()
        return SimpleNamespace(log_probs=raw.log(), hidden=hidden)


class TinyDecoder:
    device = torch.device("cpu")
    mask_id = 4

    def reveal(self, output, tokens, selected, generator):
        result = tokens.clone()
        result[selected] = torch.randint(0, 4, (int(selected.sum()),), generator=generator)
        return result


def test_replay_preserves_global_rng_and_only_intervenes_on_visible_target_ids(monkeypatch):
    global_head, context, zero, _, potential = toy_heads()
    row = dict(id="dev-x", source_ids=[1, 2], target_ids=[0, 1, 2, 1, 0, 2, 1])
    model, decoder = TinyModel(), TinyDecoder()
    actual = replay.replay_example(row, 37, model, decoder, potential, global_head, context,
                                   zero_head=zero, mask_id=4, eos_id=3)
    # Run the exact same scheduler without any scoring observer.
    streams = {name: torch.Generator().manual_seed(stream_seed("iwslt-101", 37, name))
               for name in ("reveal", "proposal")}
    reference = sample_window(torch.full((1, 8), 4), lambda tokens, t: TinyModel()(tokens, tokens.ne(-1)),
        decoder, steps=2, reveal_generator=streams["reveal"], proposal_generator=streams["proposal"])
    assert actual["rollout_tokens"] == reference.tokens[0].tolist()
    assert actual["schedule"] == reference.stats["schedule"]
    assert set(actual["states"]) == set(replay.STATE_NAMES)
    first, generated, gold = (actual["states"][name] for name in replay.STATE_NAMES)
    assert first["tokens"] == [4] * 8
    assert generated["selected_positions"] == gold["selected_positions"]
    assert [i for i, x in enumerate(generated["tokens"]) if x == 4] == gold["selected_positions"]
    for pos, value in enumerate(gold["tokens"]):
        assert value == (4 if generated["tokens"][pos] == 4 else actual["reference_tokens"][pos])
    assert all(call[0, :4].tolist() == [3, 1, 2, 3] for call in model.calls)
    assert len(model.calls) == actual["actual_forward_calls"] + actual["counterfactual_forward_calls"]
    monkeypatch.setattr(replay, "MASK_ID", 4)
    monkeypatch.setattr(replay, "EOS_ID", 3)
    replay.validate_record(actual, row, 37)


def test_empty_reveal_is_not_a_zero_nll_observation(monkeypatch):
    global_head, context, zero, _, potential = toy_heads()
    row = dict(id="dev-short", source_ids=[1], target_ids=[])
    first_empty = next(i for i in range(100) if torch.rand((1, 1), dtype=torch.float64,
        generator=torch.Generator().manual_seed(stream_seed("iwslt-101", i, "reveal"))).item() >= .5)
    result = replay.replay_example(row, first_empty, TinyModel(), TinyDecoder(), potential,
                                  global_head, context, zero_head=zero, mask_id=4, eos_id=3)
    assert result["reveal_counts"] == [0, 1]
    assert "first_all_masked" not in result["states"]
    a, b = result["states"]["later_generated_visible"], result["states"]["later_gold_visible"]
    assert a["models"] == b["models"]
    assert a["candidate_sha256"] == b["candidate_sha256"]
    assert result["counterfactual_forward_calls"] == 0
    summary = replay.aggregate([result])
    assert summary["states"]["first_all_masked"]["blocks"] == 0
    assert summary["states"]["first_all_masked"]["models"]["global"]["covered_block_joint_nll_per_token"] is None


def candidate(path, epoch):
    metadata = dict(checkpoint_id=f"epoch-{epoch:04d}", epoch=epoch, update=epoch * 10,
                    identity=dict(source_sha256={"heads.py": "training-source"}))
    return evaluation.candidate_entry(path, metadata), metadata


def selection(kind, common, candidates, scores):
    evaluated = {c["name"]: [dict(cell=cell, bleu=score, chrf=score) for cell in evaluation.cells("selection")]
                 for c, score in zip(candidates, scores)}
    winner, ranking = evaluation.choose_checkpoint(candidates, evaluated)
    return dict(format=evaluation.FORMAT, split="dev", kind=kind, examples=2, common=common,
        manifest=dict(kind=kind, candidates=candidates, complete=True), evaluated=evaluated,
        selected=winner, ranking=ranking)


@pytest.fixture
def assets(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "EXAMPLES", 2)
    monkeypatch.setattr(replay, "FULL_EXAMPLES", 3)
    monkeypatch.setattr(replay, "SELECTION_EXAMPLES", 2)
    rows = [dict(id=f"dev-{i}", source_ids=[1, 2], target_ids=[1], target="a") for i in range(3)]
    metadata, candidates = {}, {}
    for name, epoch in (("global-zero", 0), ("global-five", 5), ("context-zero", 0), ("context-one", 1)):
        path = tmp_path / (name + ".pt")
        path.write_bytes(name.encode())
        c, m = candidate(path, epoch)
        metadata[str(path)], candidates[name] = m, c
    common = dict(model_sha256="model", data_sha256="data", counts_sha256="counts",
                  inherited_global_sha256=candidates["global-five"]["sha256"])
    global_common = dict(common, inherited_global_sha256=None)
    global_doc = selection("global", global_common, [candidates["global-zero"], candidates["global-five"]], [10., 15.])
    context_doc = selection("contextual", common, [candidates["context-zero"], candidates["context-one"]], [15., 14.])
    args = SimpleNamespace(trained=tmp_path / "trained.pt", data=tmp_path / "data",
        length_head=tmp_path / "length.pt", selection=tmp_path / "count.json",
        global_head=Path(candidates["global-five"]["head"]), global_selection=tmp_path / "global.json",
        contextual_selection=tmp_path / "context.json", official_evaluation_root=tmp_path / "evaluation",
        output=tmp_path / "replay", command="run", max_new_examples=0,
        checkpoint=tmp_path / "pretrained", tokenizer=tmp_path / "tokenizer")
    atomic_json(args.global_selection, global_doc)
    called = []
    monkeypatch.setattr(evaluation, "common_assets", lambda args: common)
    monkeypatch.setattr(evaluation, "source_hashes", lambda: {"heads.py": "frozen-source"})
    monkeypatch.setattr(evaluation, "development_rows", lambda data, pop: called.append(pop) or (rows[:2] if pop == "selection" else rows))
    monkeypatch.setattr(replay, "stratified_rows", lambda data, count: [data[2], data[0]])
    class Head:
        def cuda(self):
            return self
    head = Head()
    monkeypatch.setattr(evaluation, "load_heads", lambda path, kind, assets, global_path=None:
                        (head, head if global_path else None, metadata[str(path)]))
    cell = evaluation.cells("selection")[0]
    relative = f"epoch-0000/selection/{output_name(cell)}/result.json"
    reference = args.official_evaluation_root / relative
    reference.parent.mkdir(parents=True)
    atomic_json(reference, dict(summary=dict(examples=2), predictions=[dict(id=row["id"]) for row in rows[:2]]))
    atomic_json(reference.with_name("result-identity.json"), dict(output_sha256=file_sha256(reference),
        identity=evaluation.build_identity(cell, rows[:2], "selection", "contextual", candidates["context-zero"], common)))
    context_doc["outputs"] = {relative: file_sha256(reference)}
    atomic_json(args.contextual_selection, context_doc)
    return SimpleNamespace(args=args, common=common, rows=rows, called=called, candidates=candidates,
                           context_doc=context_doc, global_doc=global_doc, head=head)


def test_prepare_binds_frozen_source_parent_original_selections_and_full_stream_indices(assets):
    before = read(assets.args.contextual_selection)
    state = replay.prepare(assets.args)
    assert state.identity["ids"] == ["dev-2", "dev-0"]
    assert state.identity["stream_indices"] == [2, 0]
    assert state.identity["official_contextual"]["update"] == 0
    assert state.identity["contextual_diagnostic"]["epoch"] == 1
    assert state.identity["selected_global"]["epoch"] == 5
    assert assets.called == ["selection", "full"]
    assert read(assets.args.contextual_selection) == before


@pytest.mark.parametrize("damage", ["wrong-source", "wrong-parent", "nonzero-official", "incomplete", "test", "wrong-epoch", "source-receipt"])
def test_prepare_rejects_wrong_inputs(assets, monkeypatch, damage):
    if damage == "wrong-source":
        monkeypatch.setattr(evaluation, "source_hashes", lambda: {"heads.py": "different"})
    elif damage == "wrong-parent":
        assets.common["inherited_global_sha256"] = "changed"
    elif damage == "source-receipt":
        path = next(assets.args.official_evaluation_root.rglob("result.json"))
        path.write_text("changed")
    else:
        doc = read(assets.args.contextual_selection)
        if damage == "nonzero-official":
            doc = selection("contextual", assets.common,
                [assets.candidates["context-zero"], assets.candidates["context-one"]], [12., 15.])
        elif damage == "incomplete":
            doc["manifest"]["complete"] = False
        elif damage == "test":
            doc["split"] = "test"
        elif damage == "wrong-epoch":
            doc["manifest"]["candidates"][1]["epoch"] = 2
        atomic_json(assets.args.contextual_selection, doc)
    with pytest.raises(ValueError):
        replay.prepare(assets.args)


def fake_state(tokens, selected, supported, global_nll, context_nll, zero=True):
    values = {}
    for name, nll in zip(replay.MODEL_NAMES, (global_nll, context_nll)):
        individual = [nll if p in supported else None for p in selected]
        covered = len(supported) == len(selected)
        values[name] = dict(joint_gold_nll=nll * len(selected) if covered else None,
            joint_nll_status="finite" if covered else "infinite_gold_out_of_support",
            singleton_gold_nll=individual, sum_singleton_gold_nll=nll * len(selected) if covered else None,
            supported_singleton_nll_sum=nll * len(supported))
    visible = sum(token != replay.MASK_ID for token in tokens)
    return dict(tokens=tokens, selected_positions=selected, supported_positions=supported,
        selected_tokens=len(selected), supported_selected_tokens=len(supported),
        all_selected_supported=len(selected) == len(supported), candidate_counts=[1] * len(tokens),
        candidate_sha256="f" * 64, support_shared_by_heads=True, gold_injected_tokens=0,
        zero_context_nesting_checked=True if zero else None, models=values,
        visible_tokens=visible, visible_reference_mismatches=0,
        visible_reference_mismatch_fraction=0. if visible else None, early_visible_eos=False)


def fake_record(row, position):
    gold = row["target_ids"] + [replay.EOS_ID]
    return dict(id=row["id"], stream_index=position, reference_tokens=gold,
        source_sha256=replay.digest(row["source_ids"]), reference_length=2, rollout_tokens=gold,
        scheduled_steps=2, executed_steps=2, actual_forward_calls=2, reveal_counts=[1, 1],
        counterfactual_forward_calls=0,
        states=dict(first_all_masked=fake_state([replay.MASK_ID] * 2, [0], [0], 2., 1.),
            later_generated_visible=fake_state([gold[0], replay.MASK_ID], [1], [1], 3., 4.),
            later_gold_visible=fake_state([gold[0], replay.MASK_ID], [1], [1], 2., 1., zero=False)))


def test_aggregate_uses_token_weights_and_matched_population_difference_in_differences():
    a = fake_record(dict(id="a", source_ids=[0], target_ids=[1]), 0)
    b = deepcopy(a)
    b["id"] = "b"
    # This block is supported only in the generated state, so it must be absent
    # from both sides of the context-paired contrast.
    b["states"]["later_gold_visible"] = fake_state([1, replay.MASK_ID], [1], [], 90., 100., zero=False)
    summary = replay.aggregate([a, b])
    paired = summary["later_common_supported"]
    assert paired["retained_ids"] == ["a"]
    assert paired["eligible_selected_tokens"] == 2 and paired["retained_selected_tokens"] == 1
    assert paired["contextual_head_contrast_generated_minus_gold"]["covered_block_joint_nll_per_token"] == 2.
    assert paired["contextual_head_contrast_generated_minus_gold"]["covered_block_singleton_nll_per_token"] == 2.
    assert summary["states"]["later_gold_visible"]["complete_block_coverage"] == .5
    # Weighted by selected tokens, not by block means.
    states = [fake_state([4], [0], [0], 2., 1.), fake_state([4] * 3, [0, 1, 2], [0, 1, 2], 6., 2.)]
    assert replay.summarize_states(states)["models"]["global"]["covered_block_joint_nll_per_token"] == 5.


@pytest.mark.parametrize("damage", ["source", "reference", "visible", "reveal", "coverage", "finite-missing", "singleton", "zero", "mismatch"])
def test_record_validation_rejects_changed_state_or_metrics(damage):
    row = dict(id="a", source_ids=[0], target_ids=[1])
    record = fake_record(row, 10)
    replay.validate_record(record, row, 10)
    later = record["states"]["later_generated_visible"]
    if damage == "source":
        record["source_sha256"] = "changed"
    elif damage == "reference":
        record["reference_tokens"][0] = 2
    elif damage == "visible":
        later["tokens"][0] = 2
    elif damage == "reveal":
        later["selected_positions"] = [0]
    elif damage == "coverage":
        later["supported_selected_tokens"] = 0
    elif damage == "finite-missing":
        later["models"]["global"]["joint_nll_status"] = "infinite_gold_out_of_support"
    elif damage == "singleton":
        later["models"]["global"]["singleton_gold_nll"] = [5.]
    elif damage == "zero":
        later["zero_context_nesting_checked"] = None
    else:
        later["visible_reference_mismatches"] = 1
    with pytest.raises(ValueError):
        replay.validate_record(record, row, 10)


def test_run_partial_resume_and_complete_readout_use_same_records(assets, monkeypatch):
    import additive_chain.counts as counts_module
    import additive_chain.generation as generation
    import additive_chain.iwslt_residual_assets as residual_assets
    import additive_chain.translation as translation
    state = replay.prepare(assets.args)
    monkeypatch.setattr(replay, "prepare", lambda args: state)
    monkeypatch.setattr(replay, "unchanged", lambda args, current: None)
    class Model:
        def load_state_dict(self, state):
            pass
        def eval(self):
            return self
        def requires_grad_(self, value):
            return self
    class DeviceValue:
        def cuda(self):
            return self
    count_values = {key: DeviceValue() for key in ("left_counts", "right_counts", "pair_keys", "pair_counts")}
    monkeypatch.setattr(translation, "load_conditional", lambda *args: (Model(), None))
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: {"model": {}} if path == assets.args.trained else count_values)
    monkeypatch.setattr(counts_module.SparseCountPotential, "from_counts", lambda *args, **kwargs: "potential")
    monkeypatch.setattr(residual_assets, "dense_count_scores", lambda potential: "table")
    class Decoder:
        def __init__(self, **kwargs):
            assert kwargs["sampling"] == "joint" and kwargs["temperature"] == .85
        def prepare(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(generation, "ChainDecoder", Decoder)
    visited = []
    def one(row, position, *args, **kwargs):
        visited.append((row["id"], position))
        return fake_record(row, position)
    monkeypatch.setattr(replay, "replay_example", one)
    assets.args.max_new_examples = 1
    replay.run(assets.args)
    first_path = replay.record_path(assets.args, 2)
    first_hash = file_sha256(first_path)
    assert read(assets.args.output / "readout.json")["completed"] == 1
    assets.args.command = "readout"
    with pytest.raises(ValueError, match="all fixed"):
        replay.run(assets.args)
    assets.args.command, assets.args.max_new_examples = "run", 0
    replay.run(assets.args)
    assert visited == [("dev-2", 2), ("dev-0", 0)]
    assert file_sha256(first_path) == first_hash
    assets.args.command = "readout"
    replay.run(assets.args)
    assert read(assets.args.output / "readout.json")["complete"] is True
    document = read(first_path)
    document["record"]["rollout_tokens"][0] = 2
    atomic_json(first_path, document)
    with pytest.raises(ValueError, match="hash changed"):
        replay.run(assets.args)


def test_empty_common_supported_population_is_not_zero_effect():
    record = fake_record(dict(id="x", source_ids=[0], target_ids=[1]), 0)
    record["states"]["later_gold_visible"] = fake_state([1, replay.MASK_ID], [1], [], 1., 1., zero=False)
    result = replay.aggregate([record])["later_common_supported"]
    assert result["retained_blocks"] == 0
    assert all(value is None for value in result["contextual_head_contrast_generated_minus_gold"].values())


def test_token_matched_contrast_uses_position_intersection_not_separate_covered_sets():
    record = dict(id="partial", reveal_counts=[1, 3], states={
        "later_generated_visible": fake_state([1, 4, 4, 4], [1, 2, 3], [1, 2], 10., 13.),
        "later_gold_visible": fake_state([1, 4, 4, 4], [1, 2, 3], [2, 3], 2., 1., zero=False)})
    result = replay.aggregate([record])
    assert result["later_common_supported"]["retained_blocks"] == 0
    tokens = result["later_common_supported_tokens"]
    assert tokens["eligible_selected_tokens"] == 3
    assert tokens["retained_tokens"] == 1
    assert tokens["paired_positions"] == [dict(id="partial", positions=[2])]
    assert tokens["singleton_nll_sums"]["later_generated_visible"]["global"] == 10.
    assert tokens["contextual_head_contrast_generated_minus_gold"] == 4.
