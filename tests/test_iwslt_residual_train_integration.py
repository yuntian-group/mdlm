"""Small CPU integrations of the real training/checkpoint loop."""
from collections import Counter
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from additive_chain.backbone import file_sha256
from additive_chain.heads import GlobalAdditiveHead
from additive_chain.iwslt_residual_assets import PROTOCOL, head_checkpoint
import additive_chain.iwslt_residual_train as trainer


class TinyTranslator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.vocab_embed = torch.nn.Module()
        self.backbone.vocab_embed.embedding = torch.nn.Parameter(
            torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10)


def _nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _nested_equal(a, b)
    else:
        assert first == second


@pytest.fixture
def harness(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    count_keys = torch.tensor([1, 7])
    pair_counts = torch.tensor([3., 2.], dtype=torch.float64)
    left, right = torch.zeros(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64)
    left.index_add_(0, count_keys // 5, pair_counts)
    right.index_add_(0, count_keys % 5, pair_counts)
    torch.save(dict(left_counts=left, right_counts=right, pair_keys=count_keys,
                    pair_counts=pair_counts), data / "counts.pt")
    (data / "manifest.json").write_text(json.dumps(
        dict(counts_sha256=file_sha256(data / "counts.pt"))))
    trained = tmp_path / "translator.pt"
    torch.save(dict(model=TinyTranslator().state_dict()), trained)
    train = [dict(id=f"train-{i}", source_ids=[i % 3], target_ids=[1] * (1 + i % 4))
             for i in range(9)]
    dev = [dict(id=f"dev-{i}", source_ids=[i % 3], target_ids=[1] * (1 + i))
           for i in range(3)]
    calls, models, inherited, accesses, events = [], [], [], [], []

    def load_model(*_args, **_kwargs):
        model = TinyTranslator()
        models.append(model)
        return model, None

    def training_rows(_data, split):
        accesses.append(split)
        assert split == "train"
        return train

    def development_rows(_data, split):
        accesses.append(split)
        assert split == "dev"
        return dev

    def collate(rows, device):
        assert device == "cpu"
        return SimpleNamespace(rows=rows)

    def objective(batch, **kwargs):
        model, head, generator = kwargs["frozen"], kwargs["head"], kwargs["generator"]
        assert not model.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert kwargs["temperature"] == PROTOCOL["temperature"] == .85
        assert kwargs["nucleus_p"] == PROTOCOL["nucleus_p"] == .9
        assert kwargs["top_k"] == PROTOCOL["top_k"] == 500
        assert kwargs.get("schedule_steps", (2, 4, 8)) == (2, 4, 8)
        global_head = kwargs["global_head"]
        assert kwargs["global_strength"] == (1. if global_head is not None else 0.)
        if global_head is not None:
            assert not global_head.training
            assert all(not p.requires_grad and p.grad is None for p in global_head.parameters())
            if not any(global_head is old for old in inherited):
                inherited.append(global_head)
        selected = sum(len(row["target_ids"]) + 1 for row in batch.rows)
        draw = torch.rand((), generator=generator)
        parameter = head.right_embedding.weight if isinstance(head, GlobalAdditiveHead) \
            else head.right_token.weight
        total = (parameter.mean() - 1. - draw * .01).square() * selected
        calls.append(dict(training=head.training, ids=[row["id"] for row in batch.rows],
                          draw=draw.item(), inherited=global_head is not None))
        return SimpleNamespace(loss=total / selected, total_nll=total,
                               selected_tokens=selected,
                               metrics={"original_out_of_support_tokens": 0})

    monkeypatch.setattr(trainer, "registered_rows", training_rows)
    monkeypatch.setattr(trainer, "ordered_rows", development_rows)
    monkeypatch.setattr(trainer, "stratified_rows", lambda rows, n: rows[:n])
    monkeypatch.setattr(trainer, "load_conditional", load_model)
    monkeypatch.setattr(trainer, "collate_pairs", collate)
    monkeypatch.setattr(trainer, "residual_objective", objective)
    monkeypatch.setattr(trainer.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(trainer, "log", lambda event, **fields: events.append((event, fields)))

    def args(name, **changes):
        config = trainer.TrainingConfig(rank=2, context_size=3, microbatch=2,
            accumulate=2, max_epochs=2, patience=20, dev_examples=3, checkpoint_every=2)
        values = dict(asdict(config), checkpoint=tmp_path / "release",
            tokenizer=tmp_path / "tokenizer", trained=trained, data=data,
            output=tmp_path / name, global_head=None, device="cpu", max_new_updates=0)
        values.update(changes)
        return SimpleNamespace(**values)

    def inherited_checkpoint():
        identity = dict(PROTOCOL, model_sha256=file_sha256(trained),
            data_sha256=file_sha256(data / "manifest.json"),
            counts_sha256=file_sha256(data / "counts.pt"), inherited_global_sha256=None)
        head = GlobalAdditiveHead(5, 2)
        with torch.no_grad():
            head.right_embedding.weight.fill_(.125)
        path = tmp_path / "global-head.pt"
        torch.save(head_checkpoint(head, "global", identity, checkpoint_id="selected",
            epoch=1, update=9, metrics={"nll_per_selected_token": 1.}), path)
        return path

    return SimpleNamespace(args=args, calls=calls, models=models, inherited=inherited,
        accesses=accesses, events=events, train=train, dev=dev, inherited_checkpoint=inherited_checkpoint)


def _load(output):
    return torch.load(output / "latest.pt", weights_only=True)


def _training_calls(calls):
    return [item for item in calls if item["training"]]


def _verify_completed(output, expected_epochs=2):
    saved = _load(output)
    assert saved["state"]["complete"] is True
    assert saved["state"]["epoch"] == expected_epochs + 1
    assert saved["state"]["seen_epoch"] == 0
    assert saved["state"]["next_group"] == 0
    candidates = json.loads((output / "candidates.json").read_text())
    assert candidates["complete"] is True
    names = [item["name"] for item in candidates["candidates"]]
    assert len(names) == len(set(names))
    assert "epoch-0000" in names
    assert all(f"epoch-{epoch:04}" in names for epoch in range(1, expected_epochs + 1))
    assert any(name.startswith("best-loss-") for name in names)
    assert any(entry["improved"] and entry["update"] > 0 for entry in saved["state"]["history"])
    for candidate in candidates["candidates"]:
        from pathlib import Path
        path = Path(candidate["head"])
        assert path.exists() and file_sha256(path) == candidate["sha256"]
        packet = torch.load(path, weights_only=True)
        assert packet["checkpoint_id"] == candidate["name"]
        assert packet["identity"]["test_access"] is False
        assert packet["identity"]["schedules"] == [2, 4, 8]
        assert all(packet["identity"][key] == value for key, value in PROTOCOL.items())
    assert json.loads((output / "summary.json").read_text())["complete"] is True
    return saved


def test_real_run_resume_matches_uninterrupted_optimizer_rng_and_full_coverage(harness):
    full_args = harness.args("full")
    trainer.run(full_args)
    full_calls = _training_calls(harness.calls)
    full = _verify_completed(full_args.output)
    assert Counter(row for call in full_calls for row in call["ids"]) == Counter(
        {row["id"]: 2 for row in harness.train})
    config = trainer.TrainingConfig(**{key: getattr(full_args, key)
        for key in trainer.TrainingConfig.__dataclass_fields__})
    expected_ids = [harness.train[index]["id"] for epoch in (1, 2)
        for group in trainer.epoch_groups(harness.train, epoch, config)
        for batch in group for index in batch]
    assert [row for call in full_calls for row in call["ids"]] == expected_ids

    harness.calls.clear()
    partial_args = harness.args("resumed", max_new_updates=1)
    trainer.run(partial_args)
    partial = _load(partial_args.output)
    assert partial["state"]["update"] == partial["state"]["next_group"] == 1
    assert partial["state"]["epoch"] == 1 and not partial["state"]["complete"]
    assert partial["optimizer"]["state"]
    first_calls = _training_calls(harness.calls)
    assert partial["state"]["seen_epoch"] == sum(len(call["ids"]) for call in first_calls)
    assert not (partial_args.output / "candidates.json").exists()
    partial_args.max_new_updates = 0
    trainer.run(partial_args)
    resumed = _verify_completed(partial_args.output)
    assert _training_calls(harness.calls) == full_calls
    _nested_equal(full["head_state"], resumed["head_state"])
    _nested_equal(full["optimizer"], resumed["optimizer"])
    _nested_equal(full["generator"], resumed["generator"])
    _nested_equal(full["state"]["history"], resumed["state"]["history"])
    assert harness.accesses == ["train", "dev"] * 3
    assert all(not model.training and all(p.grad is None and not p.requires_grad
               for p in model.parameters()) for model in harness.models)


def test_contextual_stage_inherits_frozen_global_and_only_trains_new_head(harness):
    path = harness.inherited_checkpoint()
    original_hash = file_sha256(path)
    args = harness.args("contextual", kind="contextual", global_head=path, max_epochs=1)
    trainer.run(args)
    final = _verify_completed(args.output, expected_epochs=1)
    assert final["identity"]["inherited_global_sha256"] == original_hash
    assert file_sha256(path) == original_hash
    assert len(harness.inherited) == 1
    assert all(not p.requires_grad and p.grad is None for p in harness.inherited[0].parameters())
    assert final["head_state"]["right_token.weight"].abs().sum() > 0
    assert all(call["inherited"] for call in harness.calls)
    assert Counter(row for call in _training_calls(harness.calls) for row in call["ids"]) == Counter(
        row["id"] for row in harness.train)
    assert harness.accesses == ["train", "dev"]


def test_completed_run_is_idempotent_and_does_not_reload_model(harness):
    args = harness.args("complete")
    trainer.run(args)
    count = len(harness.models)
    recorded = len(harness.calls)
    before = file_sha256(args.output / "latest.pt")
    trainer.run(args)
    assert len(harness.models) == count and len(harness.calls) == recorded
    assert file_sha256(args.output / "latest.pt") == before
    assert harness.events[-1][0] == "residual_training_already_complete"


def _same_training_result(first, second):
    for key in ("head_state", "optimizer", "generator"):
        _nested_equal(first[key], second[key])
    for key in ("history", "update", "epoch", "seen_epoch", "next_group", "bad_checks",
                "initialized", "complete", "termination"):
        _nested_equal(first["state"][key], second["state"][key])


@pytest.mark.parametrize("snapshot_name", ["epoch-0000.pt", "epoch-0001.pt"])
def test_crash_after_immutable_epoch_snapshot_replays_deterministically(
        harness, monkeypatch, snapshot_name):
    baseline = harness.args("reference")
    trainer.run(baseline)
    expected = _verify_completed(baseline.output)
    args = harness.args("interrupted")
    original_save = trainer.atomic_save
    crashed = []

    def save_then_crash(value, path):
        original_save(value, path)
        if path.name == snapshot_name and not crashed:
            crashed.append(True)
            raise RuntimeError("Injected interruption after immutable snapshot")

    monkeypatch.setattr(trainer, "atomic_save", save_then_crash)
    with pytest.raises(RuntimeError, match="Injected interruption"):
        trainer.run(args)
    assert crashed and (args.output / snapshot_name).exists()
    snapshot_hash = file_sha256(args.output / snapshot_name)
    interrupted = _load(args.output)
    if snapshot_name == "epoch-0000.pt":
        assert not interrupted["state"]["initialized"]
        assert interrupted["state"]["history"] == []
        assert interrupted["state"]["update"] == 0
    else:
        assert interrupted["state"]["initialized"]
        assert interrupted["state"]["epoch"] == 1
        assert "epoch-0001" not in [c["name"] for c in interrupted["state"]["candidates"]]
    monkeypatch.setattr(trainer, "atomic_save", original_save)
    trainer.run(args)
    recovered = _verify_completed(args.output)
    _same_training_result(expected, recovered)
    assert file_sha256(args.output / snapshot_name) == snapshot_hash


def test_initial_validation_interruption_resumes_from_saved_initialization(harness, monkeypatch):
    baseline = harness.args("reference")
    trainer.run(baseline)
    expected = _verify_completed(baseline.output)
    args = harness.args("interrupted")
    original_objective = trainer.residual_objective
    interrupted_draw = []

    def loss_then_crash(batch, **kwargs):
        value = original_objective(batch, **kwargs)
        if not kwargs["head"].training and not interrupted_draw:
            interrupted_draw.append(True)
            raise RuntimeError("Injected interruption inside initial development loss")
        return value

    monkeypatch.setattr(trainer, "residual_objective", loss_then_crash)
    with pytest.raises(RuntimeError, match="initial development"):
        trainer.run(args)
    saved = _load(args.output)
    assert saved["state"]["initialized"] is False
    assert saved["state"]["update"] == 0 and saved["state"]["history"] == []
    assert saved["optimizer"]["state"] == {}
    assert not (args.output / "epoch-0000.pt").exists()
    monkeypatch.setattr(trainer, "residual_objective", original_objective)
    trainer.run(args)
    _same_training_result(expected, _verify_completed(args.output))


@pytest.mark.parametrize("missing_receipt", ["candidates.json", "summary.json"])
def test_complete_checkpoint_reconstructs_missing_receipts_without_training(
        harness, monkeypatch, missing_receipt):
    args = harness.args("completion-interrupted")
    original_json = trainer.atomic_json
    interrupted = []

    def fail_completion(path, value):
        if path.name == missing_receipt and not interrupted:
            interrupted.append(True)
            raise RuntimeError("Injected interruption before completion receipt")
        return original_json(path, value)

    monkeypatch.setattr(trainer, "atomic_json", fail_completion)
    with pytest.raises(RuntimeError, match="completion receipt"):
        trainer.run(args)
    saved = _load(args.output)
    assert saved["state"]["complete"] is True
    assert not (args.output / missing_receipt).exists()
    if missing_receipt == "candidates.json":
        assert not (args.output / "summary.json").exists()
    checkpoint_hash = file_sha256(args.output / "latest.pt")
    calls, models = len(harness.calls), len(harness.models)
    monkeypatch.setattr(trainer, "atomic_json", original_json)
    trainer.run(args)
    _same_training_result(saved, _verify_completed(args.output))
    assert file_sha256(args.output / "latest.pt") == checkpoint_hash
    assert len(harness.calls) == calls and len(harness.models) == models
    assert harness.events[-1][0] == "residual_training_already_complete"


def test_epoch_end_plateau_survives_crash_before_progress_receipt(harness, monkeypatch):
    # Improvement after the first update, then a flat development objective,
    # triggers patience after the first epoch.
    # The saved stopping decision must survive a crash before finalization.
    def plateau_loss(_rows, _model, _potential, head, *_args, **_kwargs):
        value = 2. if bool(head.right_embedding.weight.eq(0).all()) else 1.
        return dict(examples=3, selected_tokens=9, total_nll=value * 9,
                    nll_per_selected_token=value, original_out_of_support_fraction=0.)

    monkeypatch.setattr(trainer, "evaluate_block_loss", plateau_loss)
    reference = harness.args("plateau-reference", patience=2)
    trainer.run(reference)
    expected = _verify_completed(reference.output, expected_epochs=1)
    assert expected["state"]["termination"] == "development_block_loss_plateau"
    harness.calls.clear()
    args = harness.args("plateau-interrupted", patience=2)
    original_json = trainer.atomic_json
    interrupted = []

    def interrupt_progress(path, value):
        if path.name == "candidates-progress.json" and not interrupted:
            interrupted.append(True)
            raise RuntimeError("Injected interruption after plateau persisted")
        return original_json(path, value)

    monkeypatch.setattr(trainer, "atomic_json", interrupt_progress)
    with pytest.raises(RuntimeError, match="plateau persisted"):
        trainer.run(args)
    saved = _load(args.output)
    assert saved["state"]["stop_training"] is True
    assert saved["state"]["epoch"] == 2 and saved["state"]["complete"] is False
    training_calls = len(harness.calls)
    monkeypatch.setattr(trainer, "atomic_json", original_json)
    trainer.run(args)
    recovered = _verify_completed(args.output, expected_epochs=1)
    _same_training_result(expected, recovered)
    assert recovered["state"]["stop_training"] is True
    assert len(harness.calls) == training_calls
