import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from additive_chain import iwslt_backbone_continue as continuation
from additive_chain import iwslt_length_refit as refit
from additive_chain.backbone import file_sha256


def test_plateau_keeps_actual_best_and_accumulates_small_improvements():
    state = continuation.initial_plateau()
    assert continuation.update_plateau(state, 10000, {"bleu": 20.})
    assert continuation.update_plateau(state, 15000, {"bleu": 20.04})
    assert state["bad_checks"] == 1 and state["anchor_bleu"] == 20.
    assert continuation.update_plateau(state, 20000, {"bleu": 20.08})
    assert state["bad_checks"] == 2
    assert continuation.update_plateau(state, 25000, {"bleu": 20.1})
    assert state["bad_checks"] == 0 and state["anchor_bleu"] == 20.1
    for step, bleu in ((30000, 20.12), (35000, 20.11), (40000, 20.13), (45000, 20.12)):
        continuation.update_plateau(state, step, {"bleu": bleu})
    assert state["plateau"] and state["best_bleu"] == 20.13 and state["best_step"] == 40000
    with pytest.raises(ValueError, match="exactly once"):
        continuation.update_plateau(state, 45000, {"bleu": 20.12})
    with pytest.raises(ValueError, match="Invalid"):
        continuation.update_plateau(state, 50000, {"bleu": float("nan")})


def test_boundary_is_pending_until_validation_is_committed():
    state = continuation.initial_plateau()
    assert continuation.validation_due(10000, state)
    continuation.update_plateau(state, 10000, {"bleu": 20.})
    assert not continuation.validation_due(10000, state)
    assert not continuation.validation_due(14999, state)
    assert continuation.validation_due(15000, state)
    assert not state["plateau"]


def test_validation_rng_restored_even_when_interrupted():
    torch.manual_seed(17)
    before = torch.get_rng_state().clone()
    with pytest.raises(refit.Interrupted):
        with continuation.validation_rng("cpu"):
            torch.randn(40)
            torch.manual_seed(1)
            raise refit.Interrupted("stop")
    assert torch.equal(torch.get_rng_state(), before)


def test_prediction_resume_accepts_only_matching_committed_prefix(tmp_path):
    rows = [dict(id="a"), dict(id="b")]
    records = [dict(id="a", reference="ref a", hypothesis="hyp a", token_ids=[1], seconds=2., predicted_length=2)]
    path = tmp_path / "predictions.jsonl"
    line = json.dumps(records[0]) + "\n"
    path.write_text(line + '{"unfinished":')
    assert continuation._read_progress(path, rows, ["ref a", "ref b"]) == records
    assert path.read_text() == line
    with pytest.raises(ValueError, match="prefix"):
        continuation._read_progress(path, rows, ["changed", "ref b"])
    path.write_text(line + line)
    with pytest.raises(ValueError, match="prefix"):
        continuation._read_progress(path, rows, ["ref a", "ref b"])


def test_decode_resume_preserves_original_example_indices(tmp_path, monkeypatch):
    class Decoder:
        def __init__(self, **kwargs):
            assert kwargs["temperature"] == .85 and kwargs["top_k"] == 500 and kwargs["nucleus_p"] == .9
        def prepare(self):
            pass
        def close(self):
            pass
    rows = [dict(id=str(i), target="this is the same sentence") for i in range(4)]
    calls = []
    def translate(model, row, head, decoder, *, steps, seed, example_id):
        assert steps == 32 and seed == 101 and row["id"] == str(example_id)
        calls.append(example_id)
        return [example_id], dict(predicted_length=5)
    monkeypatch.setattr(continuation, "ChainDecoder", Decoder)
    monkeypatch.setattr(continuation, "translate_one", translate)
    model = ToyModel()
    tok = SimpleNamespace(decode=lambda ids: "this is the same sentence")
    identity = dict(step=10000)
    with pytest.raises(refit.Interrupted):
        continuation.score_resumable(model, tok, model, rows, tmp_path, identity,
                                     "cpu", lambda: len(calls) == 2)
    result = continuation.score_resumable(model, tok, model, rows, tmp_path, identity, "cpu", lambda: False)
    assert calls == [0, 1, 2, 3] and result["examples"] == 4
    assert result["bleu"] == pytest.approx(100.)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.1, .2]))
        self.pretrained = SimpleNamespace(config=SimpleNamespace(hidden_dim=2))

    def loss_sum(self, batch, generator):
        noise = torch.rand(2, generator=generator) + torch.rand(2)
        target = batch.tokens[batch.target].float().mean() / 100000.
        return ((self.weight - target - noise) ** 2).sum(), int(batch.target.sum())


def pilot_state():
    torch.manual_seed(8)
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999), weight_decay=.01)
    model.weight.square().sum().backward()
    optimizer.step()
    return dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
        step=10000, corruption_rng=torch.Generator().manual_seed(111).get_state(),
        order_rng=random.Random(11).getstate(), cpu_rng=torch.get_rng_state().clone(), cuda_rng=None,
        best=16.84, spent_seconds=1.)


def test_restore_keeps_optimizer_and_each_cpu_random_stream():
    state = pilot_state()
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.5)
    corruption, order = torch.Generator(), random.Random()
    continuation.restore_training_state(state, model, optimizer, corruption, order, "cpu")
    assert torch.equal(model.weight, state["model"]["weight"])
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert torch.equal(optimizer.state[model.weight]["exp_avg"], state["optimizer"]["state"][0]["exp_avg"])
    assert torch.equal(corruption.get_state(), state["corruption_rng"])
    assert order.getstate() == state["order_rng"]
    assert torch.equal(torch.get_rng_state(), state["cpu_rng"])
    bad = copy.deepcopy(state)
    bad["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape"):
        continuation.restore_training_state(bad, model, optimizer, corruption, order, "cpu")
    bad = copy.deepcopy(state)
    del bad["corruption_rng"]
    with pytest.raises(ValueError, match="Incomplete"):
        continuation.validate_training_state(bad)
    bad = copy.deepcopy(state)
    bad["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        continuation.validate_training_state(bad)
    with pytest.raises(ValueError, match="Unexpected"):
        continuation.validate_training_state(dict(state, step=9000), initial=True)


def test_parent_identity_binds_data_code_and_training_semantics(tmp_path):
    root = tmp_path / "code/additive_chain"
    root.mkdir(parents=True)
    local = Path(continuation.__file__).parent
    names = ("translation.py", "iwslt_run.py", "generation.py")
    for name in names:
        (root / name).write_text((local / name).read_text())
    identity = dict(continuation.PILOT, smoke=False, dataset_sha256="data",
                    code_sha256={name: file_sha256(root / name) for name in names})
    continuation.validate_pilot_identity(identity, "data", root.parent)
    with pytest.raises(ValueError, match="dataset"):
        continuation.validate_pilot_identity(identity, "other", root.parent)
    with pytest.raises(ValueError, match="configuration"):
        continuation.validate_pilot_identity(dict(identity, lr=.001), "data", root.parent)
    (root / "translation.py").write_text((root / "translation.py").read_text().replace(".001 + .998", ".002 + .997"))
    identity["code_sha256"]["translation.py"] = file_sha256(root / "translation.py")
    with pytest.raises(ValueError, match="semantics"):
        continuation.validate_pilot_identity(identity, "data", root.parent)


def setup_run(tmp_path, monkeypatch):
    # All scheduling/checkpoint code runs; only the expensive model and evaluation
    # are replaced. Tiny rows still pass the real target collation/training loop.
    monkeypatch.setitem(continuation.VALIDATION, "validate_every", 2)
    identity = dict(format=continuation.FORMAT, pilot={"pilot": "fixed"}, dataset_sha256="data", source_sha256={})
    monkeypatch.setattr(continuation, "build_identity", lambda args: identity)
    rows = [dict(id=str(i), source_ids=[i + 1], target_ids=[i + 2]) for i in range(8)]
    calls = []
    def registered(data, split):
        assert split in ("train", "dev")
        calls.append(split)
        return rows
    monkeypatch.setattr(refit, "registered_rows", registered)
    monkeypatch.setattr(continuation, "ordered_rows", lambda data, split: list(reversed(rows)))
    monkeypatch.setattr(continuation, "load_conditional", lambda *a, **k: (ToyModel(), None))
    source = tmp_path / "source"
    source.mkdir()
    state = pilot_state()
    head = continuation.LengthPredictor(2)
    torch.save(state, source / "latest.pt")
    torch.save(dict(model=state["model"], step=10000, identity=identity["pilot"],
                    length_head=head.state_dict()), source / "best.pt")
    args = SimpleNamespace(output=tmp_path / "output", data=tmp_path / "data", checkpoint=tmp_path / "ckpt",
        tokenizer=tmp_path / "tokenizer", device="cpu", source_latest=source / "latest.pt",
        source_best=source / "best.pt", seconds_limit=3600, updates_per_chunk=6, save_every=1)
    evaluations = []
    def validate(model, tokenizer, train_rows, dev_rows, decode_rows, old_head, step, root, identity, device, stopped):
        assert decode_rows == list(reversed(dev_rows))
        assert model.weight.requires_grad
        evaluations.append(step)
        torch.rand(20)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "length.pt"
        torch.save(dict(format=refit.FORMAT, identity=dict(source_sha256=refit.source_hashes(),
            data_sha256="data", model_sha256="before-export"), hidden_size=2, length_head=old_head), path)
        return dict(bleu=20. + (step - 10000) * .01, examples=8), old_head, path
    monkeypatch.setattr(continuation, "run_validation", validate)
    return args, evaluations, calls, validate


def test_training_resume_matches_uninterrupted_and_chunk_is_not_plateau(tmp_path, monkeypatch):
    args, validations, calls, _ = setup_run(tmp_path, monkeypatch)
    args.output = tmp_path / "complete"
    complete = continuation.run(args)
    whole = torch.load(args.output / "latest.pt", weights_only=False)
    assert validations == [10000, 10002, 10004, 10006]
    assert not whole["plateau"]["plateau"]
    assert json.loads((args.output / "status.json").read_text())["stopping_reason"] == "chunk_limit"
    validations.clear()
    args.output = tmp_path / "resumed"
    args.updates_per_chunk = 3
    continuation.run(args)
    continuation.run(args)
    resumed = torch.load(args.output / "latest.pt", weights_only=False)
    assert validations == [10000, 10002, 10004, 10006]
    assert whole["plateau"] == resumed["plateau"]
    assert torch.equal(whole["model"]["weight"], resumed["model"]["weight"])
    assert torch.equal(whole["optimizer"]["state"][0]["exp_avg"], resumed["optimizer"]["state"][0]["exp_avg"])
    for key in ("corruption_rng", "cpu_rng"):
        assert torch.equal(whole[key], resumed[key])
    assert whole["order_rng"] == resumed["order_rng"]
    assert set(calls) == {"train", "dev"}
    head = refit.load_refit_head(args.output / "best-length.pt", file_sha256(args.output / "best.pt"), "data", "cpu")
    assert head is not None


def test_interrupted_validation_resumes_before_any_training(tmp_path, monkeypatch):
    args, validations, _, original = setup_run(tmp_path, monkeypatch)
    def interrupted(*a, **k):
        torch.rand(50)
        raise refit.Interrupted("simulated preemption")
    monkeypatch.setattr(continuation, "run_validation", interrupted)
    continuation.run(args)
    saved = torch.load(args.output / "latest.pt", weights_only=False)
    source = torch.load(args.source_latest, weights_only=False)
    assert saved["step"] == 10000 and saved["plateau"]["last_validation_step"] is None
    assert torch.equal(saved["cpu_rng"], source["cpu_rng"])
    assert torch.equal(saved["model"]["weight"], source["model"]["weight"])
    assert json.loads((args.output / "status.json").read_text())["validation_pending"]
    monkeypatch.setattr(continuation, "run_validation", original)
    continuation.run(args)
    assert validations == [10000, 10002, 10004, 10006]


def test_partial_best_export_is_repaired_before_advancing_training(tmp_path, monkeypatch):
    args, validations, _, _ = setup_run(tmp_path, monkeypatch)
    real_save = continuation.atomic_save
    def fail_length_export(value, path):
        if path.name == "best-length.pt":
            raise OSError("simulated export interruption")
        real_save(value, path)
    monkeypatch.setattr(continuation, "atomic_save", fail_length_export)
    with pytest.raises(OSError, match="interruption"):
        continuation.run(args)
    saved = torch.load(args.output / "latest.pt", weights_only=False)
    assert saved["step"] == 10000 and saved["plateau"]["last_validation_step"] is None
    assert (args.output / "best.pt").exists() and not (args.output / "best-length.pt").exists()
    monkeypatch.setattr(continuation, "atomic_save", real_save)
    continuation.run(args)
    assert validations == [10000, 10000, 10002, 10004, 10006]
    refit.load_refit_head(args.output / "best-length.pt", file_sha256(args.output / "best.pt"), "data", "cpu")


def test_only_four_failed_dev_checks_can_finish_training(tmp_path, monkeypatch):
    args, validations, _, _ = setup_run(tmp_path, monkeypatch)
    args.updates_per_chunk = 0
    continuation.run(args)
    status = json.loads((args.output / "status.json").read_text())
    assert status["completed"] and status["stopping_reason"] == "dev_bleu_plateau"
    assert validations == [10000, 10002, 10004, 10006, 10008]
    assert status["plateau"]["best_step"] == 10008
