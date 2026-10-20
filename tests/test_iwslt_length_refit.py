from types import SimpleNamespace

import pytest
import torch

from additive_chain import iwslt_length_refit as refit
from additive_chain.translation import CONTEXT_LENGTH, LengthPredictor


def rows():
    return [dict(id=str(i), source_ids=list(range(1, i + 2)), target_ids=[1] * (i + 2))
            for i in range(5)]


class FeatureModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def source_features(self, sources, device):
        assert all(set(row) == {"source_ids"} for row in sources)
        self.calls.extend(sources)
        return torch.tensor([[len(row["source_ids"]), row["source_ids"][0], 2.]
                             for row in sources], device=device)


def test_cache_is_source_only_order_preserving_and_resumable(tmp_path):
    model = FeatureModel()
    population = rows()[::-1]
    identity = dict(model_sha256="model", data_sha256="data")
    args = dict(hidden_size=2, device="cpu", shard_size=3, batch_size=2)
    cached = refit.cache_features(model, population, "train", tmp_path, identity, **args)
    assert len(model.calls) == 5
    assert cached["features"][:, 0].tolist() == [5, 4, 3, 2, 1]
    assert cached["labels"].tolist() == [7, 6, 5, 4, 3]
    assert cached["maximum"].tolist() == [CONTEXT_LENGTH - n - 2 for n in [5, 4, 3, 2, 1]]
    refit.cache_features(model, population, "train", tmp_path, identity, **args)
    assert len(model.calls) == 5
    with pytest.raises(ValueError, match="identity"):
        refit.cache_features(model, population, "train", tmp_path, dict(model_sha256="different"), **args)
    with pytest.raises(ValueError, match="Invalid"):
        refit.cache_features(model, population, "test", tmp_path, identity, **args)


def test_feature_interruption_and_corrupt_cache(tmp_path):
    model, population = FeatureModel(), rows()
    kwargs = dict(hidden_size=2, device="cpu", shard_size=3, batch_size=2)
    with pytest.raises(refit.Interrupted):
        refit.cache_features(model, population, "dev", tmp_path, {}, stopped=lambda: True, **kwargs)
    assert len(model.calls) == 3
    refit.cache_features(model, population, "dev", tmp_path, {}, **kwargs)
    assert len(model.calls) == 5
    path = tmp_path / "dev-000000.pt"
    saved = torch.load(path, weights_only=True)
    saved["features"][0, 0] = float("nan")
    torch.save(saved, path)
    with pytest.raises(ValueError, match="values"):
        refit.cache_features(model, population, "dev", tmp_path, {}, **kwargs)


def test_feature_batches_are_source_only_and_respect_padding_budget():
    population = rows()[::-1]
    batches = refit.feature_batches(population, batch_size=3, token_budget=12)
    assert sorted(i for batch in batches for i in batch) == list(range(5))
    for batch in batches:
        assert len(batch) <= 3
        assert len(batch) * max(len(population[i]["source_ids"]) + 2 for i in batch) <= 12
    changed_targets = [dict(row, target_ids=[1] * 500) for row in population]
    assert refit.feature_batches(changed_targets, batch_size=3, token_budget=12) == batches
    with pytest.raises(ValueError, match="exceeds"):
        refit.feature_batches(population, token_budget=3)


def test_batch_check_has_no_target_access_and_reports_predictions():
    model = FeatureModel()
    head = LengthPredictor(2)
    result = refit.check_feature_batching(model, head, rows(), "cpu")
    assert result["feature_max_abs_difference"] == 0
    assert result["old_head_prediction_agreement"] == 1
    assert len(result["ids"]) == 5
    state = head.state_dict()
    original = refit._state_digest(state)
    state["net.0.weight"][0] += 1
    assert refit._state_digest(state) != original


def test_registered_population_is_complete_train_dev_only(tmp_path, monkeypatch):
    calls = []
    def dataset(root, split):
        calls.append(split)
        return rows()
    monkeypatch.setattr(refit, "dataset", dataset)
    monkeypatch.setattr(refit, "SPLIT_COUNTS", dict(train=5, dev=5))
    assert refit.registered_rows(tmp_path, "train") == rows()
    assert refit.registered_rows(tmp_path, "dev") == rows()
    with pytest.raises(ValueError, match="train/dev"):
        refit.registered_rows(tmp_path, "test")
    assert calls == ["train", "dev"]
    monkeypatch.setitem(refit.SPLIT_COUNTS, "train", 6)
    with pytest.raises(ValueError, match="complete"):
        refit.registered_rows(tmp_path, "train")


def test_each_epoch_covers_every_row_once_including_tail():
    first = refit.epoch_batches(519, 1)
    assert [len(part) for part in first] == [256, 256, 7]
    assert torch.equal(torch.cat(first).sort().values, torch.arange(519))
    assert torch.equal(torch.cat(first), torch.cat(refit.epoch_batches(519, 1)))
    assert not torch.equal(torch.cat(first), torch.cat(refit.epoch_batches(519, 2)))


def test_metrics_match_generation_length_clipping():
    class Head:
        def eval(self):
            return self
        def __call__(self, x):
            logits = torch.full((len(x), CONTEXT_LENGTH), -100.)
            logits[torch.arange(len(x)), x[:, 0].long()] = 0.
            return logits
    cached = dict(features=torch.tensor([[0.], [9.], [4.], [4.]]),
        labels=torch.tensor([1, 6, 3, 5]), maximum=torch.tensor([10, 6, 10, 10]))
    metrics = refit.length_metrics(Head(), cached, "cpu", batch_size=2)
    assert metrics["examples"] == 4
    assert metrics["mae"] == .5
    assert metrics["mean_error"] == 0
    assert metrics["exact"] == .5
    assert metrics["within1"] == 1
    assert metrics["under_fraction"] == .25
    assert metrics["over_fraction"] == .25
    assert metrics["clipped_fraction"] == .5


def _cached(examples=9):
    return dict(features=torch.arange(examples * 3).reshape(examples, 3).float() / 100,
        labels=torch.full((examples,), 3), maximum=torch.full((examples,), 100))


def test_training_resume_matches_uninterrupted_and_selects_dev_only(tmp_path, monkeypatch):
    monkeypatch.setattr(refit, "TRAINING", dict(seed=2718, learning_rate=.001, weight_decay=.01, batch_size=4))
    monkeypatch.setattr(refit, "source_hashes", lambda: {"test": "source"})
    identity = dict(model_sha256="model", data_sha256="data", source_sha256=refit.source_hashes())
    torch.manual_seed(8)
    old_head = LengthPredictor(2)
    old_state = {name: tensor.clone() for name, tensor in old_head.state_dict().items()}
    args = dict(hidden_size=2, device="cpu", max_epochs=3, patience=5)
    complete = refit.fit_cached(_cached(), _cached(4), old_head, identity, tmp_path / "complete", **args)
    with pytest.raises(refit.Interrupted):
        refit.fit_cached(_cached(), _cached(4), old_head, identity, tmp_path / "resume",
                         stopped=lambda: True, **args)
    resumed = refit.fit_cached(_cached(), _cached(4), old_head, identity, tmp_path / "resume", **args)
    assert complete["history"] == resumed["history"]
    assert complete["refit_dev"] == resumed["refit_dev"]
    assert complete["old_dev"] == resumed["old_dev"]
    assert complete["epochs"] == 3
    for name, tensor in old_head.state_dict().items():
        assert torch.equal(tensor, old_state[name])
    first = refit.load_refit_head(tmp_path / "complete/best.pt", "model", "data", "cpu")
    second = refit.load_refit_head(tmp_path / "resume/best.pt", "model", "data", "cpu")
    for name in first.state_dict():
        assert torch.equal(first.state_dict()[name], second.state_dict()[name])
    with pytest.raises(ValueError, match="identity"):
        refit.load_refit_head(tmp_path / "complete/best.pt", "wrong", "data", "cpu")
    with pytest.raises(ValueError, match="identity"):
        refit.fit_cached(_cached(), _cached(4), old_head, dict(identity, model_sha256="wrong"),
                         tmp_path / "complete", **args)
    state = torch.load(tmp_path / "complete/best.pt", weights_only=True)
    assert "model" not in state and "optimizer" not in state
    assert all(r["train_examples"] == 9 for r in complete["history"])


def test_dev_patience_stops_and_keeps_first_best_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(refit, "length_metrics", lambda *a, **k: dict(length_ce=3., examples=4))
    summary = refit.fit_cached(_cached(), _cached(4), LengthPredictor(2), {},
        tmp_path, hidden_size=2, device="cpu", max_epochs=10, patience=2)
    assert summary["epochs"] == 3
    assert summary["selected_epoch"] == 1
    assert summary["stopping_reason"] == "dev_patience"


def test_run_freezes_translator_and_never_reads_test(tmp_path, monkeypatch):
    class FrozenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.pretrained = SimpleNamespace(config=SimpleNamespace(hidden_dim=2))

        def source_features(self, sources, device):
            assert not self.training and not self.weight.requires_grad
            assert all(set(row) == {"source_ids"} for row in sources)
            return torch.tensor([[len(row["source_ids"]), 1., 2.] for row in sources])

    model, old_head = FrozenModel(), LengthPredictor(2)
    trained = tmp_path / "model.pt"
    torch.save(dict(model=model.state_dict(), length_head=old_head.state_dict()), trained)
    original_sha = refit.file_sha256(trained)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    calls = []
    def read_dataset(root, split):
        calls.append(split)
        return rows()
    monkeypatch.setattr(refit, "dataset", read_dataset)
    monkeypatch.setattr(refit, "SPLIT_COUNTS", dict(train=5, dev=5))
    monkeypatch.setattr(refit, "load_conditional", lambda *args, **kwargs: (model, None))
    args = SimpleNamespace(checkpoint=tmp_path, tokenizer=tmp_path, trained=trained,
        data=data, output=tmp_path / "run", device="cpu", feature_batch_size=3,
        feature_token_budget=16384, max_epochs=2, patience=1)
    refit.run(args)
    assert calls == ["train", "dev"]
    assert refit.file_sha256(trained) == original_sha
    saved = torch.load(args.output / "best.pt", weights_only=True)
    assert saved["identity"]["old_length_head_sha256"] == refit._state_digest(old_head.state_dict())
    assert saved["identity"]["train_examples"] == 5
    assert (args.output / "summary.json").exists()
    assert (args.output / "feature-batch-check.json").exists()
