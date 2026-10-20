from types import SimpleNamespace

import pytest
import torch

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


@pytest.fixture
def mocked_full_run(tmp_path, monkeypatch):
    from additive_chain import generation, iwslt_length_refit, iwslt_run, translation
    from additive_chain import iwslt_followup as followup
    from additive_chain.iwslt_confirm import dev_cells

    data = tmp_path / "data"
    data.mkdir()
    counts_path = data / "counts.pt"
    counts_path.write_bytes(b"mock training counts")
    full.atomic_json(data / "manifest.json",
                     dict(counts_sha256=full.file_sha256(counts_path)))
    trained, head_path = tmp_path / "translator.pt", tmp_path / "refit-head.pt"
    trained.write_bytes(b"mock frozen translator and old head")
    head_path.write_bytes(b"mock refitted head")
    cells = [dict(c, bleu=10 + (c["temperature"] == .85) + c["strength"])
             for steps in (8, 32)
             for c in dev_cells(steps) + followup.calibration_cells(steps)]
    selection_path = tmp_path / "selection.json"
    full.atomic_json(selection_path, dict(
        cells=cells, selected=followup.select_settings(cells),
        model_sha256=full.file_sha256(trained),
        data_sha256=full.file_sha256(data / "manifest.json"),
        source_sha256=followup.source_hashes()))
    args = SimpleNamespace(data=data, trained=trained, length_head=head_path,
        selection=selection_path, output=tmp_path / "outputs",
        checkpoint=tmp_path / "pretrained", tokenizer=tmp_path / "tokenizer",
        cell_index=0)
    rows = [dict(id=f"dev-{i:05}") for i in range(7283)]
    row_calls, loader_calls, score_calls, decoders, events = [], [], [], [], []
    counts = {"mock": "training counts"}
    backbone_state = {"weight": torch.tensor([1.])}
    old_head_state = {"weight": torch.tensor([[2.]])}
    new_head = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        new_head.weight.fill_(3.)
    new_head.eval()

    class Model:
        def __init__(self):
            self.eval_called = False

        def load_state_dict(self, state):
            assert state is backbone_state

        def eval(self):
            self.eval_called = True
            return self

    model, tokenizer = Model(), object()

    def ordered_rows(root, split):
        row_calls.append((root, split))
        assert root == data and split == "dev"
        return rows

    def load_conditional(checkpoint, tokenizer_path):
        assert checkpoint == args.checkpoint and tokenizer_path == args.tokenizer
        return model, tokenizer

    def load_refit_head(path, model_sha, data_sha, device):
        loader_calls.append((path, model_sha, data_sha, device))
        assert path == head_path and device == "cuda"
        assert model_sha == full.file_sha256(trained)
        assert data_sha == full.file_sha256(data / "manifest.json")
        return new_head

    def load(path, **kwargs):
        if path == trained:
            assert kwargs == dict(map_location="cuda", weights_only=False)
            return dict(model=backbone_state, length_head=old_head_state)
        assert path == counts_path and kwargs == dict(weights_only=True)
        return counts

    class Decoder:
        def __init__(self, **kwargs):
            self.kwargs, self.prepared, self.closed = kwargs, False, False
            decoders.append(self)

        def prepare(self):
            self.prepared = True

        def close(self):
            self.closed = True

    def score(actual_model, actual_tokenizer, head, actual_rows, decoder, cell):
        assert actual_model is model and model.eval_called
        assert actual_tokenizer is tokenizer and head is new_head
        assert actual_rows is rows and len(actual_rows) == 7283
        assert decoder.prepared and not decoder.closed
        score_calls.append(dict(cell))
        return dict(summary=dict(examples=7283, bleu=12., chrf=34., seconds=1.),
                    predictions=[dict(id=row["id"]) for row in actual_rows])

    monkeypatch.setattr(full, "ordered_rows", ordered_rows)
    monkeypatch.setattr(full, "score", score)
    monkeypatch.setattr(translation, "load_conditional", load_conditional)
    monkeypatch.setattr(iwslt_length_refit, "load_refit_head", load_refit_head)
    monkeypatch.setattr(generation, "ChainDecoder", Decoder)
    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "Mock H200")
    monkeypatch.setattr(iwslt_run, "log", lambda event, **fields: events.append((event, fields)))
    inputs = (trained, head_path, selection_path, counts_path, data / "manifest.json")
    return SimpleNamespace(args=args, rows=rows, row_calls=row_calls,
        loader_calls=loader_calls, score_calls=score_calls, decoders=decoders,
        events=events, inputs={path: path.read_bytes() for path in inputs},
        head=new_head, old_head_state=old_head_state, counts=counts)


@pytest.mark.parametrize("cell_index", [0, 1, 2])
def test_run_wires_full_development_and_refit_head_and_resumes(
        mocked_full_run, monkeypatch, cell_index):
    from additive_chain.translation import MASK_ID

    fixture = mocked_full_run
    args = fixture.args
    args.cell_index = cell_index
    original_args = dict(vars(args))
    cell = full.full_cells()[cell_index]
    full.run(args)
    assert fixture.row_calls == [(args.data, "dev")]
    assert len(fixture.loader_calls) == 1
    assert fixture.score_calls == [cell]
    decoder = fixture.decoders[0]
    assert decoder.kwargs == dict(mask_id=MASK_ID, device="cuda", top_k=500,
        nucleus_p=.9, fast_candidates=True, count_statistics=fixture.counts,
        count_strength=cell["strength"], sampling=cell["sampling"], temperature=.85)
    assert decoder.prepared and decoder.closed
    folder = args.output / full.output_name(cell)
    receipt = full.read(folder / "result-identity.json")
    identity = receipt["identity"]
    assert identity["ids"] == [row["id"] for row in fixture.rows]
    assert identity["cell"] == cell and identity["split"] == "dev"
    assert identity["head_sha256"] == full.head_fingerprint(fixture.head.state_dict())
    assert identity["old_head_sha256"] == full.head_fingerprint(fixture.old_head_state)
    assert identity["head_sha256"] != identity["old_head_sha256"]
    assert identity["head_artifact_sha256"] == full.file_sha256(args.length_head)
    assert receipt["output_sha256"] == full.file_sha256(folder / "result.json")
    assert full.read(folder / "summary.json")["examples"] == 7283
    assert fixture.events[0][0] == "full_dev_started"
    assert fixture.events[0][1]["gpu"] == "Mock H200"

    def no_second_score(*args, **kwargs):
        raise AssertionError("A complete checked cell must not be regenerated")

    monkeypatch.setattr(full, "score", no_second_score)
    full.run(args)
    assert fixture.row_calls == [(args.data, "dev")] * 2
    assert len(fixture.loader_calls) == 2
    assert len(fixture.decoders) == 1
    assert fixture.score_calls == [cell]
    assert vars(args) == original_args
    assert all(path.read_bytes() == content for path, content in fixture.inputs.items())


@pytest.mark.parametrize("cell_index", [-1, 12])
def test_run_rejects_invalid_index_before_accessing_any_files(cell_index):
    with pytest.raises(ValueError, match="cell index"):
        full.run(SimpleNamespace(cell_index=cell_index))
