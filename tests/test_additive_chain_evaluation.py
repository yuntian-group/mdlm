"""CPU-only metric/CLI tests: no Hugging Face downloads or real GPU scoring."""
import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from additive_chain import evaluation as ev


class TinyScorer(torch.nn.Module):
    """Context-sensitive mock: retaining internal EOS changes later logits."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        context = (input_ids * attention_mask).cumsum(-1).float()[..., None]
        vocab = torch.arange(7, device=input_ids.device).float()
        logits = torch.cos(context * .13 + vocab * .71) + vocab * .03
        return SimpleNamespace(logits=logits)


def upstream_reference(model, ids, mask, eos=6, context=1024, stride=512):
    """Literal single-batch arithmetic from BD3-LM metrics.py, commit1c3e8f43.

    Independent accumulation/slicing path; no dependency on the private runner.
    """
    nlls_accum = torch.zeros_like(ids, dtype=torch.float32)
    weights = torch.zeros_like(ids, dtype=torch.float32)
    count = max(math.ceil((ids.shape[-1] - context + stride) / stride), 1)
    for i in range(count):
        start = 0 if i == 0 else i * stride
        end = min(start + context, ids.shape[-1])
        chunk = ids[..., start:end]
        logits = model(chunk, attention_mask=mask[..., start:end]).logits.transpose(-1, -2)
        nlls = F.cross_entropy(logits[..., :-1], chunk[..., 1:], reduction="none")
        valid = (chunk[..., 1:] != eos).float()
        if i == 0:
            nlls_accum[..., start + 1:end] += nlls
            weights[..., start + 1:end] += valid
        else:
            update_start = start + context - stride
            width = end - update_start
            nlls_accum[..., update_start:end] += nlls[..., -width:]
            weights[..., update_start:end] += valid[..., -width:]
    return (nlls_accum.double() * weights.double()).sum().item(), int(weights.sum()), \
        ((nlls_accum * weights).sum() / weights.sum()).exp().item(), weights.bool()


@pytest.mark.parametrize("length", [2, 17, 1023, 1024, 1025, 1536, 2048, 2051])
def test_scoring_matches_pinned_upstream_cpu_arithmetic(length):
    ids = torch.arange(length).remainder(7)[None].long()
    mask = torch.ones_like(ids)
    actual = ev.score_token_batch(TinyScorer().eval(), ids, mask, eos_token_id=6)[0]
    total, count, legacy, valid = upstream_reference(TinyScorer().eval(), ids, mask)
    assert actual["total_nll_float64_sum"] == total
    assert actual["scored_tokens"] == count
    assert actual["legacy_sequence_ppl"] == legacy
    assert actual["scored_target_mask"] == valid[0].tolist()


@pytest.mark.parametrize("length", [2, 1024, 1025, 1536, 2048, 2049])
def test_each_target_scored_once_and_context_bounded(length):
    covered = []
    for start, end, first in ev.scoring_windows(length):
        assert 0 <= start < first < end <= length
        assert end - start <= 1024
        covered += list(range(first, end))
    assert covered == list(range(1, length))


def test_eos_remains_context_but_never_target_or_padding_target():
    ids = torch.tensor([[6, 1, 6, 2, 6, 6], [6, 2, 1, 3, 2, 1]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    model = TinyScorer().eval()
    rows = ev.score_token_batch(model, ids, mask, eos_token_id=6)
    assert rows[0]["scored_target_mask"] == [False, True, False, True, False, False]
    assert len(model.calls) == 2
    assert torch.equal(model.calls[0][0], ids[:1])
    assert torch.equal(model.calls[0][1], mask[:1])
    removed = ev.score_token_batch(TinyScorer().eval(), ids[:, [0, 1, 3]],
        torch.ones(2, 3, dtype=torch.long), eos_token_id=6)
    assert removed[0]["total_nll_float64_sum"] != rows[0]["total_nll_float64_sum"]


def test_pooled_not_mean_sequence_ppl():
    result = ev.aggregate_scores([
        {"total_nll_float64_sum": math.log(2), "scored_tokens": 1, "legacy_sequence_ppl": 2.},
        {"total_nll_float64_sum": 9 * math.log(8), "scored_tokens": 9, "legacy_sequence_ppl": 8.}])
    assert result["legacy_mean_sequence_ppl"] == 5
    assert result["pooled_token_ppl"] == pytest.approx(2 ** .1 * 8 ** .9)
    assert result["scored_tokens"] == 10


@pytest.mark.parametrize("kind", ["eos_only", "left_pad", "wrong_pad", "train", "bf16"])
def test_scoring_rejects_incompatible_or_undefined_inputs(kind):
    model = TinyScorer().eval()
    ids, mask = torch.tensor([[6, 1, 2]]), torch.tensor([[1, 1, 1]])
    if kind == "eos_only":
        ids[:] = 6
    elif kind == "left_pad":
        mask[0, 0] = 0
    elif kind == "wrong_pad":
        mask[0, -1] = 0
    elif kind == "train":
        model.train()
    else:
        original = model.forward
        model.forward = lambda *a, **kw: SimpleNamespace(logits=original(*a, **kw).logits.bfloat16())
    with pytest.raises(ValueError):
        ev.score_token_batch(model, ids, mask, eos_token_id=6)


def attempts():
    return [{"attempt_id": i, "accepted": i != 1, "token_ids": [6, 1, 2, 1, 2],
             "generation_seconds": i + 1., "new_tokens": 4} for i in range(3)]


def test_rejected_work_included_and_setup_warmup_separate():
    result = ev.summarize_attempts(attempts(), eos_token_id=6, samples_requested=2,
        timing_complete=True, setup_seconds=8., warmup_seconds=4.)
    assert result["population_complete"] and result["timing_complete"]
    assert (result["attempts"], result["accepted"], result["rejected"]) == (3, 2, 1)
    assert result["generation_seconds_recorded"] == 6
    assert result["seconds_per_accepted_sequence"] == 3
    assert result["seconds_per_raw_sequence"] == 2
    assert result["raw_new_tokens_per_second"] == 2
    assert result["accepted_new_tokens_per_second"] == pytest.approx(8 / 6)
    assert result["setup_warmup_amortized_seconds_per_accepted_sequence"] == 9


def test_missing_warmup_is_not_zero_and_partial_population_not_final():
    result = ev.summarize_attempts(attempts(), eos_token_id=6, samples_requested=4,
                                 timing_complete=True, setup_seconds=0.)
    assert not result["population_complete"]
    assert result["setup_warmup_amortized_seconds_per_accepted_sequence"] is None
    assert result["seconds_per_accepted_sequence"] == 3


@pytest.mark.parametrize("missing", [False, True])
def test_incomplete_timing_not_advertised_as_complete_latency(missing):
    rows = attempts()
    if missing:
        rows[1]["generation_seconds"] = None
    result = ev.summarize_attempts(rows, eos_token_id=6, samples_requested=2,
                                 timing_complete=missing)
    assert not result["timing_complete"]
    assert result["seconds_per_accepted_sequence"] is None
    assert result["raw_new_tokens_per_second"] is None


def test_attempt_validation_keeps_rejections_and_unknown_tokens():
    rows = attempts()
    rows[0].pop("new_tokens")
    result = ev.summarize_attempts(rows, eos_token_id=6, samples_requested=2, timing_complete=True)
    assert result["accepted_new_tokens_per_second"] is None
    with pytest.raises(ValueError, match="raw attempt"):
        ev.summarize_attempts(rows[::2], eos_token_id=6, samples_requested=2, timing_complete=True)
    rows[1]["generation_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ev.summarize_attempts(rows, eos_token_id=6, samples_requested=2, timing_complete=True)


def test_entropy_and_repeated_ngrams_use_raw_tokens_including_eos():
    result = ev.token_diagnostics([6, 6, 6, 6, 6], 6)
    assert result["token_entropy_nats"] == 0
    assert result["repeated_4gram_fraction"] == .5
    assert result["eos_tokens_including_initial_context"] == 5
    assert ev.token_diagnostics([6, 1, 2], 6)["repeated_4gram_fraction"] == 0
    assert ev.last256_entropy([6] * 256) == 0
    assert ev.last256_entropy(list(range(256))) == pytest.approx(math.log(256), abs=1e-6)


class Tokenizer:
    eos_token_id, pad_token_id, eos_token = 6, 6, "<eos>"
    padding_side = "right"

    def batch_decode(self, sequences, skip_special_tokens):
        assert skip_special_tokens is False
        return [" ".join(str(v) for v in row) for row in sequences]

    def __call__(self, texts, **kwargs):
        assert kwargs == dict(return_tensors="pt", return_token_type_ids=False,
            return_attention_mask=True, truncation=True, padding=True, max_length=1024)
        rows = [[int(v) for v in text.split()][:1024] for text in texts]
        length = max(map(len, rows))
        return {"input_ids": torch.tensor([r + [6] * (length - len(r)) for r in rows]),
                "attention_mask": torch.tensor([[1] * len(r) + [0] * (length - len(r)) for r in rows])}


def saved():
    return dict(schema="additive_chain_saved_generations_v1", eos_token_id=6,
        samples_requested=2, generation_timing_complete=True, attempts=attempts())


def test_saved_integration_decodes_only_accepted_without_filtering():
    data = saved()
    result = ev.evaluate_saved(data, Tokenizer(), Tokenizer(), TinyScorer().eval())
    assert result["quality_population"] == "complete"
    assert result["quality"]["samples"] == 2
    assert [r["attempt_id"] for r in result["score_rows"]] == [0, 2]
    assert result["generation"]["rejected"] == 1
    # Low entropy here is intentional: evaluation must not re-filter saved data.
    assert max(result["generation"]["last256_entropy_nats_all_attempts"]) < 4
    data["attempts"][0]["text"] = "EOS removed"
    with pytest.raises(ValueError, match="retained"):
        ev.evaluate_saved(data, Tokenizer(), Tokenizer(), TinyScorer().eval())


def test_zero_accepted_or_partial_never_claims_complete_quality():
    data = saved()
    for row in data["attempts"]:
        row["accepted"] = False
    result = ev.evaluate_saved(data, Tokenizer(), Tokenizer(), TinyScorer().eval())
    assert result["quality"] is None and result["quality_population"] == "partial"
    assert result["generation"]["seconds_per_accepted_sequence"] is None


def test_cli_cached_only_cpu_mock_and_no_overwrite(tmp_path, monkeypatch):
    calls = []
    def tokenizer_loader(name, **kwargs):
        calls.append(kwargs)
        return Tokenizer()
    def model_loader(name, **kwargs):
        calls.append(kwargs)
        return TinyScorer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader)))
    source, output = tmp_path / "samples.json", tmp_path / "metrics.json"
    source.write_text(json.dumps(saved()))
    ev.main([str(source), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["quality"]["samples"] == 2
    assert result["environment"]["device"] == "cpu"
    assert result["scoring_seconds"] > 0
    assert all(c["local_files_only"] is True and c["trust_remote_code"] is False for c in calls)
    assert calls[-1]["torch_dtype"] == torch.float32
    with pytest.raises(ValueError, match="Output exists"):
        ev.main([str(source), "--output", str(output)])
