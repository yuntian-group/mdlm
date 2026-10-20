from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from additive_chain.generation import ChainDecoder, generator_device
from additive_chain.iwslt_data import filter_training, xml_segments, unix_lines
from additive_chain.translation import (ConditionalMDLM, collate_pairs, corrupt_target,
    translate_one, MASK_ID, EOS_ID)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)

    def forward(self, x, rotary, c, mask=None, **kwargs):
        q, k, v = self.qkv(x).chunk(3, -1)
        return x + F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None],
                                                 attn_mask=mask)[:, 0]


class Final(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, MASK_ID + 1)

    def forward(self, x, c):
        return self.linear(x)


def tiny():
    b = nn.Module()
    b.vocab_embed = nn.Embedding(MASK_ID + 1, 4)
    b.sigma_map = lambda t: torch.zeros(len(t), 4)
    b.rotary_emb = lambda x: None
    b.blocks = nn.ModuleList([Block()])
    b.output_layer = Final()
    p = nn.Module()
    p.config = SimpleNamespace(cross_attn=False, time_conditioning=False, hidden_dim=4)
    p.backbone = b
    return ConditionalMDLM(p, amp=False)


def test_collation_and_target_only_mask():
    rows = [dict(source_ids=[2, 3], target_ids=[4, 5]), dict(source_ids=[6], target_ids=[7])]
    batch = collate_pairs(rows)
    noisy, selected, _ = corrupt_target(batch, torch.Generator().manual_seed(1), times=torch.ones(2))
    assert torch.equal(selected, batch.target)
    assert torch.equal(noisy[~batch.target], batch.tokens[~batch.target])
    assert batch.lengths == [3, 2]
    assert not batch.target[1, -1]
    assert not batch.valid[1, -1]
    assert batch.tokens[1, -1] == EOS_ID
    assert noisy[0].tolist() == [EOS_ID, 2, 3, EOS_ID, MASK_ID, MASK_ID, MASK_ID]


def test_padding_invariance_and_gradient():
    torch.manual_seed(1)
    model = tiny()
    short = dict(source_ids=[2], target_ids=[3])
    long = dict(source_ids=[3, 4, 5, 6], target_ids=[7, 8])
    one, padded = collate_pairs([short]), collate_pairs([short, long])
    a = model(one.tokens, one.valid).log_probs
    b = model(padded.tokens, padded.valid).log_probs[:1, :one.tokens.shape[1]]
    torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    loss, count = model.loss_sum(padded, torch.Generator().manual_seed(3), times=torch.ones(2))
    assert count > 0
    loss.backward()
    assert torch.isfinite(loss) and model.backbone.vocab_embed.weight.grad.abs().sum() > 0


def test_source_only_does_not_read_target():
    class BadTarget(dict):
        def __getitem__(self, key):
            if key in ("target_ids", "target"):
                raise AssertionError("Target leaked")
            return super().__getitem__(key)
    row = BadTarget(source_ids=[4, 5])
    batch = collate_pairs([row], source_only=True)
    assert not batch.target.any()
    model = tiny().eval()
    head = lambda features: torch.tensor([[0., 0., 0., 10., 0.]])
    decoder = ChainDecoder(mask_id=MASK_ID, top_k=2, nucleus_p=1.)
    decoder.prepare()
    try:
        ids, stats = translate_one(model, row, head, decoder, steps=4, seed=1, example_id=0)
        assert len(ids) <= 3 and stats["predicted_length"] == 3
        assert MASK_ID not in ids
    finally:
        decoder.close()


def test_context_overflow_not_silently_truncated():
    with pytest.raises(ValueError, match="context"):
        collate_pairs([dict(source_ids=[1] * 1020, target_ids=[2, 3])])


def test_reference_length_requires_explicit_override_and_is_labeled():
    model = tiny().eval()
    head = lambda features: torch.tensor([[0., 0., 0., 10., 0.]])
    decoder = ChainDecoder(mask_id=MASK_ID, top_k=2, nucleus_p=1.)
    decoder.prepare()
    row = dict(source_ids=[4, 5], target_ids=[6] * 20)
    try:
        _, ordinary = translate_one(model, row, head, decoder, steps=2, seed=1, example_id=0)
        _, oracle = translate_one(model, row, head, decoder, steps=2, seed=1,
                                  example_id=0, length_override=5)
        assert ordinary["predicted_length"] == 3
        assert "length_mode" not in ordinary
        assert oracle["predicted_length"] == oracle["requested_reference_length"] == 5
        assert oracle["source_predicted_length"] == 3
        assert oracle["length_mode"] == "reference-diagnostic"
        with pytest.raises(ValueError, match="context"):
            translate_one(model, row, head, decoder, steps=2, seed=1,
                          example_id=0, length_override=1024)
    finally:
        decoder.close()


def test_fairseq_text_filters():
    assert unix_lines("one\x85sentence\nnext\n") == ["one\x85sentence", "next"]
    assert filter_training(["<url>x</url>", "<talkid>1</talkid>", "<title>hello</title>", "text"]) == ["hello", "text"]
    assert xml_segments('<seg id="1"> hello </seg>\n<seg id="2"> isn’t </seg>') == ["hello", "isn't"]


def test_temperature_preserves_candidates_and_tempers_complete_chain():
    stats = dict(left_counts=torch.tensor([3., 2., 0.]), right_counts=torch.tensor([2., 3., 0.]),
                 pair_keys=torch.tensor([1, 3]), pair_counts=torch.tensor([3., 2.]))
    ordinary = ChainDecoder(mask_id=2, count_statistics=stats, count_strength=.2,
                            top_k=2, nucleus_p=1.)
    tempered = ChainDecoder(mask_id=2, count_statistics=stats, count_strength=.2,
                            temperature=.5, top_k=2, nucleus_p=1.)
    probabilities = torch.tensor([[[.7, .3, 0.], [.4, .6, 0.]]]).log()
    tokens = torch.tensor([[2, 2]])
    a, b = ordinary._candidates(probabilities, tokens), tempered._candidates(probabilities, tokens)
    assert torch.equal(a.candidate_ids, b.candidate_ids)
    torch.testing.assert_close(b.log_unary, (a.log_unary / .5).log_softmax(-1))
    def dense(p):
        result = p.left[:, None] * p.right[None]
        result.flatten()[p.pair_keys] += p.row_values
        return result * math.exp(p.log_scale)
    import math
    torch.testing.assert_close(dense(tempered.potential), dense(ordinary.potential) ** 2)


def test_implicit_cuda_generator_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert generator_device(SimpleNamespace(device=torch.device("cuda"))) == torch.device("cuda:2")
    assert generator_device(SimpleNamespace(device=torch.device("cuda:1"))) == torch.device("cuda:1")
    assert generator_device(torch.Generator()) == torch.device("cpu")
