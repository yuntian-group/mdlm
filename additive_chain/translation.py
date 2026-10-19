"""Source-conditioned MDLM training and decoding, with no target-side oracle.

The conditional forward reuses the released DDiT layers and weights. It adds
an explicit key-padding mask, not a different attention architecture. Calling
each block with sample_mode=True selects the ordinary single-sequence rotary
path; dropout still follows module.training and gradients remain enabled.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import CHECKPOINT_FILES, verify_files, verify_tokenizer_files
from .generation import sample_window, stream_seed

MASK_ID = 50257
EOS_ID = 50256
CONTEXT_LENGTH = 1024


@dataclass
class TranslationBatch:
    tokens: torch.Tensor
    valid: torch.Tensor
    target: torch.Tensor
    starts: list
    lengths: list


def collate_pairs(rows, device="cpu", *, source_only=False):
    """Rows have already-tokenized source/target IDs; never truncate here."""
    sequences, starts, lengths = [], [], []
    for row in rows:
        source = list(row["source_ids"])
        target = [] if source_only else list(row["target_ids"]) + [EOS_ID]
        if not source or any(not 0 <= x < EOS_ID for x in source + target[:-1]):
            raise ValueError("Expected nonempty source and ordinary text token IDs")
        prefix = [EOS_ID] + source + [EOS_ID]
        sequence = prefix + target
        if len(sequence) > CONTEXT_LENGTH:
            raise ValueError("Pair exceeds context; preprocessing must declare its handling")
        sequences.append(sequence)
        starts.append(len(prefix))
        lengths.append(len(target))
    width = max(map(len, sequences))
    tokens = torch.full((len(rows), width), EOS_ID, dtype=torch.long, device=device)
    valid = torch.zeros_like(tokens, dtype=torch.bool)
    target = torch.zeros_like(valid)
    for i, sequence in enumerate(sequences):
        tokens[i, :len(sequence)] = torch.tensor(sequence, device=device)
        valid[i, :len(sequence)] = True
        target[i, starts[i]:len(sequence)] = not source_only
    return TranslationBatch(tokens, valid, target, starts, lengths)


def corrupt_target(batch, generator, *, times=None):
    """Log-linear MDLM: p(mask)=t and masked-token loss weight 1/t."""
    if times is None:
        times = .001 + .998 * torch.rand((len(batch.tokens),),
            generator=generator, device=batch.tokens.device)
    if times.shape != (len(batch.tokens),) or not bool(((times > 0) & (times <= 1)).all()):
        raise ValueError("Invalid corruption probabilities")
    selected = (torch.rand(batch.tokens.shape, generator=generator,
        device=batch.tokens.device) < times[:, None]) & batch.target
    noisy = batch.tokens.masked_fill(selected, MASK_ID)
    return noisy, selected, times


class ConditionalMDLM(nn.Module):
    def __init__(self, pretrained, *, amp=True):
        super().__init__()
        self.pretrained = pretrained
        self.amp = amp
        if pretrained.config.cross_attn or pretrained.config.time_conditioning:
            raise ValueError("This adapter requires the pinned non-time-conditioned MDLM")

    @property
    def backbone(self):
        return self.pretrained.backbone

    def features(self, tokens, valid):
        if tokens.shape != valid.shape or valid.dtype != torch.bool or not bool(valid.any(-1).all()):
            raise ValueError("Invalid key-padding mask")
        b = self.backbone
        x = b.vocab_embed(tokens)
        c = F.silu(b.sigma_map(torch.zeros(len(tokens), device=tokens.device)))
        rotary = b.rotary_emb(x)
        mask = valid[:, None, None, :]
        with torch.autocast(tokens.device.type, dtype=torch.bfloat16,
                            enabled=self.amp and tokens.is_cuda):
            for block in b.blocks:
                x = block(x, rotary, c, mask=mask, sample_mode=True, store_kv=False)
        return x, c

    def project(self, hidden, conditioning):
        with torch.autocast(hidden.device.type, dtype=torch.bfloat16,
                            enabled=self.amp and hidden.is_cuda):
            logits = self.backbone.output_layer(hidden, conditioning)
        return logits.float()

    def forward(self, tokens, valid):
        hidden, c = self.features(tokens, valid)
        logits = self.project(hidden, c)
        logits[..., MASK_ID] = -torch.inf
        return SimpleNamespace(log_probs=logits.log_softmax(-1), hidden=hidden.float())

    def loss_sum(self, batch, generator, *, times=None):
        noisy, selected, times = corrupt_target(batch, generator, times=times)
        hidden, c = self.features(noisy, batch.valid)
        if not bool(selected.any()):
            return hidden.sum() * 0, 0
        rows = selected.nonzero()[:, 0]
        logits = self.project(hidden[selected][:, None], c[rows])[:, 0, :MASK_ID]
        nll = F.cross_entropy(logits, batch.tokens[selected], reduction="none")
        return (nll / times[rows]).sum(), int(selected.sum())

    @torch.no_grad()
    def source_features(self, rows, device):
        batch = collate_pairs(rows, device, source_only=True)
        hidden, _ = self.features(batch.tokens, batch.valid)
        pooled = (hidden.float() * batch.valid[..., None]).sum(1) / batch.valid.sum(1)[:, None]
        sizes = torch.tensor([len(r["source_ids"]) for r in rows], device=device).float().log()
        return torch.cat((pooled, sizes[:, None]), -1)


def load_conditional(checkpoint, tokenizer, *, device="cuda", amp=True):
    """Authenticate the existing local release; never download weights."""
    from transformers import AutoModelForMaskedLM, GPT2TokenizerFast
    verify_files(checkpoint, CHECKPOINT_FILES)
    verify_tokenizer_files(tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(str(checkpoint), local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.float32)
    tok = GPT2TokenizerFast.from_pretrained(str(tokenizer), local_files_only=True)
    if tok.vocab_size != 50257 or tok.eos_token_id != EOS_ID:
        raise ValueError("Tokenizer identity changed")
    return ConditionalMDLM(model, amp=amp).to(device), tok


class LengthPredictor(nn.Module):
    """Predict length from SOURCE-ONLY features, including source token count."""
    def __init__(self, hidden_size=768):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size + 1),
            nn.Linear(hidden_size + 1, 256), nn.SiLU(), nn.Linear(256, CONTEXT_LENGTH))

    def forward(self, features):
        return self.net(features.float())


@torch.no_grad()
def translate_one(model, row, length_head, decoder, *, steps, seed, example_id):
    """Only source IDs enter generation. Gold target IDs are deliberately ignored."""
    device = next(model.parameters()).device
    source = {"source_ids": row["source_ids"]}
    features = model.source_features([source], device)
    length = int(length_head(features).argmax(-1).item())
    prefix = [EOS_ID] + list(source["source_ids"]) + [EOS_ID]
    length = max(1, min(length, CONTEXT_LENGTH - len(prefix)))
    prefix = torch.tensor([prefix], device=device)
    target = torch.full((1, length), MASK_ID, dtype=torch.long, device=device)

    def forward(current, _time):
        tokens = torch.cat((prefix, current), -1)
        output = model(tokens, torch.ones_like(tokens, dtype=torch.bool))
        return SimpleNamespace(log_probs=output.log_probs[:, -length:],
                               hidden=output.hidden[:, -length:])

    generators = {k: torch.Generator(device=device).manual_seed(
        stream_seed(f"iwslt-{seed}", example_id, k)) for k in ("reveal", "proposal")}
    result = sample_window(target, forward, decoder, steps=steps,
        reveal_generator=generators["reveal"], proposal_generator=generators["proposal"])
    ids = result.tokens[0].tolist()
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]
    return ids, dict(result.stats, predicted_length=length, length_prediction_forwards=1)
