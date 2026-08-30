"""Lightweight, tokenizer-agnostic metrics for text-generation pilots.

The functions in this module operate on integer token ids so that the primary
metrics do not silently change when decoded text normalization changes.  A
reference language model can be enabled explicitly through
``TransformersReferenceLMScorer``; importing this module never downloads a
model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


TokenSequence = Sequence[int]
REFERENCE_LM_SEQUENCE_POLICY = (
  'retokenize_decoded_text_score_through_first_nonleading_eos_v1')


def validate_reference_lm_spec(
    model_name_or_path: str | None,
    revision: str | None,
) -> tuple[str | None, str | None]:
  """Require an immutable revision whenever reference-LM scoring is enabled."""
  model = (
    str(model_name_or_path).strip()
    if model_name_or_path is not None else None)
  pinned_revision = str(revision).strip() if revision is not None else None
  if model == '':
    model = None
  if pinned_revision == '':
    pinned_revision = None
  if model is not None and pinned_revision is None:
    raise ValueError(
      'reference-LM scoring requires --reference-lm-revision')
  if model is None and pinned_revision is not None:
    raise ValueError('--reference-lm-revision requires --reference-lm')
  if (pinned_revision is not None
      and (len(pinned_revision) != 40
           or any(character not in '0123456789abcdef'
                  for character in pinned_revision))):
    raise ValueError(
      '--reference-lm-revision must be a 40-character lowercase commit SHA')
  return model, pinned_revision


def _validate_n(n: int) -> None:
  if n <= 0:
    raise ValueError('n must be positive')


def ngrams(tokens: TokenSequence, n: int) -> list[tuple[int, ...]]:
  """Return contiguous n-grams without adding boundary tokens."""
  _validate_n(n)
  values = [int(token) for token in tokens]
  if len(values) < n:
    return []
  return [tuple(values[index:index + n])
          for index in range(len(values) - n + 1)]


def repetition_rate(tokens: TokenSequence, n: int = 4) -> float:
  """Fraction of within-sequence n-gram occurrences beyond their first.

  A sequence with n-grams ``a, b, a`` has repetition rate 1/3.  Sequences
  shorter than ``n`` have a defined rate of zero.
  """
  values = ngrams(tokens, n)
  if not values:
    return 0.0
  return float(1.0 - len(set(values)) / len(values))


def distinct_n(sequences: Iterable[TokenSequence], n: int = 2) -> float:
  """Corpus-level distinct-n: unique n-grams divided by all n-grams."""
  _validate_n(n)
  counts: Counter[tuple[int, ...]] = Counter()
  for sequence in sequences:
    counts.update(ngrams(sequence, n))
  total = sum(counts.values())
  if total == 0:
    return 0.0
  return float(len(counts) / total)


def ngram_distribution(
    sequences: Iterable[TokenSequence],
    n: int = 2,
) -> dict[tuple[int, ...], float]:
  """Return the empirical corpus n-gram distribution."""
  _validate_n(n)
  counts: Counter[tuple[int, ...]] = Counter()
  for sequence in sequences:
    counts.update(ngrams(sequence, n))
  total = sum(counts.values())
  if total == 0:
    return {}
  return {key: value / total for key, value in counts.items()}


def jensen_shannon_divergence(
    first: dict[tuple[int, ...], float],
    second: dict[tuple[int, ...], float],
) -> float | None:
  """Jensen-Shannon divergence in nats over the union of sparse supports.

  ``None`` is returned when either empirical distribution is empty: assigning
  a numerical score in that case would hide a malformed or too-short sample.
  """
  if not first or not second:
    return None
  support = set(first) | set(second)
  result = 0.0
  for key in support:
    first_value = float(first.get(key, 0.0))
    second_value = float(second.get(key, 0.0))
    mixture = 0.5 * (first_value + second_value)
    if first_value > 0:
      result += 0.5 * first_value * math.log(first_value / mixture)
    if second_value > 0:
      result += 0.5 * second_value * math.log(second_value / mixture)
  return float(result)


def ngram_js_divergence(
    generated: Iterable[TokenSequence],
    reference: Iterable[TokenSequence],
    n: int = 2,
) -> float | None:
  """Convenience wrapper for empirical token n-gram JS divergence."""
  return jensen_shannon_divergence(
    ngram_distribution(generated, n=n),
    ngram_distribution(reference, n=n))


def paired_token_metrics(
    generated: TokenSequence,
    reference: TokenSequence,
) -> dict[str, float | int | bool]:
  """Exact-recovery metrics for aligned generated/reference token spans."""
  generated_values = [int(token) for token in generated]
  reference_values = [int(token) for token in reference]
  if len(generated_values) != len(reference_values):
    raise ValueError('generated and reference token lengths differ')
  matches = sum(
    first == second
    for first, second in zip(generated_values, reference_values))
  count = len(reference_values)
  return {
    'reference_token_count': count,
    'reference_token_accuracy': (float(matches / count) if count else 1.0),
    'reference_exact_match': generated_values == reference_values,
  }


def summarize_token_metrics(
    generated: Sequence[TokenSequence],
    reference: Sequence[TokenSequence] | None = None,
    n_values: Sequence[int] = (1, 2, 4),
) -> dict[str, object]:
  """Compute a compact corpus summary used by the pilot manifest."""
  n_values = tuple(int(n) for n in n_values)
  for n in n_values:
    _validate_n(n)
  summary: dict[str, object] = {
    'num_sequences': len(generated),
    'num_tokens': sum(len(sequence) for sequence in generated),
    'distinct_n': {
      str(n): distinct_n(generated, n=n) for n in n_values
    },
    'mean_repetition_rate': {
      str(n): (
        sum(repetition_rate(sequence, n=n) for sequence in generated)
        / len(generated) if generated else 0.0)
      for n in n_values
    },
  }
  if reference is None:
    summary['reference'] = None
    return summary
  if len(generated) != len(reference):
    raise ValueError('generated and reference sequence counts differ')
  paired = [
    paired_token_metrics(sample, target)
    for sample, target in zip(generated, reference)
  ]
  summary['reference'] = {
    'num_sequences': len(reference),
    'num_tokens': sum(len(sequence) for sequence in reference),
    'exact_match_rate': (
      sum(bool(item['reference_exact_match']) for item in paired)
      / len(paired) if paired else 0.0),
    'token_accuracy': (
      sum(
        float(item['reference_token_accuracy'])
        * int(item['reference_token_count'])
        for item in paired)
      / sum(int(item['reference_token_count']) for item in paired)
      if sum(int(item['reference_token_count']) for item in paired) else 1.0),
    'ngram_js_divergence_nats': {
      str(n): ngram_js_divergence(generated, reference, n=n)
      for n in n_values
    },
  }
  return summary


@dataclass(frozen=True)
class ReferenceLMScore:
  """Length-normalized reference-LM score for one decoded sample."""

  token_count: int
  mean_nll_nats: float | None
  perplexity: float | None


class TransformersReferenceLMScorer:
  """Optional causal-LM scoring hook loaded only when explicitly requested."""

  def __init__(
      self,
      model_name_or_path: str,
      *,
      revision: str,
      device: str = 'cuda',
      batch_size: int = 8,
      max_length: int | None = None,
  ) -> None:
    if batch_size <= 0:
      raise ValueError('batch_size must be positive')
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validated_model, validated_revision = validate_reference_lm_spec(
      model_name_or_path, revision)
    assert validated_model is not None and validated_revision is not None
    self.model_name_or_path = validated_model
    self.revision = validated_revision
    self.device = torch.device(device)
    self.batch_size = int(batch_size)
    self.tokenizer = AutoTokenizer.from_pretrained(
      self.model_name_or_path, revision=self.revision)
    if self.tokenizer.pad_token_id is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model = AutoModelForCausalLM.from_pretrained(
      self.model_name_or_path, revision=self.revision).to(self.device).eval()
    configured_limit = getattr(self.model.config, 'max_position_embeddings', None)
    self.max_length = int(
      max_length if max_length is not None
      else configured_limit if configured_limit is not None
      else 1024)
    if self.max_length < 2:
      raise ValueError('reference-LM max_length must be at least two')

  @torch.no_grad()
  def score(self, texts: Sequence[str]) -> list[ReferenceLMScore]:
    """Score decoded texts through their first EOS, excluding padding.

    EOS is interpreted by the pinned reference tokenizer after decoding and
    re-tokenization.  Its loss is included, while every token after it is
    excluded.  As usual for a causal LM, the first input token has no loss.
    """
    scores: list[ReferenceLMScore] = []
    for offset in range(0, len(texts), self.batch_size):
      batch = list(texts[offset:offset + self.batch_size])
      encoded = self.tokenizer(
        batch,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=self.max_length,
        add_special_tokens=True)
      input_ids = encoded['input_ids'].to(self.device)
      attention_mask = encoded['attention_mask'].to(self.device)
      logits = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask).logits
      shifted_logits = logits[:, :-1].float()
      shifted_targets = input_ids[:, 1:]
      shifted_mask = attention_mask[:, 1:].bool()
      eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
      if eos_token_id is not None:
        valid_eos = (
          input_ids.eq(int(eos_token_id)) & attention_mask.bool())
        # GPT-2 uses the same ID for BOS and EOS.  MDLM sequences carry that
        # boundary marker at position zero, so treating it as the terminal EOS
        # would assign every generated sample zero scored tokens.  Ignore only
        # a valid leading BOS marker; a later occurrence remains terminal.
        bos_token_id = getattr(self.tokenizer, 'bos_token_id', None)
        if bos_token_id is not None:
          first_valid = attention_mask.bool().to(torch.int64).argmax(
            dim=1, keepdim=True)
          leading_bos = (
            input_ids.eq(int(bos_token_id))
            & attention_mask.bool()
            & torch.arange(
              input_ids.shape[1], device=input_ids.device).unsqueeze(0).eq(
                first_valid))
          valid_eos &= ~leading_bos
        positions = torch.arange(
          input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        sentinel = torch.full_like(positions, input_ids.shape[1])
        first_eos = torch.where(valid_eos, positions, sentinel).amin(
          dim=1, keepdim=True)
        through_first_eos = positions.le(first_eos)
        shifted_mask &= through_first_eos[:, 1:]
      token_losses = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_targets,
        reduction='none')
      for row in range(len(batch)):
        count = int(shifted_mask[row].sum().item())
        if count == 0:
          scores.append(ReferenceLMScore(
            token_count=0, mean_nll_nats=None, perplexity=None))
          continue
        mean_nll = float(token_losses[row][shifted_mask[row]].mean().item())
        scores.append(ReferenceLMScore(
          token_count=count,
          mean_nll_nats=mean_nll,
          perplexity=float(math.exp(min(mean_nll, 80.0)))))
    return scores
