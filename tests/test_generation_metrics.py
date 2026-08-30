import math
import unittest
from unittest import mock

from types import SimpleNamespace

from evaluation.generation_metrics import (
  distinct_n,
  jensen_shannon_divergence,
  ngram_distribution,
  ngram_js_divergence,
  ngrams,
  paired_token_metrics,
  repetition_rate,
  summarize_token_metrics,
  TransformersReferenceLMScorer,
  validate_reference_lm_spec,
)


class GenerationMetricsTest(unittest.TestCase):

  def test_ngrams_distinct_and_repetition_have_explicit_semantics(self):
    self.assertEqual(ngrams([1, 2, 1, 2], 2), [(1, 2), (2, 1), (1, 2)])
    self.assertAlmostEqual(repetition_rate([1, 2, 1, 2], n=2), 1 / 3)
    self.assertAlmostEqual(
      distinct_n([[1, 2, 1, 2], [1, 2]], n=2), 2 / 4)
    self.assertEqual(repetition_rate([1, 2], n=4), 0.0)
    self.assertEqual(distinct_n([[1], []], n=2), 0.0)

  def test_sparse_js_is_symmetric_bounded_and_uses_natural_log(self):
    first = ngram_distribution([[1, 1]], n=1)
    second = ngram_distribution([[2, 2]], n=1)
    forward = jensen_shannon_divergence(first, second)
    backward = jensen_shannon_divergence(second, first)
    self.assertAlmostEqual(forward, math.log(2))
    self.assertAlmostEqual(backward, forward)
    self.assertAlmostEqual(
      ngram_js_divergence([[1, 2]], [[1, 2]], n=1), 0.0)
    self.assertIsNone(ngram_js_divergence([[1]], [[1]], n=2))

  def test_paired_token_metrics_validate_alignment(self):
    metrics = paired_token_metrics([1, 2, 3], [1, 9, 3])
    self.assertEqual(metrics['reference_token_count'], 3)
    self.assertAlmostEqual(metrics['reference_token_accuracy'], 2 / 3)
    self.assertFalse(metrics['reference_exact_match'])
    with self.assertRaisesRegex(ValueError, 'lengths differ'):
      paired_token_metrics([1], [1, 2])

  def test_summary_uses_corpus_level_ngram_js_and_token_weighting(self):
    summary = summarize_token_metrics(
      [[1, 2], [3]],
      reference=[[1, 9], [3]],
      n_values=(1, 2))
    self.assertEqual(summary['num_sequences'], 2)
    self.assertEqual(summary['num_tokens'], 3)
    self.assertAlmostEqual(summary['reference']['token_accuracy'], 2 / 3)
    self.assertAlmostEqual(summary['reference']['exact_match_rate'], 0.5)
    self.assertIn('1', summary['reference']['ngram_js_divergence_nats'])

  def test_invalid_n_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'positive'):
      ngrams([1, 2], 0)

  def test_reference_lm_requires_paired_model_and_revision(self):
    self.assertEqual(
      validate_reference_lm_spec('org/model', 'a' * 40),
      ('org/model', 'a' * 40))
    self.assertEqual(validate_reference_lm_spec(None, None), (None, None))
    with self.assertRaisesRegex(ValueError, 'requires.*revision'):
      validate_reference_lm_spec('org/model', None)
    with self.assertRaisesRegex(ValueError, 'requires --reference-lm'):
      validate_reference_lm_spec(None, 'a' * 40)
    with self.assertRaisesRegex(ValueError, '40-character'):
      validate_reference_lm_spec('org/model', 'main')

  @mock.patch('transformers.AutoModelForCausalLM.from_pretrained')
  @mock.patch('transformers.AutoTokenizer.from_pretrained')
  def test_reference_lm_revision_is_passed_to_both_loaders(
      self, tokenizer_loader, model_loader):
    tokenizer = SimpleNamespace(pad_token_id=0)
    tokenizer_loader.return_value = tokenizer
    model = mock.Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.config = SimpleNamespace(max_position_embeddings=128)
    model_loader.return_value = model

    scorer = TransformersReferenceLMScorer(
      'org/model', revision='b' * 40, device='cpu')

    tokenizer_loader.assert_called_once_with(
      'org/model', revision='b' * 40)
    model_loader.assert_called_once_with(
      'org/model', revision='b' * 40)
    self.assertEqual(scorer.revision, 'b' * 40)


if __name__ == '__main__':
  unittest.main()
