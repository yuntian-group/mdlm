import unittest
from unittest import mock

from evaluation.generation_metrics import ReferenceLMScore
from scripts.run_generation_pilot import (
  _attach_reference_lm_scores,
  _summarize_attached_reference_lm,
)


class GenerationReferenceLMPlumbingTest(unittest.TestCase):

  @mock.patch('scripts.run_generation_pilot.TransformersReferenceLMScorer')
  def test_revision_is_recorded_in_rows_and_summaries(self, scorer_class):
    scorer = scorer_class.return_value
    scorer.score.return_value = [
      ReferenceLMScore(token_count=4, mean_nll_nats=1.0, perplexity=2.718),
      ReferenceLMScore(token_count=2, mean_nll_nats=2.0, perplexity=7.389),
    ]
    rows = [{'text': 'one'}, {'text': 'two'}]
    revision = 'c' * 40

    overall = _attach_reference_lm_scores(
      rows,
      model_name_or_path='org/model',
      revision=revision,
      device='cpu',
      batch_size=2)
    group = _summarize_attached_reference_lm(rows)

    scorer_class.assert_called_once_with(
      'org/model', revision=revision, device='cpu', batch_size=2)
    self.assertTrue(all(
      row['reference_lm']['revision'] == revision for row in rows))
    self.assertEqual(overall['revision'], revision)
    self.assertEqual(group['revision'], revision)
    self.assertEqual(group['model_name_or_path'], 'org/model')
    self.assertAlmostEqual(group['mean_nll_nats'], 4 / 3)


if __name__ == '__main__':
  unittest.main()
