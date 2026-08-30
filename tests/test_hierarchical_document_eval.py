import hashlib
import unittest

from scripts.aggregate_hierarchical_document_eval import (
  aggregate_records,
  validate_record,
)


def _digest(text):
  return hashlib.sha256(text.encode()).hexdigest()


def _records():
  rows = []
  for dataset in ('wiki-pinned', 'arxiv-pinned'):
    for mask_rate in (0.5, 0.9):
      for train_seed in (1, 2):
        for corruption_seed in (201, 202):
          pairing = _digest(
            f'{dataset}|{mask_rate}|{corruption_seed}|corruption')
          for arm in ('static_static', 'dynamic_dynamic'):
            for document_index in (10, 20):
              for chunk_index in (0, 1):
                masked_tokens = 4 + chunk_index
                baseline_nll = 2.0 + 0.1 * (document_index == 20)
                per_token = (
                  baseline_nll if arm == 'static_static'
                  else baseline_nll - 0.5)
                rows.append({
                  'schema_version': 1,
                  'protocol_id': 'contextual-forest-expansion-v1',
                  'job_id': (
                    f'{arm}-{train_seed}-{corruption_seed}-{dataset}'),
                  'arm': arm,
                  'train_seed': train_seed,
                  'corruption_seed': corruption_seed,
                  'dataset': dataset,
                  'dataset_revision': 'a' * 40,
                  'mask_rate': mask_rate,
                  'candidate_k': 64,
                  'rank': 0,
                  'batch_index': document_index,
                  'example_index': 0,
                  'document_id': f'{dataset}:{document_index}',
                  'document_index': document_index,
                  'document_sha256': _digest(
                    f'{dataset}:{document_index}'),
                  'chunk_index': chunk_index,
                  'nll_sum': per_token * masked_tokens,
                  'masked_tokens': masked_tokens,
                  'candidate_hits': masked_tokens - 1,
                  'retained_mass_sum': 0.8 * masked_tokens,
                  'pairing_digest_sha256': pairing,
                })
  return rows


class HierarchicalDocumentEvaluationTest(unittest.TestCase):

  def test_averages_corruptions_and_chunks_before_hierarchical_bootstrap(self):
    result = aggregate_records(
      _records(),
      baseline_arm='static_static',
      treatment_arm='dynamic_dynamic',
      protocol_id='contextual-forest-expansion-v1',
      suite_name='test-suite',
      comparison_name='contextual-vs-static',
      num_resamples=500,
      rng_seed=77,
      confidence_level=0.95,
      timestamp_utc='2026-08-30T00:00:00+00:00')

    bootstrap = result['by_candidate_k']['64']
    self.assertAlmostEqual(bootstrap['pooled']['mean_improvement'], 0.5)
    self.assertAlmostEqual(bootstrap['pooled']['ci_lower'], 0.5)
    self.assertAlmostEqual(bootstrap['pooled']['ci_upper'], 0.5)
    self.assertEqual(bootstrap['num_train_seeds'], 2)
    self.assertEqual(bootstrap['num_strata'], 4)
    self.assertIn('average corruption replications', bootstrap['nesting'][0])
    self.assertIn('no diffusion ELBO', result['scope_note'])

  def test_rejects_pairing_digest_mismatch_across_training_seeds(self):
    rows = _records()
    rows[0]['pairing_digest_sha256'] = 'f' * 64
    with self.assertRaisesRegex(ValueError, 'pairing digests differ'):
      aggregate_records(
        rows,
        baseline_arm='static_static',
        treatment_arm='dynamic_dynamic',
        protocol_id='contextual-forest-expansion-v1',
        suite_name='test-suite',
        comparison_name='contextual-vs-static',
        num_resamples=10)

  def test_rejects_incomplete_document_factorial_cell(self):
    rows = _records()
    rows = [
      row for row in rows
      if not (
        row['arm'] == 'dynamic_dynamic'
        and row['train_seed'] == 2
        and row['corruption_seed'] == 202
        and row['dataset'] == 'wiki-pinned'
        and row['mask_rate'] == 0.5
        and row['document_index'] == 20)]
    with self.assertRaisesRegex(ValueError, 'window identities differ'):
      aggregate_records(
        rows,
        baseline_arm='static_static',
        treatment_arm='dynamic_dynamic',
        protocol_id='contextual-forest-expansion-v1',
        suite_name='test-suite',
        comparison_name='contextual-vs-static',
        num_resamples=10)

  def test_record_schema_rejects_zero_masked_tokens(self):
    row = _records()[0]
    row['masked_tokens'] = 0
    with self.assertRaisesRegex(ValueError, 'must be positive'):
      validate_record(row)


if __name__ == '__main__':
  unittest.main()
