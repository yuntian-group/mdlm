import unittest

from evaluation.hierarchical_causal_statistics import (
  ContrastTerm,
  aggregate_candidate_support,
  aggregate_causal_contrast,
  aggregate_topology_permutation_diagnostic,
)


class HierarchicalCausalStatisticsTest(unittest.TestCase):

  def _records(self):
    arm_nll = {
      'dynamic_dynamic': 3.0,
      'fixed_dynamic': 3.5,
      'dynamic_fixed': 3.8,
      'static_static': 4.0,
    }
    rows = []
    for arm, joint_nll in arm_nll.items():
      for train_seed in (1, 2):
        for corruption_seed in (301, 302):
          for document_index in (10, 11):
            masked_tokens = 10
            rows.append({
              'schema_version': 2,
              'arm': arm,
              'train_seed': train_seed,
              'corruption_seed': corruption_seed,
              'dataset': 'toy',
              'dataset_revision': 'a' * 40,
              'mask_rate': 0.75,
              'candidate_k': 128,
              'document_id': f'toy:{document_index}',
              'document_index': document_index,
              'document_sha256': f'{document_index:064x}',
              'chunk_index': 0,
              'pairing_digest_sha256': 'b' * 64,
              'masked_tokens': masked_tokens,
              'nll_sum': joint_nll * masked_tokens,
              'structured_marginal_nll_sum': 3.4 * masked_tokens,
              'factorized_backbone_nll_sum': 5.0 * masked_tokens,
              'parameter_matched_no_edge_nll_sum': 5.0 * masked_tokens,
              'matched_permuted_topology_nll_sum': 3.7 * masked_tokens,
              'selected_edges': 10,
              'permuted_changed_edges': 9,
              'candidate_support': [
                {'candidate_k': 32, 'candidate_hits': 5,
                 'retained_mass_sum': 6.0},
                {'candidate_k': 64, 'candidate_hits': 7,
                 'retained_mass_sum': 8.0},
                {'candidate_k': 128, 'candidate_hits': 9,
                 'retained_mass_sum': 9.5},
                {'candidate_k': 256, 'candidate_hits': 10,
                 'retained_mass_sum': 9.9},
              ],
            })
    return rows

  def test_adapter_seed_is_top_level_for_within_adapter_contrast(self):
    result = aggregate_causal_contrast(
      self._records(),
      name='factorized_vs_joint',
      terms=(
        ContrastTerm(
          'dynamic_dynamic', 'factorized_backbone_nll_sum', 1.0),
        ContrastTerm('dynamic_dynamic', 'nll_sum', -1.0)),
      num_resamples=100,
      rng_seed=7)
    analysis = result['analysis']
    self.assertEqual(analysis['top_level_resampling_unit'],
                     'adapter_training_seed')
    self.assertEqual(analysis['num_adapter_seeds'], 2)
    self.assertAlmostEqual(analysis['pooled']['estimate'], 2.0)

  def test_four_arm_factorial_interaction(self):
    result = aggregate_causal_contrast(
      self._records(),
      name='interaction',
      terms=(
        ContrastTerm('fixed_dynamic', 'nll_sum', 1.0),
        ContrastTerm('dynamic_dynamic', 'nll_sum', -1.0),
        ContrastTerm('static_static', 'nll_sum', -1.0),
        ContrastTerm('dynamic_fixed', 'nll_sum', 1.0)),
      num_resamples=100,
      rng_seed=9)
    self.assertAlmostEqual(result['analysis']['pooled']['estimate'], 0.3)

  def test_candidate_support_aggregates_all_four_k_values(self):
    result = aggregate_candidate_support(
      self._records(), arm='dynamic_dynamic',
      num_resamples=100, rng_seed=11)
    self.assertEqual(result['support_candidate_ks'], [32, 64, 128, 256])
    self.assertAlmostEqual(
      result['by_candidate_k']['64']['candidate_recall']['pooled']['estimate'],
      0.7)

  def test_topology_permutation_diagnostic_is_gated(self):
    result = aggregate_topology_permutation_diagnostic(
      self._records(), arm='dynamic_dynamic',
      minimum_pooled_changed_edge_fraction=0.85,
      minimum_condition_changed_edge_fraction=0.85)
    self.assertAlmostEqual(
      result['pooled']['changed_edge_fraction'], 0.9)
    self.assertTrue(result['gate']['passed'])

  def test_topology_permutation_diagnostic_rejects_noop(self):
    records = self._records()
    for row in records:
      row['permuted_changed_edges'] = 0
    result = aggregate_topology_permutation_diagnostic(
      records, arm='dynamic_dynamic',
      minimum_pooled_changed_edge_fraction=0.85,
      minimum_condition_changed_edge_fraction=0.85)
    self.assertFalse(result['gate']['passed'])


if __name__ == '__main__':
  unittest.main()
