"""No production code changed: exact checks for experimental top-down batching."""
import unittest
from unittest import mock

import torch

import structured_utils as utils
import structured_objective as objective
from scripts.audit_ccf_sampling_v4 import MODES, experiment, grouped_pair_rows, verify_buffered_noise
from scripts.verify_ccf_sampling_optimization import verify, problem, compare_draws
from scripts import verify_ccf_sampling_v3_gpu as runner
from scripts.audit_ccf_sampling_v4 import BEST_V3


class LevelSamplingTests(unittest.TestCase):
  def test_buffered_noise_values_and_rng(self):
    self.assertEqual(verify_buffered_noise(torch.device('cpu')), 32)
    self.assertEqual(verify_buffered_noise(torch.device('cpu'), aligned=True), 32)

  def test_complete_benchmark_loop_does_not_shadow_draw_function(self):
    # CPU execution of the actual GPU-check runner, with synchronization mocked.
    with mock.patch.object(torch.cuda, 'synchronize'):
      result = runner.checks('level_draws', torch.device('cpu'), BEST_V3)
    self.assertEqual(result['native_draw_cases'], 96)
    self.assertEqual(len(result['isolated_benchmarks']), 3)

  def test_pinned_reference_cases_and_patch_restoration(self):
    original = utils.sample_forest_low_rank
    for mode in MODES:
      with experiment(mode):
        self.assertEqual(verify(torch.device('cpu'))['equivalence_cases'], 54)
      self.assertIs(utils.sample_forest_low_rank, original)

  def test_grouped_conditional_rows_bitwise_equal(self):
    torch.manual_seed(441)
    source = torch.randn(7, 32, 8)
    target = torch.randn(7, 32, 8)
    states = torch.randint(0, 33, (7, 5))
    states[0, 0] = 32
    old = torch.stack([utils._low_rank_pair_rows(states[i], source[i], target[i]) for i in range(7)])
    self.assertTrue(torch.equal(old, grouped_pair_rows(states, source, target)))

  def test_l1024_seed_and_rng_equality(self):
    for active_kind in ('all', 'mixed', 'none'):
      output, logits, active = problem(torch.device('cpu'), k=128, length=1024, active_kind=active_kind)
      def baseline(g):
        return objective.sample_structured_tokens(output, logits, active, generator=g)
      for mode in MODES:
        def candidate(g):
          with experiment(mode):
            return baseline(g)
        for seed in (91001, 91002):
          compare_draws(baseline, candidate, torch.device('cpu'), seed)


if __name__ == '__main__':
  torch.set_num_threads(1)
  unittest.main()
