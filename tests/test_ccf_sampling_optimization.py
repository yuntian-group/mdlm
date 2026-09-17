"""Exact regression tests for conservative CCF sampling speed fixes."""
from unittest import mock
import unittest

import torch

import structured_objective
import structured_utils
from evaluation import generation_harness
from scripts.verify_ccf_sampling_optimization import load_baseline, problem, verify
from tests.test_generation_harness import FakeDiffusionModel


class SamplingOptimizationTest(unittest.TestCase):
  def test_cpu_tokens_and_rng_match_pinned_prefixed_code(self):
    result = verify(torch.device('cpu'))
    self.assertGreaterEqual(result['equivalence_cases'], 50)

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
  def test_cuda_tokens_and_rng_match_pinned_prefixed_code(self):
    self.assertTrue(verify(torch.device('cuda'))['rng_identical'])

  def test_no_unused_inference_and_single_topology_build(self):
    output, logits, active = problem(torch.device('cpu'))
    with mock.patch.object(structured_objective, 'infer_structured_distribution',
                           side_effect=AssertionError('Unused inference pass')), \
         mock.patch.object(structured_utils, '_build_topology',
                           wraps=structured_utils._build_topology) as build:
      structured_objective.sample_structured_tokens(output, logits, active)
    self.assertEqual(build.call_count, 1)

  def test_harness_replay_preserves_tokens_nfe_rng_and_evidence(self):
    old = load_baseline('evaluation/generation_harness.py')
    for observed in (False, True):
      tokens = torch.full((2, 8), 99)
      if observed:
        tokens[:, 0] = 3
      results = []
      for module in (old, generation_harness):
        model = FakeDiffusionModel()
        torch.manual_seed(191)
        results.append((*module.sample_from_initial_state(
          model, tokens, nfe_budget=1001), torch.get_rng_state()))
      self.assertTrue(torch.equal(results[0][0], results[1][0]))
      self.assertEqual(results[0][1], results[1][1])
      self.assertTrue(torch.equal(results[0][2], results[1][2]))

  def test_evidence_checks_not_removed_for_conditional_generation(self):
    model = FakeDiffusionModel()
    model._ddpm_update = lambda x, t, dt: torch.zeros_like(x)
    with self.assertRaisesRegex(AssertionError, 'modified observed'):
      generation_harness.sample_from_initial_state(
        model, torch.tensor([[3, 99]]), nfe_budget=3)

  def test_host_orientation_matches_noncanonical_edges(self):
    edges = torch.tensor([[[1, 0], [1, 2], [3, 2], [99, -1]]])
    topology = structured_utils._build_topology(
      edges, torch.tensor([[True, True, True, False]]), 5, None, None)[0]
    self.assertEqual(topology.edge_left, (1, 1, 3, 99))


if __name__ == '__main__':
  unittest.main()
