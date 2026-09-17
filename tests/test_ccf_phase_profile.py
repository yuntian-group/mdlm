import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch
import structured_utils as utils
import structured_objective as objective
from scripts.profile_ccf_full_sample import PhaseTimer, install_timers
from scripts.audit_ccf_sampling_v4 import experiment as level_experiment
from scripts.verify_ccf_sampling_optimization import problem, compare_draws
from scripts.audit_ccf_sampling_v3 import experiment


class PhaseProfileTest(unittest.TestCase):
  def test_level_timer_installation_preserves_tokens_and_rng(self):
    output, logits, active = problem(torch.device('cpu'), k=32)
    noop = lambda *args, **kwargs: None
    head = SimpleNamespace(forward=noop, _candidate_lattice=noop,
      _selected_edges=noop, _node_candidate_factors=noop,
      edge_proposer=SimpleNamespace(forward=noop))
    model = SimpleNamespace(structured_head=head, _ddpm_update=noop,
      _structured_backbone_output=noop)
    timer = PhaseTimer(cuda=False)
    with level_experiment('level_draws'):
      def baseline(generator):
        return objective.sample_structured_tokens(output, logits, active, generator=generator)
      def profiled(generator):
        with ExitStack() as stack:
          install_timers(stack, timer, model, 'level_draws')
          return baseline(generator)
      compare_draws(baseline, profiled, torch.device('cpu'), 91001)
    self.assertGreater(timer.stats['batched_child_conditional_arithmetic']['calls'], 0)
    self.assertGreater(timer.stats['batched_softmax_divide_argmax']['calls'], 0)

  def test_v3_batched_profile_preserves_sample_and_rng(self):
    output, logits, active = problem(torch.device('cpu'), k=32)
    timer = PhaseTimer(cuda=False)
    with experiment('unchecked_batched_roots_kruskal'):
      def baseline(generator):
        return objective.sample_structured_tokens(output, logits, active, generator=generator)
      def profiled(generator):
        with mock.patch.object(utils, '_sample_rows', timer.wrap('draw', utils._sample_rows)), \
             mock.patch.object(utils, '_batched_low_rank_message', timer.wrap('batched', utils._batched_low_rank_message)):
          return timer.wrap('sample', baseline)(generator)
      compare_draws(baseline, profiled, torch.device('cpu'), 91001)
    self.assertEqual(timer.stats['draw']['calls'], 18)
    self.assertGreater(timer.stats['batched']['calls'], 0)

  def test_nested_exclusive_time_sums_to_parent(self):
    timer = PhaseTimer(cuda=False)
    child = timer.wrap('child', lambda: time.sleep(.002))
    parent = timer.wrap('parent', lambda: (child(), child()))
    parent()
    self.assertEqual(timer.stats['child']['calls'], 2)
    self.assertAlmostEqual(sum(row['host_exclusive_seconds'] for row in timer.stats.values()),
                           timer.stats['parent']['host_inclusive_seconds'])

  def test_cpu_instrumentation_preserves_sample_and_rng(self):
    output, logits, active = problem(torch.device('cpu'), k=32)
    timer = PhaseTimer(cuda=False)
    def baseline(generator):
      return objective.sample_structured_tokens(output, logits, active, generator=generator)
    def profiled(generator):
      with mock.patch.object(utils, '_sample_rows', timer.wrap('draw', utils._sample_rows)):
        return timer.wrap('sample', baseline)(generator)
    compare_draws(baseline, profiled, torch.device('cpu'), 91001)
    self.assertEqual(timer.stats['draw']['calls'], 18)


if __name__ == '__main__':
  unittest.main()
