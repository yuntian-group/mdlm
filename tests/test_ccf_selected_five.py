from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock
import torch

from scripts.evaluate_ccf_selected_five import matrix, generation_args, gated_pilot
import structured_utils as utils


class SelectedFiveTest(unittest.TestCase):
  def test_all_1k_matrix(self):
    cells = matrix('all_1k')
    self.assertEqual(len(cells), 8)
    self.assertEqual([c['family'] for c in cells], ['B'] * 4 + ['C'] * 2 + ['D'] * 2)
    self.assertEqual({c['step'] for c in cells}, {1000})
    self.assertEqual(len({c['checkpoint'] for c in cells}), 8)
    self.assertEqual([(c['rank'], c['arm']) for c in cells[4:]],
                     [(8, 'fixed_dynamic'), (8, 'dynamic_dynamic'),
                      (16, 'fixed_dynamic'), (16, 'dynamic_dynamic')])
    for cell in cells:
      self.assertTrue(cell['checkpoint'].endswith('0-1000.ckpt'))
      args = generation_args(cell, Path('/tmp/test'), 'a', 'm')
      for flag, value in (('--num-samples', '5'), ('--nfe-budgets', '1001'),
                          ('--batch-size', '1'), ('--base-seed', '91001')):
        self.assertEqual(args[args.index(flag) + 1], value)

  def test_original_trend_adds_only_5k_and_6k(self):
    cells = matrix('original_5k6k')
    self.assertEqual(len(cells), 8)
    self.assertEqual([c['step'] for c in cells], [5000] * 4 + [6000] * 4)
    self.assertEqual({c['arm'] for c in cells},
                     {'static_static', 'fixed_dynamic', 'dynamic_fixed', 'dynamic_dynamic'})
    for cell in cells:
      self.assertEqual((cell['family'], cell['rank'], cell['embedding']), ('B', 16, 'shared'))
      self.assertIn('3k_to6k_run-continue-6k', cell['checkpoint'])
      self.assertTrue(cell['checkpoint'].endswith(f"0-{cell['step']}.ckpt"))
      args = generation_args(cell, Path('/tmp/test'), 'a', 'm')
      self.assertEqual(args[args.index('--num-samples') + 1], '5')
      self.assertEqual(args[args.index('--nfe-budgets') + 1], '1001')

  def test_matrix_and_unchanged_arguments(self):
    cells = matrix()
    self.assertEqual(len(cells), 13)
    self.assertEqual([c['family'] for c in cells], ['A'] + ['B'] * 4 + ['C'] * 4 + ['D'] * 4)
    self.assertEqual({c['step'] for c in cells if c['family'] == 'B'}, {7000})
    self.assertEqual({c['step'] for c in cells if c['family'] == 'C'}, {2000, 6000})
    self.assertEqual({c['step'] for c in cells if c['family'] == 'D'}, {2000, 4000})
    for cell in cells:
      args = generation_args(cell, Path('/tmp/test'), 'a', 'm')
      for flag, value in (('--num-samples', '5'), ('--nfe-budgets', '1001'),
                          ('--batch-size', '1'), ('--base-seed', '91001')):
        self.assertEqual(args[args.index(flag) + 1], value)

  def exercise_gate(self, mismatch):
    old_sampler = utils.sample_forest_low_rank
    calls = []
    def sample(*args, **kwargs):
      fast = utils.sample_forest_low_rank is not old_sampler
      calls.append(fast)
      return [{'sample_token_ids': [int(fast and mismatch)]}], {'measured_nfe': 1000}
    pilot = SimpleNamespace(run_sampling_group=sample)
    def main(args):
      for _ in range(5):
        pilot.run_sampling_group(None)
      return 0
    pilot.main = main
    with TemporaryDirectory() as directory, \
         mock.patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([1])), \
         mock.patch.object(torch.cuda, 'get_device_name', return_value='mock'):
      if mismatch:
        with self.assertRaises(AssertionError):
          gated_pilot(pilot, matrix()[1], Path(directory), [])
        self.assertEqual(calls, [False, True])
      else:
        self.assertEqual(gated_pilot(pilot, matrix()[1], Path(directory), []), 0)
        self.assertEqual(calls, [False] + [True] * 5)
    self.assertIs(utils.sample_forest_low_rank, old_sampler)
    self.assertIs(pilot.run_sampling_group, sample)

  def test_gate_reuses_verified_first_sample(self):
    self.exercise_gate(False)

  def test_failed_gate_stops_before_remaining_samples(self):
    self.exercise_gate(True)
