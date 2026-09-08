import math
import dataclasses
import unittest

import torch

from evaluation.fixed_group_sampling import generate_fixed_groups, make_reveal_order
from models.structured_decoder import StructuredDecoderOutput


def _output(logits, active, edge_index=None, edge_mask=None, left=None, right=None):
  batch, length, vocab = logits.shape
  if edge_index is None:
    edge_index = torch.empty(batch, 0, 2, dtype=torch.long)
    edge_mask = torch.empty(batch, 0, dtype=torch.bool)
  edges = edge_index.shape[1]
  if left is None:
    left = right = torch.ones(batch, edges, vocab, 1)
  return StructuredDecoderOutput(
    candidate_ids=torch.arange(vocab).expand(batch, length, vocab),
    unary_log_potentials=torch.cat((logits, logits.new_full((batch, length, 1), -torch.inf)), -1),
    candidate_state_mask=torch.cat((torch.ones_like(logits, dtype=torch.bool),
                                    torch.zeros(batch, length, 1, dtype=torch.bool)), -1),
    retained_mass=torch.ones(batch, length), residual_log_mass=logits.new_full((batch, length), -torch.inf),
    proposal_edge_index=torch.empty(batch, 0, 2, dtype=torch.long),
    proposal_edge_mask=torch.empty(batch, 0, dtype=torch.bool),
    proposal_scores=torch.empty(batch, 0), anchor_logits=torch.zeros(batch, length, 1),
    anchor_indices=torch.zeros(batch, 1, dtype=torch.long), slot_logits=torch.zeros(batch, length, 1),
    edge_index=edge_index, edge_mask=edge_mask, edge_scores=torch.zeros(batch, edges),
    pair_left_factors=left, pair_right_factors=right,
    topology_mode='fixed', factor_mode='fixed', independent_mode=False)


class PairOracle:
  """AB/BA oracle; after one reveal its unary predicts the other token."""

  def __init__(self):
    self.calls = []

  def __call__(self, tokens, sigma, active):
    self.calls.append((tokens.clone(), sigma.clone(), active.clone()))
    batch = tokens.shape[0]
    logits = torch.zeros(batch, 2, 2)
    for row in range(batch):
      if active[row].sum() == 1:
        observed = tokens[row, ~active[row]].item()
        position = active[row].nonzero().item()
        logits[row, position, observed] = -30
    edges = torch.tensor([[[0, 1]]]).expand(batch, -1, -1).clone()
    edge_mask = active.all(-1, keepdim=True)
    eps = 1e-8
    left = torch.tensor([[1.0, eps], [eps, 1.0]]).expand(batch, 1, 2, 2).clone()
    right = torch.tensor([[eps, 1.0], [1.0, eps]]).expand(batch, 1, 2, 2).clone()
    return _output(logits, active, edges, edge_mask, left, right), logits


class UniformOracle:
  def __init__(self, vocab=7):
    self.vocab = vocab
    self.calls = []

  def __call__(self, tokens, sigma, active):
    self.calls.append((tokens.clone(), sigma.clone(), active.clone()))
    logits = torch.zeros(*tokens.shape, self.vocab)
    return _output(logits, active), logits


class FixedGroupSamplingTest(unittest.TestCase):

  def test_pair_joint_generation_preserves_both_modes_but_marginals_break_pairs(self):
    count = 1500
    tokens = torch.full((count, 2), 2, dtype=torch.long)
    order = torch.tensor([[0, 1]]).expand(count, -1)
    results = {}
    for mode in ('joint', 'marginal', 'backbone'):
      model = PairOracle()
      result = generate_fixed_groups(
        model, tokens, mask_index=2, group_size=2, mode=mode, reveal_order=order,
        sampling_generator=torch.Generator().manual_seed(117))
      results[mode] = result
      self.assertEqual(result.batch_calls, 1)
      torch.testing.assert_close(result.nfe, torch.ones(count, dtype=torch.long))
      self.assertTrue(bool(result.steps[0].jointly_committed_edge_mask.all()))
    joint = results['joint'].final_tokens
    self.assertTrue(bool(joint[:, 0].ne(joint[:, 1]).all()))
    self.assertAlmostEqual(float(joint[:, 0].float().mean()), 0.5, delta=0.05)
    for mode in ('marginal', 'backbone'):
      independent = results[mode].final_tokens
      invalid = independent[:, 0].eq(independent[:, 1]).float().mean()
      self.assertAlmostEqual(float(invalid), 0.5, delta=0.05)

  def test_sequential_oracle_uses_newly_observed_token_and_reveals_one_at_a_time(self):
    count = 80
    tokens = torch.full((count, 2), 2, dtype=torch.long)
    for mode in ('joint', 'marginal', 'backbone'):
      model = PairOracle()
      result = generate_fixed_groups(
        model, tokens, mask_index=2, group_size=1, mode=mode,
        reveal_seeds=list(range(count)), sampling_generator=torch.Generator().manual_seed(9))
      self.assertEqual(result.batch_calls, 2)
      self.assertTrue(bool(result.final_tokens[:, 0].ne(result.final_tokens[:, 1]).all()))
      self.assertTrue(bool(result.nfe.eq(2).all()))
      for step in result.steps:
        self.assertTrue(bool(step.revealed_mask.sum(-1).eq(1).all()))
        self.assertFalse(bool(step.jointly_committed_edge_mask.any()))
      torch.testing.assert_close(model.calls[1][0], result.steps[0].tokens_after)
      self.assertTrue(bool(model.calls[1][2].sum(-1).eq(1).all()))

  def test_quotas_nfe_mixed_rows_and_original_context_for_every_group_size(self):
    initial = torch.tensor([[7] * 17, [7] * 5 + [3] * 12, [4] * 17])
    for group_size in (1, 2, 4, 8, 16):
      model = UniformOracle()
      result = generate_fixed_groups(
        model, initial, mask_index=7, group_size=group_size, mode='backbone',
        reveal_seeds=[11, 12, 13])
      expected = torch.tensor([math.ceil(17 / group_size), math.ceil(5 / group_size), 0])
      torch.testing.assert_close(result.nfe, expected)
      self.assertEqual(result.batch_calls, int(expected.max()))
      self.assertEqual(len(model.calls), result.batch_calls)
      observed = initial.ne(7)
      torch.testing.assert_close(result.final_tokens[observed], initial[observed])
      self.assertFalse(bool(result.final_tokens.eq(7).any()))
      for step, (_, sigma, active) in zip(result.steps, model.calls):
        self.assertTrue(bool(active.any(-1).all()))
        self.assertFalse(bool(step.row_indices.eq(2).any()))
        torch.testing.assert_close(step.revealed_mask.sum(-1), active.sum(-1).clamp_max(group_size))
        torch.testing.assert_close(step.active_after, active & ~step.revealed_mask)
        torch.testing.assert_close(step.tokens_after[~step.revealed_mask], step.tokens_before[~step.revealed_mask])
        torch.testing.assert_close(step.masked_fraction, active.float().mean(-1))
        expected_sigma = -torch.log1p(-step.masked_fraction.clamp_max(0.999))
        torch.testing.assert_close(sigma, expected_sigma)
        torch.testing.assert_close(step.diffusion_time, (step.masked_fraction / 0.999).clamp_max(1))
      self.assertEqual(result.steps[0].diffusion_time[0].item(), 1)

  def test_all_modes_use_same_reveal_order_despite_different_identity_rng(self):
    initial = torch.tensor([[7, 0, 7, 7, 2, 7], [1, 7, 7, 4, 7, 7]])
    order = make_reveal_order(6, [35, 47])
    traces = []
    for index, mode in enumerate(('joint', 'marginal', 'backbone')):
      result = generate_fixed_groups(
        UniformOracle(), initial, mask_index=7, group_size=2, mode=mode,
        reveal_order=order, sampling_generator=torch.Generator().manual_seed(index + 99))
      traces.append(result.steps)
      torch.testing.assert_close(result.reveal_order, order)
    for steps in traces[1:]:
      for reference, actual in zip(traces[0], steps):
        torch.testing.assert_close(actual.row_indices, reference.row_indices)
        torch.testing.assert_close(actual.revealed_mask, reference.revealed_mask)
        torch.testing.assert_close(actual.sigma, reference.sigma)

  def test_reveal_seeding_does_not_consume_global_rng_and_is_prompt_local(self):
    before = torch.get_rng_state().clone()
    first = make_reveal_order(17, [33, 44])
    second = make_reveal_order(17, [44, 33])
    torch.testing.assert_close(torch.get_rng_state(), before)
    torch.testing.assert_close(first[0], second[1])
    torch.testing.assert_close(first[1], second[0])

  def test_marginal_draws_expand_residual_probabilities_exactly(self):
    def model(tokens, sigma, active):
      batch = tokens.shape[0]
      logits = torch.tensor([0.25, 0.5, 0.25]).log().expand(batch, 1, 3)
      output = _output(logits, active)
      output = dataclasses.replace(
        output, candidate_ids=torch.zeros(batch, 1, 1, dtype=torch.long),
        unary_log_potentials=torch.tensor([0.25, 0.75]).log().expand(batch, 1, 2),
        candidate_state_mask=torch.ones(batch, 1, 2, dtype=torch.bool),
        pair_left_factors=torch.empty(batch, 0, 1, 1), pair_right_factors=torch.empty(batch, 0, 1, 1))
      return output, logits
    count = 2000
    result = generate_fixed_groups(
      model, torch.full((count, 1), 3, dtype=torch.long), mask_index=3,
      group_size=1, mode='marginal', reveal_order=torch.zeros(count, 1, dtype=torch.long),
      sampling_generator=torch.Generator().manual_seed(821))
    frequency = torch.bincount(result.final_tokens.flatten(), minlength=3).float() / count
    torch.testing.assert_close(frequency, torch.tensor([0.25, 0.5, 0.25]), atol=0.04, rtol=0)

  def test_empty_residual_with_explicit_absorbing_vocabulary_slot_is_safe(self):
    def model(tokens, sigma, active):
      logits = torch.tensor([[[0.0, 0.0, -torch.inf]]]).expand(tokens.shape[0], 1, 3)
      output = _output(logits[..., :2], active)
      # Mirrors top-K covering all finite clean tokens while the vocabulary
      # also contains an excluded absorbing mask: residual allowed, mass zero.
      output = dataclasses.replace(output, candidate_state_mask=torch.ones(tokens.shape[0], 1, 3, dtype=torch.bool))
      return output, logits
    for mode in ('joint', 'marginal', 'backbone'):
      result = generate_fixed_groups(model, torch.tensor([[2], [2]]), mask_index=2,
                                     group_size=1, mode=mode, reveal_seeds=[4, 5])
      self.assertTrue(bool(((result.final_tokens == 0) | (result.final_tokens == 1)).all()))

  def test_no_masks_means_no_model_call_and_no_input_mutation(self):
    initial = torch.tensor([[1, 2, 3]])
    model = UniformOracle()
    result = generate_fixed_groups(model, initial, mask_index=7, group_size=4,
                                   mode='joint', reveal_seeds=[5])
    self.assertEqual(model.calls, [])
    self.assertEqual(result.batch_calls, 0)
    self.assertEqual(len(result.steps), 0)
    self.assertEqual(result.nfe.item(), 0)
    torch.testing.assert_close(result.final_tokens, initial)
    result.final_tokens[0, 0] = 6
    self.assertEqual(initial[0, 0].item(), 1)

  def test_validation_rejects_bad_order_group_and_absorbing_output(self):
    initial = torch.tensor([[7, 7]])
    for extra in ({'reveal_order': torch.tensor([[0, 0]])},
                  {'reveal_seeds': [1, 2]}, {}):
      with self.assertRaises(ValueError):
        generate_fixed_groups(UniformOracle(), initial, mask_index=7, group_size=2, mode='joint', **extra)
    with self.assertRaises(ValueError):
      generate_fixed_groups(UniformOracle(), initial, mask_index=7, group_size=3, mode='joint', reveal_seeds=[1])
    with self.assertRaisesRegex(ValueError, 'absorbing mask'):
      generate_fixed_groups(UniformOracle(vocab=8), initial, mask_index=7, group_size=2, mode='joint', reveal_seeds=[1])


if __name__ == '__main__':
  unittest.main()
