"""Opt-in joint FiLM MLP; defaults must remain checkpoint-compatible."""

import itertools
import unittest

import torch

from models.structured_decoder import ContextualCouplingForestHead
from structured_objective import infer_structured_distribution
from structured_training import structured_denoising_loss


def head(**overrides):
  options = dict(hidden_size=8, vocab_size=11, top_k=5, rank=4,
                 time_embed_dim=8, topology_dim=8, num_anchor_slots=2,
                 contextual_neighbors=1, component_size_cap=3)
  options.update(overrides)
  return ContextualCouplingForestHead(**options)


class FactorConditionerTest(unittest.TestCase):

  def test_disabled_default_preserves_seeded_state(self):
    torch.manual_seed(41)
    original = head()
    torch.manual_seed(41)
    disabled = head(factor_conditioner_hidden_dim=0)
    self.assertEqual(original.state_dict().keys(), disabled.state_dict().keys())
    for key, value in original.state_dict().items():
      torch.testing.assert_close(value, disabled.state_dict()[key], atol=0, rtol=0)
    self.assertFalse(hasattr(disabled, 'factor_conditioner'))

  def test_invalid_widths(self):
    for width in (-1, True, 1.5, '128', None):
      with self.subTest(width=width), self.assertRaises(ValueError):
        head(factor_conditioner_hidden_dim=width)

  def test_production_parameter_delta(self):
    settings = dict(hidden_size=768, vocab_size=50258, top_k=128, rank=16,
                    time_embed_dim=64, topology_dim=128, num_anchor_slots=16)
    original = head(**settings)
    mlp = head(**settings, factor_conditioner_hidden_dim=128)
    self.assertEqual(mlp.parameter_count - original.parameter_count, 84096)
    self.assertFalse(hasattr(mlp, 'factor_hidden_projection'))
    self.assertEqual(mlp.factor_conditioner[0].in_features, 832)
    self.assertEqual(mlp.factor_conditioner[-1].out_features, 32)

  def test_hidden_time_and_conditioner_layers_receive_gradients(self):
    for topology, endpoints in itertools.product(('fixed', 'dynamic'),
                                                 ('shared', 'separate')):
      with self.subTest(topology=topology, endpoints=endpoints):
        torch.manual_seed(7)
        model = head(topology_mode=topology, factor_embedding_mode=endpoints,
                     factor_conditioner_hidden_dim=16)
        hidden = torch.randn(2, 4, 8, requires_grad=True)
        time = torch.tensor([0.2, 0.7], requires_grad=True)
        logits = torch.randn(2, 4, 11)
        active = torch.ones(2, 4, dtype=torch.bool)
        output = model(hidden, logits, time, active)
        target = output.candidate_ids[..., 0]
        loss = structured_denoising_loss(output, logits, target, active).loss
        loss.backward()
        for gradient in (hidden.grad, time.grad):
          self.assertIsNotNone(gradient)
          self.assertTrue(torch.isfinite(gradient).all())
          self.assertGreater(float(gradient.abs().sum()), 0)
        for name, parameter in model.named_parameters():
          if 'factor_conditioner' in name or 'token_factor_embedding' in name:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

  def test_fixed_factors_bypass_mlp_and_independent_mode_is_neutral(self):
    model = head(factor_conditioner_hidden_dim=16, factor_mode='fixed')
    hidden, logits, time = torch.randn(1, 4, 8), torch.randn(1, 4, 11), torch.rand(1)
    output = model(hidden, logits, time)
    altered = model(hidden * 10, logits, time + 1,
                    fixed_edge_index=output.edge_index[0][output.edge_mask[0]])
    torch.testing.assert_close(output.pair_left_factors, altered.pair_left_factors)
    torch.testing.assert_close(output.pair_right_factors, altered.pair_right_factors)
    neutral = model(hidden, logits, time, independent_mode=True).materialize_pair_factors()
    torch.testing.assert_close(neutral, torch.ones_like(neutral))

  def test_dense_low_rank_agreement_with_mlp(self):
    model = head(factor_conditioner_hidden_dim=16)
    active = torch.ones(1, 4, dtype=torch.bool)
    output = model(torch.randn(1, 4, 8), torch.randn(1, 4, 11), torch.rand(1), active)
    dense = infer_structured_distribution(output, active, backend='dense')
    low_rank = infer_structured_distribution(output, active, backend='low_rank')
    torch.testing.assert_close(dense.marginals.node_marginals,
                               low_rank.marginals.node_marginals, atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
  unittest.main()
