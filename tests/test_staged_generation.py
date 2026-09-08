"""Target-free staged heads adapt to fixed-group sampling without side effects."""

import dataclasses
import random
import unittest
from unittest import mock

import torch

from evaluation.fixed_group_sampling import generate_fixed_groups, make_reveal_order
from evaluation.staged_generation import StagedGenerationAdapter
from models.contextual_unary import ContextualUnaryAdapter
from models.directional_forest import DirectionalCouplingForestHead
from models.structured_decoder import ContextualCouplingForestHead, StructuredDecoderOutput
from structured_objective import full_vocabulary_marginals, infer_structured_distribution


VARIANTS = ('shared', 'directional', 'unary')
HIDDEN_SIZE = 6
VOCAB_SIZE = 9
MASK_INDEX = VOCAB_SIZE - 1


class RecordingBackbone(torch.nn.Module):
  """The production encoder interface, with observable dropout and buffers."""

  def __init__(self, mask_index=MASK_INDEX, output_dtype=torch.float32):
    super().__init__()
    self.mask_index = mask_index
    self.output_dtype = output_dtype
    self.backbone = torch.nn.ModuleDict({
      'embedding': torch.nn.Embedding(max(VOCAB_SIZE, mask_index + 1), HIDDEN_SIZE),
      'normalization': torch.nn.BatchNorm1d(HIDDEN_SIZE),
      'dropout': torch.nn.Dropout(0.6),
      'output': torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE),
    })
    self.calls = []
    self.failure_call = None

  def _structured_head_output(self, *args, **kwargs):
    raise AssertionError('generation must use the supplied staged head')

  def _structured_backbone_output(self, tokens, conditioning, force_no_grad=False):
    record = {
      'tokens': tokens.clone(), 'conditioning': conditioning.clone(),
      'force_no_grad': force_no_grad, 'grad_enabled': torch.is_grad_enabled(),
      'training': tuple(module.training for module in self.modules()),
    }
    self.calls.append(record)
    if len(self.calls) == self.failure_call:
      raise RuntimeError('injected encoder failure')
    hidden = self.backbone['embedding'](tokens)
    hidden = hidden + hidden.mean(1, keepdim=True) + conditioning[..., None]
    hidden = self.backbone['normalization'](hidden.transpose(1, 2)).transpose(1, 2)
    hidden = self.backbone['dropout'](hidden)
    logits = self.backbone['output'](hidden).to(self.output_dtype)
    if self.mask_index < VOCAB_SIZE:
      logits[..., self.mask_index] = -torch.inf
    hidden = hidden.to(self.output_dtype)
    record.update(hidden=hidden, logits=logits,
                  hidden_before=hidden.clone(), logits_before=logits.clone())
    return hidden, logits


def make_head(variant):
  common = dict(hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE,
                rank=3, top_k=3, time_embed_dim=4)
  if variant == 'unary':
    head = ContextualUnaryAdapter(**common, correction_domain='candidates')
    # Nonzero corrections distinguish the returned raw logits from forward().
    with torch.no_grad():
      head.hidden_projection.weight.normal_(std=0.3)
      head.hidden_projection.bias.fill_(0.4)
      head.time_projection.weight.normal_(std=0.2)
      head.token_embedding.weight.normal_(std=0.5)
    return head
  cls = DirectionalCouplingForestHead if variant == 'directional' else ContextualCouplingForestHead
  return cls(**common, topology_dim=4, local_window=1, num_anchor_slots=1,
             contextual_neighbors=0, component_size_cap=3,
             topology_mode='fixed', factor_mode='dynamic')


def module_snapshot(*roots):
  """Include nonpersistent buffers, mixed modes, existing grads and identity."""
  return [
    {
      'flags': [(module, module.training) for module in root.modules()],
      'parameters': [(name, value, value.clone(), value.requires_grad,
                      value.grad, None if value.grad is None else value.grad.clone())
                     for name, value in root.named_parameters()],
      'buffers': [(name, value, value.clone()) for name, value in root.named_buffers()],
    }
    for root in roots
  ]


class StagedGenerationAdapterTest(unittest.TestCase):

  def setUp(self):
    rng_state = torch.random.get_rng_state()
    previous_threads = torch.get_num_threads()
    self.addCleanup(torch.random.set_rng_state, rng_state)
    self.addCleanup(torch.set_num_threads, previous_threads)
    torch.manual_seed(9103)
    torch.set_num_threads(1)
    self.tokens = torch.tensor([[8, 1, 8, 8, 3], [2, 8, 4, 8, 8], [0, 1, 2, 3, 4]])
    self.sigma = torch.tensor([0.8, 1.7, 0.0])
    self.active = self.tokens.eq(MASK_INDEX)

  def assert_modules_unchanged(self, before, *roots):
    for snapshot, root in zip(before, roots):
      self.assertEqual(snapshot['flags'], [(m, m.training) for m in root.modules()])
      current_parameters = dict(root.named_parameters())
      for name, parameter, value, requires_grad, grad, grad_value in snapshot['parameters']:
        self.assertIs(current_parameters[name], parameter, name)
        torch.testing.assert_close(parameter, value, atol=0, rtol=0, msg=name)
        self.assertEqual(parameter.requires_grad, requires_grad, name)
        self.assertIs(parameter.grad, grad, name)
        if grad is not None:
          torch.testing.assert_close(grad, grad_value, atol=0, rtol=0, msg=name)
      current_buffers = dict(root.named_buffers())
      for name, buffer, value in snapshot['buffers']:
        self.assertIs(current_buffers[name], buffer, name)
        torch.testing.assert_close(buffer, value, atol=0, rtol=0, msg=name)

  @staticmethod
  def mixed_training_state(model, head):
    model.train()
    model.backbone['embedding'].eval()
    model.backbone['normalization'].eval()
    head.eval()
    head.time_embedding.train()
    head.time_embedding.mlp[0].eval()
    for index, parameter in enumerate(list(model.parameters()) + list(head.parameters())):
      parameter.requires_grad_(index % 3 != 0)
      if index % 4 == 1:
        parameter.grad = torch.full_like(parameter, 0.125)

  def assert_no_edges_to_context(self, output, active):
    for row in range(len(active)):
      edges = output.edge_index[row, output.edge_mask[row]]
      self.assertTrue(bool(active[row, edges].all()))

  def test_all_heads_encode_serially_with_sigma_and_leave_state_untouched(self):
    for variant in VARIANTS:
      with self.subTest(variant=variant):
        model, head = RecordingBackbone(), make_head(variant)
        self.mixed_training_state(model, head)
        before = module_snapshot(model, head)
        inputs = [value.clone() for value in (self.tokens, self.sigma, self.active)]
        torch_rng, python_rng = torch.random.get_rng_state(), random.getstate()
        adapter = StagedGenerationAdapter(model, head)
        output, raw = adapter(self.tokens, self.sigma, self.active)
        self.assertIsInstance(output, StructuredDecoderOutput)
        self.assertEqual(adapter.logical_batch_calls, 1)
        self.assertEqual(adapter.physical_encoder_calls, len(self.tokens))
        self.assertEqual(len(model.calls), len(self.tokens))
        for row, call in enumerate(model.calls):
          torch.testing.assert_close(call['tokens'], self.tokens[row:row + 1], atol=0, rtol=0)
          torch.testing.assert_close(call['conditioning'], self.sigma[row:row + 1, None], atol=0, rtol=0)
          self.assertTrue(call['force_no_grad'])
          self.assertFalse(call['grad_enabled'])
          self.assertFalse(any(call['training']))
          for name in ('hidden', 'logits'):
            torch.testing.assert_close(call[name], call[name + '_before'], atol=0, rtol=0)
        expected_raw = torch.cat([call['logits_before'] for call in model.calls]).float()
        torch.testing.assert_close(raw, expected_raw, atol=0, rtol=0)
        self.assertEqual(raw.dtype, torch.float32)
        self.assertTrue(bool(torch.isneginf(raw[..., MASK_INDEX]).all()))
        torch.testing.assert_close(output.candidate_ids, raw.topk(head.top_k, -1).indices, atol=0, rtol=0)
        self.assert_no_edges_to_context(output, self.active)
        self.assertEqual(output.candidate_ids.shape[:2], self.tokens.shape)
        if variant == 'unary':
          self.assertFalse(bool(output.edge_mask.any()))
        else:
          self.assertTrue(bool(output.edge_mask.any()))
          self.assertEqual(output.topology_mode, head.topology_mode)
          self.assertEqual(output.factor_mode, head.factor_mode)
        for field in dataclasses.fields(output):
          value = getattr(output, field.name)
          if torch.is_tensor(value):
            self.assertFalse(value.requires_grad, field.name)
            if value.is_floating_point():
              self.assertEqual(value.dtype, torch.float32, field.name)
        self.assertFalse(raw.requires_grad)
        self.assert_modules_unchanged(before, model, head)
        self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        for actual, expected in zip((self.tokens, self.sigma, self.active), inputs):
          torch.testing.assert_close(actual, expected, atol=0, rtol=0)

  def test_head_runs_fp32_in_eval_without_grad_inside_outer_cpu_autocast(self):
    for variant in VARIANTS:
      with self.subTest(variant=variant):
        model, head = RecordingBackbone(output_dtype=torch.bfloat16), make_head(variant)
        head.train()
        observations = []

        def observe(module, args):
          observations.append((torch.is_grad_enabled(), torch.is_autocast_enabled('cpu'),
                               tuple(item.dtype for item in args if torch.is_tensor(item)),
                               tuple(child.training for child in head.modules())))

        handle = head.hidden_norm.register_forward_pre_hook(observe)
        adapter = StagedGenerationAdapter(model, head)
        try:
          with torch.enable_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            output, raw = adapter(self.tokens, self.sigma, self.active)
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        finally:
          handle.remove()
        self.assertTrue(observations)
        for grad, autocast, dtypes, training in observations:
          self.assertFalse(grad)
          self.assertFalse(autocast)
          self.assertEqual(dtypes, (torch.float32,))
          self.assertFalse(any(training))
        self.assertTrue(head.training)
        self.assertEqual(raw.dtype, torch.float32)
        self.assertEqual(output.unary_log_potentials.dtype, torch.float32)
        self.assertEqual(output.pair_left_factors.dtype, torch.float32)
        torch.testing.assert_close(raw, torch.cat([call['logits'].float() for call in model.calls]),
                                   atol=0, rtol=0)

  def test_unary_lattice_and_expanded_marginals_match_full_forward(self):
    model, head = RecordingBackbone(), make_head('unary')
    adapter = StagedGenerationAdapter(model, head)
    # The callback must use only the compact, target-free unary API.
    with mock.patch.object(head, 'forward', side_effect=AssertionError('full logits requested')), \
         mock.patch.object(head, 'target_log_probs', side_effect=AssertionError('target helper requested')), \
         mock.patch.object(head, 'nll_loss', side_effect=AssertionError('target loss requested')):
      output, raw = adapter(self.tokens, self.sigma, self.active)
    hidden = torch.cat([call['hidden'] for call in model.calls]).float()
    with torch.no_grad():
      lattice = head.candidate_lattice(hidden, raw, self.sigma, self.active)
      corrected = head(hidden, raw, self.sigma, self.active)
    for name in ('candidate_ids', 'unary_log_potentials', 'candidate_state_mask', 'residual_log_mass'):
      torch.testing.assert_close(getattr(output, name), getattr(lattice, name), atol=2e-6, rtol=2e-6)
    self.assertFalse(bool(output.edge_mask.any()))
    self.assertFalse(bool(output.proposal_edge_mask.any()))
    inferred = infer_structured_distribution(output, self.active)
    self.assertTrue(bool(inferred.clamped_states[~self.active].eq(0).all()))
    inactive_log_marginals = inferred.marginals.node_log_marginals[~self.active]
    torch.testing.assert_close(inactive_log_marginals[:, 0],
                               torch.zeros_like(inactive_log_marginals[:, 0]), atol=0, rtol=0)
    self.assertTrue(bool(torch.isneginf(inactive_log_marginals[:, 1:]).all()))
    full = full_vocabulary_marginals(output, raw, self.active, inferred)
    torch.testing.assert_close(full[self.active], corrected.softmax(-1)[self.active], atol=2e-6, rtol=2e-6)
    self.assertFalse(torch.equal(raw[self.active], corrected[self.active]))
    expected_raw = torch.cat([call['logits_before'] for call in model.calls])
    torch.testing.assert_close(raw, expected_raw, atol=0, rtol=0)
    explicit = torch.zeros_like(raw, dtype=torch.bool).scatter(-1, output.candidate_ids, True)
    tail_logits = raw.masked_fill(explicit, -torch.inf)
    torch.testing.assert_close(output.residual_log_probs(raw), tail_logits.log_softmax(-1),
                               atol=2e-6, rtol=2e-6)

  def test_pair_heads_preserve_selected_modes_and_sigma(self):
    for variant in ('shared', 'directional'):
      for factor_mode, independent_mode in (('fixed', False), ('dynamic', True)):
        with self.subTest(variant=variant, factor_mode=factor_mode, independent=independent_mode):
          model, head = RecordingBackbone(), make_head(variant)
          head.factor_mode = factor_mode
          head.independent_mode = independent_mode
          output, raw = StagedGenerationAdapter(model, head)(self.tokens, self.sigma, self.active)
          hidden = torch.cat([call['hidden'] for call in model.calls]).float()
          with torch.no_grad():
            expected = head(hidden, raw, self.sigma, self.active)
          for field in dataclasses.fields(output):
            actual_value, expected_value = getattr(output, field.name), getattr(expected, field.name)
            if torch.is_tensor(actual_value):
              torch.testing.assert_close(actual_value, expected_value, atol=2e-6, rtol=2e-6, msg=field.name)
            else:
              self.assertEqual(actual_value, expected_value, field.name)

  def test_encoder_failure_restores_every_mode_and_counts_actual_invocations(self):
    for variant in VARIANTS:
      with self.subTest(variant=variant):
        model, head = RecordingBackbone(), make_head(variant)
        self.mixed_training_state(model, head)
        adapter = StagedGenerationAdapter(model, head)
        before, rng = module_snapshot(model, head), torch.random.get_rng_state()
        model.failure_call = 2
        with self.assertRaisesRegex(RuntimeError, 'injected encoder failure'):
          adapter(self.tokens, self.sigma, self.active)
        self.assertEqual(adapter.logical_batch_calls, 1)
        self.assertEqual(adapter.physical_encoder_calls, 2)
        self.assert_modules_unchanged(before, model, head)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        model.failure_call = None
        adapter(self.tokens[:1], self.sigma[:1], self.active[:1])
        self.assertEqual(adapter.logical_batch_calls, 2)
        self.assertEqual(adapter.physical_encoder_calls, 3)

  def test_head_failure_restores_every_mode(self):
    for variant in VARIANTS:
      with self.subTest(variant=variant):
        model, head = RecordingBackbone(), make_head(variant)
        self.mixed_training_state(model, head)
        adapter = StagedGenerationAdapter(model, head)
        before, rng = module_snapshot(model, head), torch.random.get_rng_state()

        def fail(module, args):
          self.assertFalse(torch.is_grad_enabled())
          self.assertFalse(any(child.training for child in head.modules()))
          raise RuntimeError('injected head failure')

        handle = head.hidden_norm.register_forward_pre_hook(fail)
        try:
          with self.assertRaisesRegex(RuntimeError, 'injected head failure'):
            adapter(self.tokens, self.sigma, self.active)
        finally:
          handle.remove()
        self.assertEqual(adapter.logical_batch_calls, 1)
        self.assertEqual(adapter.physical_encoder_calls, len(model.calls))
        self.assertGreater(adapter.physical_encoder_calls, 0)
        self.assert_modules_unchanged(before, model, head)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

  def test_malformed_backbone_output_restores_modes_after_first_encoder_call(self):
    mask_slots = torch.arange(VOCAB_SIZE).eq(MASK_INDEX)[None, None]
    corruptions = {
      'hidden shape': lambda hidden, logits: (hidden[..., :-1], logits),
      'logit shape': lambda hidden, logits: (hidden, logits[..., :-1]),
      'nan hidden': lambda hidden, logits: (torch.full_like(hidden, torch.nan), logits),
      'nan logits': lambda hidden, logits: (hidden, torch.full_like(logits, torch.nan)),
      'positive infinite logits': lambda hidden, logits: (hidden, torch.full_like(logits, torch.inf)),
      'no finite logit support': lambda hidden, logits: (hidden, torch.full_like(logits, -torch.inf)),
      'finite absorbing mask logit': lambda hidden, logits: (hidden, logits.masked_fill(mask_slots, 0.0)),
    }
    for variant in VARIANTS:
      for name, corrupt in corruptions.items():
        with self.subTest(variant=variant, corruption=name):
          model, head = RecordingBackbone(), make_head(variant)
          self.mixed_training_state(model, head)
          adapter = StagedGenerationAdapter(model, head)
          before, rng = module_snapshot(model, head), torch.random.get_rng_state()
          encode = model._structured_backbone_output

          def malformed(*args, **kwargs):
            return corrupt(*encode(*args, **kwargs))

          with mock.patch.object(model, '_structured_backbone_output', side_effect=malformed) as encoder:
            with self.assertRaises(ValueError):
              adapter(self.tokens, self.sigma, self.active)
          self.assertEqual(encoder.call_count, 1)
          self.assertEqual(len(model.calls), 1)
          self.assertFalse(any(model.calls[0]['training']))
          self.assertEqual(adapter.logical_batch_calls, 1)
          self.assertEqual(adapter.physical_encoder_calls, 1)
          self.assert_modules_unchanged(before, model, head)
          self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

  def test_invalid_inputs_fail_before_encoding_or_incrementing_counters(self):
    bad_tokens = self.tokens.clone()
    bad_tokens[0, 0] = VOCAB_SIZE
    cases = {
      'token rank': (self.tokens[0], self.sigma, self.active),
      'token dtype': (self.tokens.int(), self.sigma, self.active),
      'empty batch': (self.tokens[:0], self.sigma[:0], self.active[:0]),
      'empty sequence': (self.tokens[:, :0], self.sigma, self.active[:, :0]),
      'negative token': (self.tokens - 20, self.sigma, self.active),
      'invalid token': (bad_tokens, self.sigma, bad_tokens.eq(MASK_INDEX)),
      'sigma rank': (self.tokens, self.sigma[:, None], self.active),
      'sigma batch': (self.tokens, self.sigma[:1], self.active),
      'integer sigma': (self.tokens, self.sigma.long(), self.active),
      'negative sigma': (self.tokens, -self.sigma, self.active),
      'nan sigma': (self.tokens, self.sigma * torch.nan, self.active),
      'infinite sigma': (self.tokens, torch.full_like(self.sigma, torch.inf), self.active),
      'sigma overflows fp32': (self.tokens, torch.full(self.sigma.shape, 1e100, dtype=torch.float64), self.active),
      'active dtype': (self.tokens, self.sigma, self.active.long()),
      'active shape': (self.tokens, self.sigma, self.active[:, :1]),
      'missing active mask': (self.tokens, self.sigma, torch.zeros_like(self.active)),
      'active context': (self.tokens, self.sigma, torch.ones_like(self.active)),
      'sigma device': (self.tokens, self.sigma.to('meta'), self.active),
      'active device': (self.tokens, self.sigma, self.active.to('meta')),
      'token device': (self.tokens.to('meta'), self.sigma.to('meta'), self.active.to('meta')),
    }
    model, head = RecordingBackbone(), make_head('shared')
    adapter = StagedGenerationAdapter(model, head)
    before = module_snapshot(model, head)
    for name, args in cases.items():
      with self.subTest(case=name):
        with self.assertRaises((TypeError, ValueError)):
          adapter(*args)
        self.assertEqual(adapter.logical_batch_calls, 0)
        self.assertEqual(adapter.physical_encoder_calls, 0)
        self.assertEqual(model.calls, [])
        self.assert_modules_unchanged(before, model, head)

  def test_invalid_head_configuration_is_rejected_without_recasting(self):
    for variant in VARIANTS:
      for dtype in (torch.float64, torch.bfloat16):
        with self.subTest(variant=variant, dtype=dtype):
          model, head = RecordingBackbone(), make_head(variant).to(dtype)
          before = module_snapshot(model, head)
          with self.assertRaises((TypeError, ValueError)):
            adapter = StagedGenerationAdapter(model, head)
            adapter(self.tokens, self.sigma, self.active)
          self.assert_modules_unchanged(before, model, head)
          self.assertEqual(model.calls, [])
    model, head = RecordingBackbone(), make_head('unary')
    head.correction_domain = 'full'
    with self.assertRaises((TypeError, ValueError)):
      adapter = StagedGenerationAdapter(model, head)
      adapter(self.tokens, self.sigma, self.active)
    with self.assertRaises((TypeError, ValueError)):
      StagedGenerationAdapter(model, torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE))
    with self.assertRaises((TypeError, ValueError)):
      adapter = StagedGenerationAdapter(model, make_head('shared').to('meta'))
      adapter(self.tokens, self.sigma, self.active)
    self.assertEqual(model.calls, [])

  def test_mask_override_must_agree_and_external_sentinel_is_valid(self):
    for mask_index in (MASK_INDEX, VOCAB_SIZE + 2):
      with self.subTest(mask_index=mask_index):
        model, head = RecordingBackbone(mask_index), make_head('shared')
        tokens = self.tokens.masked_fill(self.active, mask_index)
        adapter = StagedGenerationAdapter(model, head, mask_index=mask_index)
        output, raw = adapter(tokens, self.sigma, tokens.eq(mask_index))
        self.assertEqual(raw.shape, (*tokens.shape, VOCAB_SIZE))
        self.assertFalse(bool(output.candidate_ids.eq(mask_index).any()))
        with self.assertRaises((TypeError, ValueError)):
          StagedGenerationAdapter(model, head, mask_index=mask_index + 1)

  def test_group_one_joint_and_marginal_have_identical_trajectories_and_rng(self):
    order = make_reveal_order(self.tokens.shape[1], [12, 13, 14])
    for variant in VARIANTS:
      with self.subTest(variant=variant):
        model, head = RecordingBackbone(), make_head(variant)
        runs, rng_states = [], []
        for mode in ('joint', 'marginal'):
          adapter = StagedGenerationAdapter(model, head)
          generator = torch.Generator().manual_seed(803)
          result = generate_fixed_groups(
            adapter, self.tokens, mask_index=MASK_INDEX, group_size=1, mode=mode,
            reveal_order=order, sampling_generator=generator)
          self.assertEqual(adapter.logical_batch_calls, result.batch_calls)
          self.assertEqual(adapter.physical_encoder_calls, int(result.nfe.sum()))
          self.assertEqual(result.nfe.tolist(), [3, 3, 0])
          runs.append(result)
          rng_states.append(generator.get_state())
        self.assertTrue(torch.equal(rng_states[0], rng_states[1]))
        for field in dataclasses.fields(runs[0]):
          if field.name in ('mode', 'steps'):
            continue
          left, right = getattr(runs[0], field.name), getattr(runs[1], field.name)
          if torch.is_tensor(left):
            torch.testing.assert_close(left, right, atol=0, rtol=0, msg=field.name)
          else:
            self.assertEqual(left, right, field.name)
        self.assertEqual(len(runs[0].steps), len(runs[1].steps))
        for left, right in zip(runs[0].steps, runs[1].steps):
          for field in dataclasses.fields(left):
            first, second = getattr(left, field.name), getattr(right, field.name)
            if torch.is_tensor(first):
              torch.testing.assert_close(first, second, atol=0, rtol=0, msg=field.name)
            else:
              self.assertEqual(first, second, field.name)

  def test_group_two_preserves_context_and_counts_shrinking_serial_batches(self):
    tokens = self.tokens.clone()
    tokens[1, 3:] = torch.tensor([5, 6])
    active = tokens.eq(MASK_INDEX)
    initial = tokens.clone()
    order = torch.arange(tokens.shape[1]).expand_as(tokens)
    for variant in VARIANTS:
      for mode in ('joint', 'marginal', 'backbone'):
        with self.subTest(variant=variant, mode=mode):
          model, head = RecordingBackbone(), make_head(variant)
          adapter = StagedGenerationAdapter(model, head)
          result = generate_fixed_groups(
            adapter, tokens, mask_index=MASK_INDEX, group_size=2, mode=mode,
            reveal_order=order, sampling_generator=torch.Generator().manual_seed(194))
          torch.testing.assert_close(tokens, initial, atol=0, rtol=0)
          torch.testing.assert_close(result.final_tokens[~active], tokens[~active], atol=0, rtol=0)
          self.assertFalse(bool(result.final_tokens.eq(MASK_INDEX).any()))
          self.assertEqual(result.nfe.tolist(), [2, 1, 0])
          self.assertEqual(adapter.logical_batch_calls, result.batch_calls)
          self.assertEqual(adapter.logical_batch_calls, 2)
          self.assertEqual(adapter.physical_encoder_calls, int(result.nfe.sum()))
          self.assertEqual(len(model.calls), 3)
          for step in result.steps:
            torch.testing.assert_close(step.tokens_after[~step.active_before],
                                       step.tokens_before[~step.active_before], atol=0, rtol=0)
            torch.testing.assert_close(step.revealed_mask.sum(-1),
                                       step.active_before.sum(-1).clamp_max(2), atol=0, rtol=0)
            for row in range(len(step.row_indices)):
              edges = step.edge_index[row, step.edge_mask[row]]
              self.assertTrue(bool(step.active_before[row, edges].all()))


if __name__ == '__main__':
  unittest.main()
