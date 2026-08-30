import unittest

import torch

from synthetic.distributions import ContextSwitchingMatching
from synthetic.neural_g1 import (
  NeuralTrainConfig,
  dependency_loss,
  model_specs,
  sample_training_batch,
  train_adapter,
)
from structured_objective import structured_token_log_probability


class NeuralG1Test(unittest.TestCase):

  def test_batch_respects_context_matching(self):
    task = ContextSwitchingMatching(vocab_size=4)
    contexts, tokens, _ = sample_training_batch(
      task, 200, torch.Generator().manual_seed(5), torch.device('cpu'))
    for row in range(tokens.shape[0]):
      for first, second in task.true_edges(int(contexts[row])):
        self.assertEqual(int(tokens[row, first]), int(tokens[row, second]))

  def test_contextual_adapter_has_finite_gradients(self):
    task = ContextSwitchingMatching(vocab_size=3)
    spec = model_specs(task)['contextual_forest']
    model, history = train_adapter(
      task, spec, seed=2,
      config=NeuralTrainConfig(
        steps=2, batch_size=6, eval_samples=20, log_every=1),
      device=torch.device('cpu'))
    self.assertTrue(all(torch.isfinite(torch.tensor([
      row['loss'], row['structured_nll'], row['dependency_loss'],
      row['gradient_norm']])).all() for row in history))
    contexts, tokens, timestep = sample_training_batch(
      task, 4, torch.Generator().manual_seed(8), torch.device('cpu'))
    output, logits, active = model(contexts, timestep)
    log_probability = structured_token_log_probability(
      output, logits, tokens, active)
    self.assertTrue(bool(torch.isfinite(log_probability).all()))
    self.assertGreater(
      float(dependency_loss(task, contexts, output).detach()), 0.0)


if __name__ == '__main__':
  unittest.main()
