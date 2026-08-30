import unittest

import torch

from scripts.profile_forest_inference import (
  bounded_chain_edges,
  verify_dense_low_rank_agreement,
)


class ForestProfileTest(unittest.TestCase):

  def test_bounded_chain_splits_components(self):
    edges, mask = bounded_chain_edges(10, 3, torch.device('cpu'))
    selected = edges[mask].tolist()
    self.assertEqual(len(selected), 6)
    self.assertNotIn([2, 3], selected)
    self.assertNotIn([5, 6], selected)
    self.assertNotIn([8, 9], selected)

  def test_profile_reference_agreement(self):
    errors = verify_dense_low_rank_agreement()
    self.assertLess(errors['max_log_partition_error'], 1e-10)
    self.assertLess(errors['max_node_marginal_error'], 1e-10)


if __name__ == '__main__':
  unittest.main()
