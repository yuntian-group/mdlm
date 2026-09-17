#!/usr/bin/env python3
"""Read-only sampler audit with an isolated root-only inference prototype.

No production module is edited. The prototype replaces a private helper only
inside scoped mock.patch contexts in this diagnostic process. Derived solely
from this repository's serial sum-product implementation; no online code used.
CPU timings do not establish a CUDA or full-generation speedup.
"""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import time
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import structured_objective as objective
import structured_utils as utils
from scripts.verify_ccf_sampling_optimization import problem, verify, compare_draws


def root_only_inference(nodes, left_logs, right_logs, edges, topology):
  """Prototype: same upward arithmetic and root normalization, no downpass.

  Return root marginals as a dictionary: the joint sampler indexes only roots.
  This is deliberately NOT a replacement for general marginal inference.
  """
  messages = {}
  for node in reversed(topology.order):
    parent = topology.parent[node]
    if parent < 0:
      continue
    local = nodes[node]
    for neighbor, _, _ in topology.adjacency[node]:
      if neighbor != parent:
        local = local + messages[(neighbor, node)]
    edge_id = topology.parent_edge[node]
    if topology.edge_left[edge_id] == node:
      source, target = left_logs[edge_id], right_logs[edge_id]
    else:
      source, target = right_logs[edge_id], left_logs[edge_id]
    messages[(node, parent)] = utils._low_rank_message(local, source, target)

  beliefs = {}
  for root in topology.roots:
    belief = nodes[root]
    for neighbor, _, _ in topology.adjacency[root]:
      belief = belief + messages[(neighbor, root)]
    beliefs[root] = belief
  partitions = torch.stack([
    torch.logsumexp(beliefs[root], dim=-1) for root in topology.roots])
  utils._require(bool(torch.isfinite(partitions).all().item()),
                 'constraints leave the forest with no finite-probability state')
  marginals = {}
  for root in topology.roots:
    value = beliefs[root] - partitions[topology.component[root]]
    marginals[root] = value - torch.logsumexp(value, dim=-1)
  return partitions.sum(), marginals, messages


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
  parser.add_argument('--repetitions', type=int, default=5)
  args = parser.parse_args()
  if args.repetitions < 1:
    parser.error('--repetitions must be positive')
  torch.set_num_threads(1)
  device = torch.device(args.device)
  original = utils._single_low_rank_sum_product
  def sync():
    if device.type == 'cuda':
      torch.cuda.synchronize(device)
  with mock.patch.object(utils, '_single_low_rank_sum_product', root_only_inference):
    equivalence = verify(device)
  records = []
  with torch.no_grad():
    for active_kind in ('all', 'mixed', 'none'):
      output, logits, active = problem(
        device, k=128, length=1024, active_kind=active_kind)
      def draw(generator):
        return objective.sample_structured_tokens(
          output, logits, active, generator=generator)
      def candidate(generator):
        with mock.patch.object(utils, '_single_low_rank_sum_product', root_only_inference):
          return draw(generator)
      for seed in (91001, 91002):
        compare_draws(draw, candidate, device, seed)
      timings = {'v1': [], 'root_only_prototype': []}
      # Alternate execution order to reduce warmup/order bias.
      for iteration in range(args.repetitions + 1):
        variants = [('v1', original), ('root_only_prototype', root_only_inference)]
        if iteration % 2:
          variants.reverse()
        for name, implementation in variants:
          generator = torch.Generator(device=device).manual_seed(91001)
          with mock.patch.object(utils, '_single_low_rank_sum_product', implementation):
            sync()
            start = time.perf_counter()
            draw(generator)
            sync()
            elapsed = time.perf_counter() - start
          if iteration:
            timings[name].append(elapsed)
      call_counts = {}
      for name in ('_sample_rows', '_low_rank_message', '_single_low_rank_sum_product'):
        with mock.patch.object(utils, name, wraps=getattr(utils, name)) as traced:
          draw(torch.Generator(device=device).manual_seed(91001))
          call_counts[name] = traced.call_count
      profiler = cProfile.Profile()
      profiler.runcall(draw, torch.Generator(device=device).manual_seed(91001))
      profile_output = io.StringIO()
      pstats.Stats(profiler, stream=profile_output).sort_stats('cumulative').print_stats(18)
      medians = {name: statistics.median(values) for name, values in timings.items()}
      records.append(dict(
        active_kind=active_kind, batch_size=2, length=1024, k=128, rank=16,
        synthetic_vocab=256, median_seconds=medians,
        isolated_speedup=medians['v1']/medians['root_only_prototype'],
        v1_call_counts=call_counts, v1_python_profile=profile_output.getvalue()))
  print(json.dumps(dict(
    device=str(device), torch_version=torch.__version__,
    production_changed=False, full_trajectory_verified=False,
    equivalence=equivalence, benchmarks=records), indent=2))


if __name__ == '__main__':
  main()
