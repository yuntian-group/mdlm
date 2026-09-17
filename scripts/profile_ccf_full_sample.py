#!/usr/bin/env python3
"""One unchanged full CCF sample, phase timings, and same-GPU MDLM control.

Instrumentation only: defaults to v2; opt-in faster experimental variants.
Host exclusive times include time blocked on CUDA. CUDA phase intervals include
host launch gaps and are not kernel-busy time; nested intervals overlap.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import functools
import json
import os
from pathlib import Path
import sys
import time
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import structured_objective as objective
import structured_utils as utils
import models.structured_decoder as decoder
from evaluation import generation_harness as harness


class PhaseTimer:
  def __init__(self, cuda=True):
    self.cuda = cuda
    self.stack = []
    self.stats = defaultdict(lambda: {'calls': 0, 'host_inclusive_seconds': 0.,
                                      'host_exclusive_seconds': 0.})
    self.events = defaultdict(list)

  def wrap(self, name, function, cuda_events=True):
    @functools.wraps(function)
    def timed(*args, **kwargs):
      begin = time.perf_counter()
      frame = [0.]
      self.stack.append(frame)
      if self.cuda and cuda_events:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
      try:
        return function(*args, **kwargs)
      finally:
        if self.cuda and cuda_events:
          end.record()
          self.events[name].append((start, end))
        elapsed = time.perf_counter() - begin
        self.stack.pop()
        record = self.stats[name]
        record['calls'] += 1
        record['host_inclusive_seconds'] += elapsed
        record['host_exclusive_seconds'] += elapsed - frame[0]
        if self.stack:
          self.stack[-1][0] += elapsed
    return timed

  def report(self, wall_seconds):
    if self.cuda:
      torch.cuda.synchronize()
    rows = []
    for name, record in self.stats.items():
      row = dict(phase=name, **record)
      row['exclusive_percent_of_generation_wall'] = 100 * record['host_exclusive_seconds'] / wall_seconds
      if name in self.events:
        row['cuda_inclusive_interval_seconds'] = sum(a.elapsed_time(b) for a, b in self.events[name]) / 1000
      rows.append(row)
    return sorted(rows, key=lambda row: row['host_exclusive_seconds'], reverse=True)


def install_timers(stack, timer, model, variant='v2'):
  targets = [
    (harness, 'sample_from_initial_state', 'schedule_and_cleanup', True),
    (model, '_ddpm_update', 'diffusion_update_other', True),
    (model, '_structured_backbone_output', 'backbone_encode_decode', True),
    (model.structured_head, 'forward', 'head_other_gathers_normalization', True),
    (model.structured_head, '_candidate_lattice', 'topk_candidate_lattice', True),
    (model.structured_head.edge_proposer, 'forward', 'edge_proposal_network', True),
    (model.structured_head, '_selected_edges', 'selected_edge_gathers', True),
    (decoder, '_bounded_kruskal_indices', 'forest_selection_kruskal', True),
    (model.structured_head, '_node_candidate_factors', 'endpoint_factor_embeddings_film', True),
    (objective, 'sample_structured_tokens', 'token_mapping_residual_draw_other', True),
    (utils, 'sample_forest_low_rank', 'ancestral_traversal_other', True),
    (utils, '_validate_low_rank_inputs', 'factor_validation_and_logs', True),
    (utils, '_build_topology', 'forest_validation_and_traversal_metadata', True),
    (utils, '_constrain_nodes', 'clamp_validation_and_masks', True),
    (utils, '_single_low_rank_sum_product', 'upward_inference_other_root_marginals', True),
    (utils, '_sample_rows', ('categorical_draws_including_native_checks'
       if variant == 'v2' else 'categorical_draws_without_native_checks'), False),
    (utils, '_low_rank_pair_rows', 'conditional_pair_rows', False),
    (utils, '_low_rank_message', 'upward_message_arithmetic', False),
    (utils, '_batched_low_rank_message', 'batched_upward_message_arithmetic', False),
    (decoder.StructuredDecoderOutput, 'residual_log_probs', 'residual_vocabulary_preparation', True),
  ]
  from scripts import audit_ccf_sampling_v4 as level
  if variant in level.MODES:
    targets.extend([
      (level, 'grouped_pair_rows', 'batched_child_conditional_arithmetic', False),
      (level, 'draw_with_noise', 'batched_softmax_divide_argmax', False),
    ])
  for owner, attribute, label, events in targets:
    stack.enter_context(mock.patch.object(owner, attribute,
      timer.wrap(label, getattr(owner, attribute), cuda_events=events)))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, required=True)
  from scripts.audit_ccf_sampling_v4 import MODES, experiment
  parser.add_argument('--variant', choices=('v2', 'unchecked_batched_roots_kruskal') + MODES, default='v2')
  parser.add_argument('--compare-uninstrumented', action='store_true')
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=False)
  start_loading = time.perf_counter()
  import dataloader
  import diffusion
  from scripts import run_generation_pilot as pilot
  cache = Path(os.environ.get('CCF_CACHE_ROOT', str(ROOT / 'local-data')))
  source = cache / 'runs/ccf_separate_r16_dynamic_dynamic_s001_step7000_run-separate-r16-dd/attempt-1543861-r0.AZ0GWx'
  backbone = cache / 'checkpoints/mdlm-owt-backbone.pt'
  adapter = source / 'dynamic_dynamic.safetensors'
  manifest = source / 'dynamic_dynamic.manifest.json'
  config_args = pilot._parse_args([
    '--backbone-checkpoint', str(backbone), '--backbone-sha256', pilot.sha256_file(backbone),
    '--adapter', str(adapter), '--adapter-sha256', pilot.sha256_file(adapter),
    '--adapter-manifest', str(manifest), '--adapter-manifest-sha256', pilot.sha256_file(manifest),
    '--output-dir', str(args.output_dir), '--sequence-length', '1024',
    '--batch-size', '1', '--num-samples', '1', '--base-seed', '91001',
    '--modes', 'structured_joint', '--nfe-budgets', '1001',
    '--model-config', 'contextual-forest-small', '--data-config', 'train_openwebtext_pinned',
    '--override', 'data.cache_dir=' + str(cache / 'huggingface'),
    '--override', 'model.structured_decoder.top_k=128',
    '--override', 'model.structured_decoder.rank=16',
    '--override', '++model.structured_decoder.factor_embedding_mode=separate',
    '--override', '++model.structured_decoder.factor_conditioner_hidden_dim=0',
    '--override', 'model.structured_decoder.topology_mode=dynamic',
    '--override', 'model.structured_decoder.factor_mode=dynamic',
    '--override', 'model.structured_decoder.independent_mode=false',
    '--override', 'model.structured_decoder.training.topology_weight=0.1'])
  config = pilot._compose_config(config_args)
  tokenizer = dataloader.get_tokenizer(config)
  device = torch.device('cuda')
  model = diffusion.Diffusion(config, tokenizer=tokenizer).to(device).eval()
  torch.cuda.synchronize()
  loading_seconds = time.perf_counter() - start_loading
  prompts = [harness.unconditional_prompt(mask_token_id=model.mask_index, sequence_length=1024)]
  samples = harness.expand_paired_samples(prompts, num_samples=1, base_seed=91001)
  timer = PhaseTimer()
  print(json.dumps({'event': 'loaded', 'loading_seconds': loading_seconds,
                    'gpu': torch.cuda.get_device_name()}), flush=True)
  with ExitStack() as stack:
    if args.variant != 'v2':
      stack.enter_context(experiment(args.variant))
    install_timers(stack, timer, model, args.variant)
    records, timing = harness.run_sampling_group(model, samples,
      sampling_mode='structured_joint', nfe_budget=1001, tokenizer=tokenizer, device=device)
  profiled_cpu_rng = torch.get_rng_state().clone()
  profiled_cuda_rng = torch.cuda.get_rng_state().clone()
  report = dict(variant=args.variant, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
    adapter=str(adapter), adapter_sha256=pilot.sha256_file(adapter),
    loading_seconds=loading_seconds, ccf_timing=timing,
    phase_rows=timer.report(timing['wall_clock_seconds']),
    timing_caveat='Host exclusive time includes CUDA waits; CUDA inclusive intervals overlap and include launch gaps. Instrumented CCF time includes wrapper/event overhead. No GPT-2 scoring.',
    ccf_text=records[0]['text'] if 'text' in records[0] else None)
  (args.output_dir / 'ccf-sample.json').write_text(json.dumps(records[0], indent=2))
  (args.output_dir / 'breakdown.json').write_text(json.dumps(report, indent=2))
  print(json.dumps({'event': 'ccf_complete', 'timing': timing,
                    'phases': report['phase_rows']}), flush=True)
  if args.compare_uninstrumented:
    with ExitStack() as stack:
      if args.variant != 'v2':
        stack.enter_context(experiment(args.variant))
      replay_records, replay_timing = harness.run_sampling_group(model, samples,
        sampling_mode='structured_joint', nfe_budget=1001, tokenizer=tokenizer, device=device)
    equality = dict(
      tokens_equal=records[0]['sample_token_ids'] == replay_records[0]['sample_token_ids'],
      nfe_equal=timing['measured_nfe'] == replay_timing['measured_nfe'],
      cpu_rng_equal=torch.equal(profiled_cpu_rng, torch.get_rng_state()),
      cuda_rng_equal=torch.equal(profiled_cuda_rng, torch.cuda.get_rng_state()))
    report['uninstrumented_ccf_timing'] = replay_timing
    report['profile_replay_equality'] = equality
    report['profile_vs_replay_extra_seconds'] = timing['wall_clock_seconds'] - replay_timing['wall_clock_seconds']
    (args.output_dir / 'ccf-uninstrumented.json').write_text(json.dumps(replay_records[0], indent=2))
    (args.output_dir / 'breakdown.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'event': 'ccf_replay_complete', 'timing': replay_timing, 'equality': equality}), flush=True)
    assert all(equality.values()), 'Profiling changed sampled tokens/NFE/RNG'
  # Same already-loaded backbone/device/settings; factorized mode bypasses CCF.
  baseline_records, baseline_timing = harness.run_sampling_group(model, samples,
    sampling_mode='factorized', nfe_budget=1001, tokenizer=tokenizer, device=device)
  report['same_gpu_mdlm_timing'] = baseline_timing
  report['instrumented_ccf_over_mdlm'] = timing['wall_clock_seconds'] / baseline_timing['wall_clock_seconds']
  if args.compare_uninstrumented:
    report['uninstrumented_ccf_over_mdlm'] = replay_timing['wall_clock_seconds'] / baseline_timing['wall_clock_seconds']
  (args.output_dir / 'mdlm-sample.json').write_text(json.dumps(baseline_records[0], indent=2))
  (args.output_dir / 'breakdown.json').write_text(json.dumps(report, indent=2))
  print(json.dumps({'event': 'profile_complete', 'mdlm_timing': baseline_timing}), flush=True)


if __name__ == '__main__':
  main()
