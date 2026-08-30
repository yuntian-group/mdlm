#!/usr/bin/env python3
"""Strictly load a released DIT backbone and run one structured CUDA pass."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import torch  # noqa: E402

import dataloader  # noqa: E402
import diffusion  # noqa: E402
import structured_objective  # noqa: E402


def _register_resolvers() -> None:
  resolvers = {
    'cwd': os.getcwd,
    'device_count': torch.cuda.device_count,
    'eval': eval,
    'div_up': lambda x, y: (x + y - 1) // y,
  }
  for name, resolver in resolvers.items():
    if not OmegaConf.has_resolver(name):
      OmegaConf.register_new_resolver(name, resolver)


def _args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--device', default='cuda')
  return parser.parse_args()


def main() -> int:
  args = _args()
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  device = torch.device(args.device)
  if device.type == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable')
  _register_resolvers()
  overrides = [
    'model=contextual-forest-small',
    'data=openwebtext',
    ('model.structured_decoder.training.backbone_checkpoint='
     + str(args.checkpoint.resolve())),
    'model.structured_decoder.training.use_ema_backbone=false',
    'model.structured_decoder.training.strict_backbone_checkpoint=true',
    'training.ema=0',
    'checkpointing.resume_from_ckpt=false',
    'eval.generate_samples=false',
  ]
  with hydra.initialize_config_dir(
      config_dir=str(REPO_ROOT / 'configs'), version_base=None):
    config = hydra.compose(config_name='config', overrides=overrides)
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion(config, tokenizer=tokenizer).to(device).eval()
  if any(parameter.requires_grad for parameter in model.backbone.parameters()):
    raise AssertionError('frozen-backbone preflight found trainable parameters')

  tokens = torch.full(
    (1, int(config.model.length)), model.mask_index,
    dtype=torch.long, device=device)
  active_mask = torch.ones_like(tokens, dtype=torch.bool)
  diffusion_time = torch.full((1,), 0.5, device=device)
  conditioning, _ = model.noise(diffusion_time)
  if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
  start = time.perf_counter()
  with torch.no_grad():
    output, unary_logits = model._structured_head_output(
      tokens=tokens,
      conditioning=conditioning[:, None],
      active_mask=active_mask,
      force_no_grad_backbone=True)
    inference = structured_objective.infer_structured_distribution(
      output, active_mask, backend='low_rank')
  if device.type == 'cuda':
    torch.cuda.synchronize(device)
  elapsed = time.perf_counter() - start

  finite_unaries = torch.isfinite(unary_logits[..., :-1]).all()
  finite_marginals = torch.isfinite(inference.node_marginals).all()
  normalization_error = (
    inference.node_marginals.sum(dim=-1) - 1.0).abs().max()
  if not bool(finite_unaries.item()):
    raise AssertionError('released backbone produced non-finite token logits')
  if not bool(finite_marginals.item()):
    raise AssertionError('structured inference produced non-finite marginals')
  if float(normalization_error) > 2e-5:
    raise AssertionError(
      f'node marginals are not normalized: {float(normalization_error)}')

  try:
    git_sha = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    git_sha = 'unknown'
  payload = {
    'benchmark': 'released_backbone_structured_preflight',
    'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'git_sha': git_sha,
    'checkpoint': str(args.checkpoint.resolve()),
    'device': str(device),
    'gpu': (torch.cuda.get_device_name(device)
            if device.type == 'cuda' else None),
    'torch': torch.__version__,
    'sequence_length': int(config.model.length),
    'vocab_size': int(model.vocab_size),
    'top_k': int(model.structured_head.top_k),
    'rank': int(model.structured_head.rank),
    'selected_edges': int(output.edge_mask.sum()),
    'retained_mass_mean': float(output.retained_mass.mean()),
    'max_node_normalization_error': float(normalization_error),
    'elapsed_seconds': elapsed,
    'peak_memory_bytes': (
      int(torch.cuda.max_memory_allocated(device))
      if device.type == 'cuda' else None),
    'backbone_frozen': True,
    'ema_used': False,
  }
  args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
  args.output.resolve().write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n')
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
