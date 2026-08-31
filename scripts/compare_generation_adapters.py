#!/usr/bin/env python3
"""Verify and compare frozen paper-scale dynamic/static generation shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.generation_adapter_comparison import (  # noqa: E402
  compare_generation_adapters,
)


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      'Fail-closed paired comparison of the frozen static_static and '
      'dynamic_dynamic paper-scale generation shard unions.'))
  parser.add_argument(
    '--baseline-shard', action='append', type=Path, required=True,
    help='Static-control shard directory; repeat for all 16 shards.')
  parser.add_argument(
    '--treatment-shard', action='append', type=Path, required=True,
    help='Dynamic-treatment shard directory; repeat for all 16 shards.')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--bootstrap-resamples', type=int, default=20_000)
  parser.add_argument('--bootstrap-seed', type=int, default=94_001)
  parser.add_argument('--bootstrap-confidence', type=float, default=0.95)
  return parser.parse_args(argv)


def _atomic_write(path: Path, content: str) -> None:
  path = path.expanduser().resolve()
  if path.exists():
    raise FileExistsError(f'refusing to overwrite comparison artifact {path}')
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
  try:
    temporary.write_text(content)
    os.replace(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def main(argv=None) -> int:
  args = _parse_args(argv)
  result = compare_generation_adapters(
    args.baseline_shard,
    args.treatment_shard,
    bootstrap_resamples=args.bootstrap_resamples,
    bootstrap_seed=args.bootstrap_seed,
    bootstrap_confidence=args.bootstrap_confidence)
  output = args.output.expanduser().resolve()
  _atomic_write(output, json.dumps(result, indent=2, sort_keys=True) + '\n')
  print(json.dumps({
    'event': 'generation_adapter_comparison_verified',
    'output': str(output),
    'dataset_id': result['dataset_id'],
    'candidate_top_k': result['identity']['candidate_top_k'],
    'num_nfe_budgets': len(result['comparisons']),
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
