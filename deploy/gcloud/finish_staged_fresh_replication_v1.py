#!/usr/bin/env python3
"""Seal complete replications, score only new seeds, and combine original shards.

Run from the authenticated analysis checkout only after seeds 2 and 3 finish.
All output namespaces are exclusive. A failed stage leaves its artifacts in
place and exits; it never replaces an earlier seal or test export.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_staged_fresh_tests import aggregate
from scripts.evaluate_staged_fresh_selected import main as evaluate
from scripts.seal_staged_fresh_selection import file_sha256, seal


DATA_SHA = '70d989988c6318a52960791667804fde30e794f324628186df8d010438439753'
TEST_SHA = '2e22dae402fe61c6febf2a0d48c8161c1c8bb44a757061730c11205d2b6bffbb'
EXPECTATIONS_SHA = 'a978c17befc2788b0b773eee7cd608ac0200379947a63a0ba8699a0b9818c956'
PILOT_MANIFEST_SHA = '7d425ba57cf1b5f419c8e90a3d7ddfe1e8afec6e2995ec46773b9a30e1bb5644'
PILOT_TEST_SHA = '538bf373dddcba7c0ed03a80f6f3bf594ccf1293f31895d1be98483a2f284cad'


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--experiment-root', type=Path, required=True)
  parser.add_argument('--run-tag', choices=('v1', 'retry1-v1'), default='retry1-v1')
  parser.add_argument('--output-tag', choices=('v1',), default='v1')
  args = parser.parse_args(argv)
  root, tag = args.experiment_root.resolve(), args.output_tag
  release = root.parent / 'releases/contextual-forest-adapter-cf8b808-20260831T131607Z'
  manifest = root / 'fresh-data-v1/manifest.json'
  pilot_run = root / 'fresh-pilot-resume-v1'
  replica_runs = [root / f'fresh-replication-seed{seed}-{args.run_tag}' for seed in (2, 3)]
  selection23 = root / f'fresh-selection-seed23-{tag}'
  selection123 = root / f'fresh-selection-seed123-{tag}'
  test23 = root / f'fresh-test-seed23-{tag}'
  test123 = root / f'fresh-test-seed123-{tag}'
  for path in (selection23, selection123, test23, test123):
    if path.exists() or path.is_symlink():
      raise FileExistsError(path)
  os.environ.update(
    HF_HOME=str(root.parent / 'hf-home'),
    HUGGINGFACE_HUB_CACHE=str(root.parent / 'huggingface/hub'),
    HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

  # These functions require final ledgers and the complete development grid.
  _, sealed23 = seal(replica_runs, manifest, DATA_SHA, selection23)
  _, sealed123 = seal([pilot_run, *replica_runs], manifest, DATA_SHA, selection123)
  print(json.dumps({'event': 'replication_development_choices_sealed',
                    'new_selected': sealed23['selected_checkpoint_files'],
                    'test_tokens_read': False}), flush=True)
  command = [
    '--selection', str(selection23 / 'selection.json'),
    '--expected-selection-sha256', sealed23['selection_file_sha256'],
    '--data-manifest', str(manifest), '--expected-data-manifest-sha256', DATA_SHA,
    '--test-jsonl', str(root / 'fresh-data-v1/test.jsonl'), '--expected-test-jsonl-sha256', TEST_SHA,
    '--expectations', str(release / 'expectations/production-expectations-v2.json'),
    '--expected-expectations-sha256', EXPECTATIONS_SHA,
    '--backbone-checkpoint', str(release / 'inputs/mdlm-owt-backbone-schema-v2.pt'),
    '--output-dir', str(test23),
  ]
  for path in sorted({item['path'] for item in sealed23['selected_checkpoint_files']}):
    command.extend(['--selected-checkpoint', path])
  evaluate(command)
  summary, lineage = aggregate(
    selection123 / 'selection-manifest.json',
    file_sha256(selection123 / 'selection-manifest.json'), manifest, DATA_SHA,
    [
      [root / 'fresh-selection-v1/selection-manifest.json', PILOT_MANIFEST_SHA,
       root / 'fresh-test-v1', PILOT_TEST_SHA],
      [selection23 / 'selection-manifest.json', file_sha256(selection23 / 'selection-manifest.json'),
       test23, file_sha256(test23 / 'results.json')],
    ], test123)
  print(json.dumps({'event': 'replication_test_complete', 'output_dir': str(test123),
                    'training_seeds': lineage['training_seeds'], 'records': lineage['records'],
                    'overall': summary['overall']}), flush=True)


if __name__ == '__main__':
  main()
