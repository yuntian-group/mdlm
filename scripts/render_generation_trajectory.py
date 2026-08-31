#!/usr/bin/env python3
"""Render an outcome-independent exact generation trajectory for the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


EXPECTED_MODES = ('structured_marginal', 'structured_joint')
DISPLAY_CALLS = (0, 16, 32, 48, 63)
ROW_SELECTION_POLICY = 'first_row_of_hash_min_full_source_batch_v1'


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--trajectory', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--tokenizer', default='gpt2-large')
  parser.add_argument('--tokenizer-revision', required=True)
  parser.add_argument('--batch-row-index', type=int, default=0)
  parser.add_argument('--local-files-only', action='store_true')
  return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
  return hashlib.sha256(json.dumps(
    payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _validate_artifact(payload: Mapping[str, Any], row_index: int) -> None:
  if payload.get('artifact') != 'exact_generation_trajectory_replay':
    raise ValueError('input is not an exact trajectory replay artifact')
  selection = payload['selection']
  if selection.get('outcome_independent') is not True:
    raise ValueError('trajectory source selection is not outcome-independent')
  if selection.get('policy') != 'hash_min_full_source_batch_v1':
    raise ValueError('trajectory source selection policy is unexpected')
  if tuple(payload.get('modes', ())) != EXPECTED_MODES:
    raise ValueError('trajectory modes are unexpected')
  if not 0 <= row_index < len(selection['sample_indices']):
    raise ValueError('batch row index lies outside the selected batch')
  for mode in EXPECTED_MODES:
    trajectory = payload['trajectories'][mode]
    if trajectory.get('final_token_ids_match_expected') is not True:
      raise ValueError(f'{mode} final raw token IDs do not match source')
    if trajectory.get('measured_nfe') != 63:
      raise ValueError(f'{mode} measured NFE is not the expected 63')
    if trajectory.get('requested_nfe_budget') != 64:
      raise ValueError(f'{mode} requested NFE budget is not 64')


def _selected_snapshots(
    trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
  selected = []
  for call_index in DISPLAY_CALLS:
    candidates = [
      snapshot for snapshot in trajectory['snapshots']
      if snapshot['model_call_index'] == call_index]
    if not candidates:
      raise ValueError(f'missing required model-call snapshot {call_index}')
    if call_index == 63:
      candidates.sort(key=lambda item: item['stage'] != 'final')
    selected.append(candidates[0])
  return selected


def _active_values(values: Sequence[Any], active_mask: Sequence[bool]) -> list[Any]:
  return [value for value, active in zip(values, active_mask) if active]


def _decode_segments(
    tokenizer: Any,
    token_ids: Sequence[int],
    unresolved: Sequence[bool],
) -> str:
  """Decode consecutive resolved runs and mark unrevealed runs explicitly."""
  pieces = []
  index = 0
  while index < len(token_ids):
    stop = index + 1
    while stop < len(token_ids) and unresolved[stop] == unresolved[index]:
      stop += 1
    if unresolved[index]:
      count = stop - index
      pieces.append(f' ⟦{count} masked⟧ ')
    else:
      decoded = tokenizer.decode(
        list(token_ids[index:stop]),
        clean_up_tokenization_spaces=False,
        skip_special_tokens=False)
      pieces.append(decoded.replace('\n', '↵').replace('\t', '⇥'))
    index = stop
  return ''.join(pieces).strip()


def _wrap(text: str, width: int = 59) -> str:
  return '\n'.join(textwrap.wrap(
    text, width=width, break_long_words=False, break_on_hyphens=False,
    replace_whitespace=False, drop_whitespace=True))


def render(
    payload: Mapping[str, Any],
    tokenizer: Any,
    output: Path,
    row_index: int,
    trajectory_file_sha256: str,
) -> dict[str, Any]:
  _validate_artifact(payload, row_index)
  output.parent.mkdir(parents=True, exist_ok=True)
  colors = {
    'structured_marginal': '#35618f',
    'structured_joint': '#ad542f',
  }
  titles = {
    'structured_marginal': 'Marginal sampling',
    'structured_joint': 'Joint forest sampling',
  }
  fig, axes = plt.subplots(
    len(DISPLAY_CALLS), len(EXPECTED_MODES),
    figsize=(7.2, 8.1), squeeze=False)
  panel_records = []
  for column, mode in enumerate(EXPECTED_MODES):
    trajectory = payload['trajectories'][mode]
    source_record = payload['source_records'][mode][row_index]
    active_mask = source_record['active_mask']
    active_count = sum(active_mask)
    for row, snapshot in enumerate(_selected_snapshots(trajectory)):
      ax = axes[row][column]
      active_ids = _active_values(snapshot['token_ids'][row_index], active_mask)
      unresolved = _active_values(
        snapshot['unresolved_active_mask'][row_index], active_mask)
      unresolved_count = sum(unresolved)
      resolved_percent = 100.0 * (active_count - unresolved_count) / active_count
      display = _decode_segments(tokenizer, active_ids, unresolved)
      ax.set_facecolor('#f7f8fa')
      for spine in ax.spines.values():
        spine.set_edgecolor('#d8dce2')
      ax.set_xticks([])
      ax.set_yticks([])
      if row == 0:
        ax.set_title(titles[mode], color=colors[mode], fontsize=10.5,
                     fontweight='bold', pad=7)
      label = (
        f"call {snapshot['model_call_index']}  |  "
        f'{resolved_percent:.0f}% revealed')
      if snapshot['stage'] == 'final':
        label += '  |  final'
      ax.text(0.025, 0.83, label, transform=ax.transAxes, fontsize=7.3,
              fontweight='bold', color=colors[mode], va='top')
      ax.text(0.025, 0.62, _wrap(display), transform=ax.transAxes,
              fontsize=7.15, family='DejaVu Sans Mono', color='#20242a',
              va='top', linespacing=1.28)
      panel_records.append({
        'mode': mode,
        'model_call_index': snapshot['model_call_index'],
        'stage': snapshot['stage'],
        'active_token_count': active_count,
        'unresolved_active_count': unresolved_count,
        'resolved_percent': resolved_percent,
      })

  batch_row = payload['trajectories'][EXPECTED_MODES[0]]['batch_order'][row_index]
  fig.suptitle(
    'Exact masked-span generation trajectory (requested NFE 64; measured 63)',
    fontsize=11.3, fontweight='bold', y=0.988)
  fig.text(
    0.5, 0.013,
    'Illustrative example selected before observing outcomes; '
    f"sample {batch_row['sample_index']}.  ⟦k masked⟧ denotes k unresolved tokens.",
    ha='center', va='bottom', fontsize=7.2, color='#4b5058')
  fig.subplots_adjust(left=0.045, right=0.985, bottom=0.053, top=0.948,
                      wspace=0.08, hspace=0.20)
  fig.savefig(output, bbox_inches='tight')
  plt.close(fig)

  provenance = {
    'schema_version': 1,
    'artifact': 'generation_trajectory_figure_provenance',
    'figure_filename': output.name,
    'trajectory_input_file_sha256': trajectory_file_sha256,
    'trajectory_artifact_sha256': payload['artifact_sha256'],
    'source_selection_policy': payload['selection']['policy'],
    'source_selection_sha256': payload['selection']['selection_sha256'],
    'source_selection_outcome_independent': True,
    'row_selection_policy': ROW_SELECTION_POLICY,
    'batch_row_index': row_index,
    'sample_index': batch_row['sample_index'],
    'pair_key': batch_row['pair_key'],
    'requested_nfe_budget': payload['nfe_budget'],
    'measured_nfe': {
      mode: payload['trajectories'][mode]['measured_nfe']
      for mode in EXPECTED_MODES
    },
    'final_raw_token_ids_match_source': {
      mode: payload['trajectories'][mode]['final_token_ids_match_expected']
      for mode in EXPECTED_MODES
    },
    'panels': panel_records,
  }
  provenance['provenance_sha256'] = _canonical_sha256(provenance)
  sidecar = output.with_suffix(output.suffix + '.provenance.json')
  sidecar.write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  return provenance


def main(argv=None) -> int:
  args = _parse_args(argv)
  payload = json.loads(args.trajectory.read_text(encoding='utf-8'))
  tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer,
    revision=args.tokenizer_revision,
    local_files_only=args.local_files_only)
  provenance = render(
    payload, tokenizer, args.output, args.batch_row_index,
    _sha256_file(args.trajectory))
  print(json.dumps({
    'event': 'generation_trajectory_figure_rendered',
    'output': str(args.output),
    'output_sha256': _sha256_file(args.output),
    'provenance_sha256': provenance['provenance_sha256'],
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
