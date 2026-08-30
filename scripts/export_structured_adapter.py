#!/usr/bin/env python3
"""Export a portable structured head from a frozen-backbone checkpoint.

The training checkpoint contains both the immutable released backbone and the
small learned structured head.  This exporter fails closed on unexpected state
names, removes the redundant frozen tensors, and writes the learned tensors in
the non-executable safetensors format plus a provenance manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping

from safetensors.torch import load_file, save_file
import torch

# Keep both ``python -m scripts.export_structured_adapter`` and direct script
# invocation usable from an arbitrary working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_released_mdlm_owt import (  # noqa: E402
  RELEASE_REPOSITORY,
  RELEASE_REVISION,
  RELEASE_SHA256,
  RELEASE_SIZE_BYTES,
  RELEASE_TENSOR_COUNT,
  sha256_file,
)


BACKBONE_PREFIX = 'backbone.'
ADAPTER_PREFIX = 'structured_head.'


def _validated_state_dict(
    checkpoint: Mapping[str, object],
    *,
    expected_backbone_tensors: int | None,
) -> tuple[dict[str, torch.Tensor], int]:
  state = checkpoint.get('state_dict')
  if not isinstance(state, Mapping) or not state:
    raise ValueError('checkpoint has no non-empty state_dict mapping')
  invalid_values = [
    key for key, value in state.items() if not torch.is_tensor(value)]
  if invalid_values:
    raise TypeError(
      f'checkpoint state contains non-tensors: {invalid_values[:5]}')
  unexpected = [
    key for key in state
    if not key.startswith((BACKBONE_PREFIX, ADAPTER_PREFIX))]
  if unexpected:
    raise ValueError(
      'checkpoint state contains tensors outside backbone.* and '
      f'structured_head.*: {unexpected[:5]}')

  backbone_count = sum(key.startswith(BACKBONE_PREFIX) for key in state)
  if (expected_backbone_tensors is not None
      and backbone_count != expected_backbone_tensors):
    raise ValueError(
      f'backbone tensor-count mismatch: expected '
      f'{expected_backbone_tensors}, found {backbone_count}')
  adapter = {
    key.removeprefix(ADAPTER_PREFIX): value.detach().cpu().contiguous()
    for key, value in state.items() if key.startswith(ADAPTER_PREFIX)
  }
  if not adapter:
    raise ValueError('checkpoint contains no structured_head.* tensors')
  if any(not key for key in adapter):
    raise ValueError('invalid empty adapter tensor name after prefix removal')
  return dict(sorted(adapter.items())), backbone_count


def load_adapter_state(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, torch.Tensor]:
  """Load a prefix-stripped structured-head state with optional byte check."""
  path = path.resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  if expected_sha256 is not None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
      raise ValueError(
        f'adapter SHA256 mismatch: expected {expected_sha256}, '
        f'found {actual_sha256}')
  state = load_file(str(path), device='cpu')
  if not state:
    raise ValueError('adapter file contains no tensors')
  invalid = [
    key for key in state
    if key.startswith((BACKBONE_PREFIX, ADAPTER_PREFIX))]
  if invalid:
    raise ValueError(
      f'adapter file must use prefix-stripped keys: {invalid[:5]}')
  return dict(sorted(state.items()))


def load_adapter_into_head(
    head: torch.nn.Module,
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> None:
  """Strictly rehydrate an instantiated structured head."""
  state = load_adapter_state(path, expected_sha256=expected_sha256)
  incompatible = head.load_state_dict(state, strict=True)
  if incompatible.missing_keys or incompatible.unexpected_keys:
    raise ValueError(
      'adapter state mismatch: '
      f'missing={incompatible.missing_keys}, '
      f'unexpected={incompatible.unexpected_keys}')


def export_adapter(
    checkpoint_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_global_step: int | None = None,
    expected_backbone_tensors: int | None = RELEASE_TENSOR_COUNT,
    overwrite: bool = False,
) -> dict[str, object]:
  checkpoint_path = checkpoint_path.resolve()
  output_path = output_path.resolve()
  manifest_path = manifest_path.resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
  if output_path == manifest_path:
    raise ValueError('adapter and manifest paths must differ')
  existing = [path for path in (output_path, manifest_path) if path.exists()]
  if existing and not overwrite:
    raise FileExistsError(
      f'{existing[0]} already exists; pass --force to replace outputs')

  if (len(expected_checkpoint_sha256) != 64
      or any(character not in '0123456789abcdef'
             for character in expected_checkpoint_sha256.lower())):
    raise ValueError('expected checkpoint SHA256 must be 64 hex characters')
  source_checkpoint_sha256 = sha256_file(checkpoint_path)
  if source_checkpoint_sha256 != expected_checkpoint_sha256.lower():
    raise ValueError(
      f'source checkpoint SHA256 mismatch: expected '
      f'{expected_checkpoint_sha256.lower()}, '
      f'found {source_checkpoint_sha256}')

  # Lightning checkpoints use pickle.  Byte identity is therefore checked
  # against a caller-supplied trusted digest before any deserialization.
  checkpoint = torch.load(
    checkpoint_path, map_location='cpu', weights_only=False)
  if not isinstance(checkpoint, Mapping):
    raise ValueError('checkpoint payload is not a mapping')
  if checkpoint.get('ema') is not None:
    raise ValueError('checkpoint unexpectedly contains EMA state')
  global_step = int(checkpoint.get('global_step', -1))
  if expected_global_step is not None and global_step != expected_global_step:
    raise ValueError(
      f'global-step mismatch: expected {expected_global_step}, '
      f'found {global_step}')
  adapter, backbone_count = _validated_state_dict(
    checkpoint,
    expected_backbone_tensors=expected_backbone_tensors)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_adapter = output_path.with_name(
    f'.{output_path.name}.tmp-{os.getpid()}')
  temporary_manifest = manifest_path.with_name(
    f'.{manifest_path.name}.tmp-{os.getpid()}')
  try:
    save_file(
      adapter,
      str(temporary_adapter),
      metadata={
        'artifact_role': 'contextual_forest_structured_head',
        'source_namespace': ADAPTER_PREFIX,
        'file_namespace': 'prefix-stripped',
      })
    adapter_sha256 = sha256_file(temporary_adapter)
    tensor_schema = {
      key: {
        'shape': list(value.shape),
        'dtype': str(value.dtype),
      }
      for key, value in adapter.items()
    }
    manifest = {
      'artifact_role': 'contextual_forest_structured_adapter',
      'schema_version': 1,
      'format': 'safetensors',
      'adapter_file': output_path.name,
      'adapter_sha256': adapter_sha256,
      'adapter_size_bytes': temporary_adapter.stat().st_size,
      'adapter_tensor_count': len(adapter),
      'adapter_parameter_count': sum(
        value.numel() for value in adapter.values()),
      'adapter_tensor_bytes': sum(
        value.numel() * value.element_size() for value in adapter.values()),
      'adapter_namespace_in_source': f'{ADAPTER_PREFIX}*',
      'adapter_namespace_in_file': 'prefix-stripped',
      'tensor_schema': tensor_schema,
      'source_checkpoint_sha256': source_checkpoint_sha256,
      'source_checkpoint_size_bytes': checkpoint_path.stat().st_size,
      'source_checkpoint_global_step': global_step,
      'source_state_dict_tensor_count': backbone_count + len(adapter),
      'omitted_frozen_backbone_tensor_count': backbone_count,
      'ema_available': False,
      'ema_used': False,
      'required_loader': (
        'scripts.export_structured_adapter.load_adapter_into_head'),
      'required_loader_strict': True,
      'released_backbone': {
        'repository': RELEASE_REPOSITORY,
        'revision': RELEASE_REVISION,
        'source_sha256': RELEASE_SHA256,
        'source_size_bytes': RELEASE_SIZE_BYTES,
        'tensor_count': RELEASE_TENSOR_COUNT,
      },
    }
    temporary_manifest.write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.replace(temporary_adapter, output_path)
    os.replace(temporary_manifest, manifest_path)
  finally:
    if temporary_adapter.exists():
      temporary_adapter.unlink()
    if temporary_manifest.exists():
      temporary_manifest.unlink()
  return manifest


def _args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Export the learned structured head from a frozen checkpoint.')
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--expected-checkpoint-sha256', required=True)
  parser.add_argument('--expected-global-step', type=int)
  parser.add_argument(
    '--expected-backbone-tensors', type=int, default=RELEASE_TENSOR_COUNT)
  parser.add_argument('--force', action='store_true')
  return parser.parse_args(argv)


def main() -> int:
  args = _args()
  manifest = export_adapter(
    args.checkpoint,
    args.output,
    args.manifest,
    expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    expected_global_step=args.expected_global_step,
    expected_backbone_tensors=args.expected_backbone_tensors,
    overwrite=args.force)
  print(json.dumps(manifest, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
