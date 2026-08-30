"""Streaming, per-window conditional-denoising evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


RECORD_SCHEMA_VERSION = 1
RECORD_BASENAME = 'conditional_denoising_records'


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _as_cpu_list(value, *, name: str, batch_size: int):
  if torch.is_tensor(value):
    result = value.detach().cpu().tolist()
  else:
    result = list(value)
  if len(result) != batch_size:
    raise ValueError(
      f'{name} has {len(result)} values for batch size {batch_size}')
  return result


class ConditionalDenoisingRecordWriter:
  """Spool one JSON object per document-local window without retaining RAM."""

  def __init__(
      self,
      *,
      output_dir: str,
      rank: int,
      metadata: Mapping[str, Any]):
    if '://' in output_dir:
      raise ValueError(
        'conditional record output must be a local/shared filesystem path')
    self.output_dir = Path(output_dir).expanduser().resolve()
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.rank = int(rank)
    if self.rank < 0:
      raise ValueError('rank must be non-negative')
    self.metadata = dict(metadata)
    required = (
      'protocol_id', 'job_id', 'arm', 'train_seed', 'corruption_seed',
      'dataset', 'dataset_revision', 'mask_rate', 'candidate_k')
    missing = [
      field for field in required
      if self.metadata.get(field) in (None, '')]
    if missing:
      raise ValueError(
        f'conditional record metadata is missing: {missing}')
    self.spool_path = self.output_dir / (
      f'{RECORD_BASENAME}.rank{self.rank}.spool.jsonl')
    self.final_path = self.output_dir / (
      f'{RECORD_BASENAME}.rank{self.rank}.jsonl')
    self._handle = self.spool_path.open('w', encoding='utf-8')
    self.num_records = 0
    self.total_masked_tokens = 0

  def close(self) -> None:
    if not self._handle.closed:
      self._handle.close()

  def append(
      self,
      *,
      batch: Mapping[str, Any],
      metrics: Mapping[str, torch.Tensor],
      batch_index: int) -> None:
    required_batch = (
      'source_document_index', 'source_document_sha256',
      'source_chunk_index')
    missing_batch = [field for field in required_batch if field not in batch]
    if missing_batch:
      raise ValueError(
        'conditional records require document-local dataset metadata; '
        f'missing {missing_batch}')
    required_metrics = (
      'nll_sum', 'active_tokens', 'candidate_hits',
      'retained_mass_sum')
    missing_metrics = [
      field for field in required_metrics if field not in metrics]
    if missing_metrics:
      raise ValueError(
        f'conditional record metrics missing {missing_metrics}')

    batch_size = int(batch['input_ids'].shape[0])
    document_indices = _as_cpu_list(
      batch['source_document_index'], name='source_document_index',
      batch_size=batch_size)
    document_hashes = _as_cpu_list(
      batch['source_document_sha256'], name='source_document_sha256',
      batch_size=batch_size)
    chunk_indices = _as_cpu_list(
      batch['source_chunk_index'], name='source_chunk_index',
      batch_size=batch_size)
    values = {
      name: _as_cpu_list(value, name=name, batch_size=batch_size)
      for name, value in metrics.items()
      if name in required_metrics
    }

    for example_index in range(batch_size):
      masked_tokens = int(values['active_tokens'][example_index])
      row = {
        'schema_version': RECORD_SCHEMA_VERSION,
        **self.metadata,
        'rank': self.rank,
        'batch_index': int(batch_index),
        'example_index': example_index,
        'document_id': (
          f'{self.metadata["dataset"]}:'
          f'{int(document_indices[example_index])}'),
        'document_index': int(document_indices[example_index]),
        'document_sha256': str(document_hashes[example_index]),
        'chunk_index': int(chunk_indices[example_index]),
        'nll_sum': float(values['nll_sum'][example_index]),
        'masked_tokens': masked_tokens,
        'candidate_hits': int(values['candidate_hits'][example_index]),
        'retained_mass_sum': float(
          values['retained_mass_sum'][example_index]),
      }
      self._handle.write(json.dumps(
        row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
      self.num_records += 1
      self.total_masked_tokens += masked_tokens
    self._handle.flush()

  def finalize(self, *, pairing_digest_sha256: str) -> dict[str, Any]:
    if (not isinstance(pairing_digest_sha256, str)
        or len(pairing_digest_sha256) != 64
        or any(character not in '0123456789abcdef'
               for character in pairing_digest_sha256)):
      raise ValueError('pairing_digest_sha256 must be lowercase SHA256')
    self.close()
    written = 0
    with self.spool_path.open(encoding='utf-8') as source, \
        self.final_path.open('w', encoding='utf-8') as destination:
      for line_number, line in enumerate(source, start=1):
        row = json.loads(line)
        if not isinstance(row, dict):
          raise ValueError(
            f'{self.spool_path}:{line_number} is not a JSON object')
        row['pairing_digest_sha256'] = pairing_digest_sha256
        destination.write(json.dumps(
          row, sort_keys=True, separators=(',', ':'), allow_nan=False)
          + '\n')
        written += 1
    if written != self.num_records:
      raise RuntimeError(
        f'conditional record count changed during finalization: '
        f'{self.num_records} versus {written}')
    return {
      'rank': self.rank,
      'path': self.final_path.name,
      'sha256': _sha256_file(self.final_path),
      'num_records': self.num_records,
      'total_masked_tokens': self.total_masked_tokens,
      'pairing_digest_sha256': pairing_digest_sha256,
    }


def write_record_manifest(
    *,
    output_dir: str,
    metadata: Mapping[str, Any],
    rank_summaries: list[Mapping[str, Any]],
    pairing_digest: Mapping[str, Any]) -> str:
  output_path = Path(output_dir).expanduser().resolve()
  ordered = sorted(
    (dict(summary) for summary in rank_summaries),
    key=lambda summary: summary['rank'])
  expected_ranks = list(range(len(ordered)))
  if [summary['rank'] for summary in ordered] != expected_ranks:
    raise ValueError('record summaries must cover contiguous ranks')
  payload = {
    'schema_version': RECORD_SCHEMA_VERSION,
    'artifact': 'conditional_denoising_record_manifest',
    'metadata': dict(metadata),
    'pairing_digest': dict(pairing_digest),
    'rank_files': ordered,
    'num_records': sum(item['num_records'] for item in ordered),
    'total_masked_tokens': sum(
      item['total_masked_tokens'] for item in ordered),
  }
  serialized = json.dumps(
    payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
  path = output_path / f'{RECORD_BASENAME}.manifest.json'
  path.write_text(serialized, encoding='utf-8')
  return str(path)
