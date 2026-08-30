#!/usr/bin/env python3
"""Aggregate paired per-window records with train-seed/document bootstrap.

Corruption replications are averaged within each source document. Chunks from
the same source document are first combined by masked-token count, so long
documents do not become pseudo-replicates. The bootstrap then resamples
training seeds, resamples source documents inside each sampled training seed,
and equal-weights the predeclared dataset/mask-rate strata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from data_provenance import canonical_sha256  # noqa: E402
from scripts.compile_experiment_matrix import (  # noqa: E402
  DEFAULT_MANIFEST,
  load_and_validate_manifest,
  sha256_file,
)
from scripts.run_compiled_job import (  # noqa: E402
  _load_plan,
  _validate_repository_checkout,
  _validated_marker,
)


ROW_FIELDS = {
  'schema_version', 'protocol_id', 'job_id', 'arm', 'train_seed',
  'corruption_seed', 'dataset', 'dataset_revision', 'mask_rate',
  'candidate_k', 'rank', 'batch_index', 'example_index', 'document_id',
  'document_index', 'document_sha256', 'chunk_index', 'nll_sum',
  'masked_tokens', 'candidate_hits', 'retained_mass_sum',
  'pairing_digest_sha256',
}


def _lower_hex(value: object, length: int, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != length
      or any(character not in '0123456789abcdef' for character in value)):
    raise ValueError(
      f'{context} must be {length} lowercase hexadecimal digits')
  return value


def _nonnegative_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f'{context} must be a non-negative integer')
  return value


def _positive_int(value: object, *, context: str) -> int:
  result = _nonnegative_int(value, context=context)
  if result == 0:
    raise ValueError(f'{context} must be positive')
  return result


def _finite(value: object, *, context: str, minimum: float | None = None) -> float:
  if (not isinstance(value, (int, float)) or isinstance(value, bool)
      or not math.isfinite(float(value))):
    raise ValueError(f'{context} must be finite')
  result = float(value)
  if minimum is not None and result < minimum:
    raise ValueError(f'{context} must be >= {minimum}')
  return result


def validate_record(
    row: object,
    *,
    source: str = 'record',
) -> dict[str, Any]:
  if not isinstance(row, Mapping):
    raise TypeError(f'{source} must be a JSON object')
  if set(row) != ROW_FIELDS:
    raise ValueError(
      f'{source} schema mismatch: missing={sorted(ROW_FIELDS - set(row))}, '
      f'unknown={sorted(set(row) - ROW_FIELDS)}')
  result = dict(row)
  if result['schema_version'] != 1:
    raise ValueError(f'{source} schema_version must equal 1')
  for field in ('protocol_id', 'job_id', 'arm', 'dataset', 'document_id'):
    if not isinstance(result[field], str) or not result[field]:
      raise ValueError(f'{source}.{field} must be non-empty')
  result['train_seed'] = _nonnegative_int(
    result['train_seed'], context=f'{source}.train_seed')
  result['corruption_seed'] = _nonnegative_int(
    result['corruption_seed'], context=f'{source}.corruption_seed')
  result['candidate_k'] = _positive_int(
    result['candidate_k'], context=f'{source}.candidate_k')
  for field in (
      'rank', 'batch_index', 'example_index', 'document_index',
      'chunk_index'):
    result[field] = _nonnegative_int(
      result[field], context=f'{source}.{field}')
  result['masked_tokens'] = _positive_int(
    result['masked_tokens'], context=f'{source}.masked_tokens')
  result['candidate_hits'] = _nonnegative_int(
    result['candidate_hits'], context=f'{source}.candidate_hits')
  if result['candidate_hits'] > result['masked_tokens']:
    raise ValueError(f'{source}.candidate_hits exceeds masked_tokens')
  result['mask_rate'] = _finite(
    result['mask_rate'], context=f'{source}.mask_rate')
  if not 0.0 < result['mask_rate'] < 1.0:
    raise ValueError(f'{source}.mask_rate must be in (0,1)')
  result['nll_sum'] = _finite(
    result['nll_sum'], context=f'{source}.nll_sum', minimum=0.0)
  result['retained_mass_sum'] = _finite(
    result['retained_mass_sum'],
    context=f'{source}.retained_mass_sum', minimum=0.0)
  if result['retained_mass_sum'] > result['masked_tokens'] + 1e-5:
    raise ValueError(f'{source}.retained_mass_sum exceeds masked_tokens')
  _lower_hex(
    result['dataset_revision'], 40,
    context=f'{source}.dataset_revision')
  _lower_hex(
    result['document_sha256'], 64,
    context=f'{source}.document_sha256')
  _lower_hex(
    result['pairing_digest_sha256'], 64,
    context=f'{source}.pairing_digest_sha256')
  return result


def _load_record_bundle(
    manifest_path: Path,
    *,
    expected_metadata: Mapping[str, Any],
    expected_pairing_digest: Mapping[str, Any],
    expected_num_records: int,
) -> list[dict[str, Any]]:
  manifest_path = manifest_path.expanduser().resolve()
  with manifest_path.open() as handle:
    payload = json.load(handle)
  expected_manifest_fields = {
    'schema_version', 'artifact', 'metadata', 'pairing_digest',
    'rank_files', 'num_records', 'total_masked_tokens',
  }
  if not isinstance(payload, dict) or set(payload) != expected_manifest_fields:
    raise ValueError(f'invalid conditional record manifest: {manifest_path}')
  if (payload['schema_version'] != 1
      or payload['artifact'] != 'conditional_denoising_record_manifest'):
    raise ValueError(f'invalid record manifest identity: {manifest_path}')
  if payload['metadata'] != dict(expected_metadata):
    raise ValueError(
      f'record manifest metadata differs from compiled job: {manifest_path}')
  pairing_digest = payload['pairing_digest']
  if not isinstance(pairing_digest, Mapping):
    raise ValueError(f'invalid pairing digest in {manifest_path}')
  if pairing_digest != dict(expected_pairing_digest):
    raise ValueError(
      f'record manifest pairing payload differs from the committed '
      f'validation digest: {manifest_path}')
  pairing_sha = _lower_hex(
    pairing_digest.get('sha256'), 64,
    context=f'{manifest_path}.pairing_digest.sha256')
  rank_files = payload['rank_files']
  if not isinstance(rank_files, list) or not rank_files:
    raise ValueError(f'{manifest_path} has no rank files')
  if [item.get('rank') for item in rank_files] != list(range(len(rank_files))):
    raise ValueError(f'{manifest_path} rank files are not contiguous')

  rows = []
  total_masked = 0
  for rank_summary in rank_files:
    expected_rank_fields = {
      'rank', 'path', 'sha256', 'num_records', 'total_masked_tokens',
      'pairing_digest_sha256',
    }
    if not isinstance(rank_summary, Mapping) \
        or set(rank_summary) != expected_rank_fields:
      raise ValueError(f'invalid rank summary in {manifest_path}')
    if rank_summary['pairing_digest_sha256'] != pairing_sha:
      raise ValueError(f'rank pairing digest mismatch in {manifest_path}')
    relative = Path(rank_summary['path'])
    if relative.is_absolute() or '..' in relative.parts:
      raise ValueError(f'unsafe rank-file path in {manifest_path}')
    rank_path = (manifest_path.parent / relative).resolve()
    try:
      rank_path.relative_to(manifest_path.parent)
    except ValueError as error:
      raise ValueError(f'rank file escapes bundle: {rank_path}') from error
    if not rank_path.is_file():
      raise FileNotFoundError(rank_path)
    expected_sha = _lower_hex(
      rank_summary['sha256'], 64, context=f'{rank_path}.sha256')
    if sha256_file(rank_path) != expected_sha:
      raise ValueError(f'rank file SHA256 mismatch: {rank_path}')
    rank_rows = []
    with rank_path.open() as handle:
      for line_number, line in enumerate(handle, start=1):
        if not line.strip():
          raise ValueError(f'{rank_path}:{line_number} is blank')
        record = validate_record(
          json.loads(line), source=f'{rank_path}:{line_number}')
        if record['rank'] != rank_summary['rank']:
          raise ValueError(f'{rank_path}:{line_number} rank mismatch')
        if record['pairing_digest_sha256'] != pairing_sha:
          raise ValueError(f'{rank_path}:{line_number} pairing mismatch')
        for field, expected in expected_metadata.items():
          if record[field] != expected:
            raise ValueError(
              f'{rank_path}:{line_number} {field}={record[field]!r}; '
              f'expected {expected!r}')
        rank_rows.append(record)
    if len(rank_rows) != rank_summary['num_records']:
      raise ValueError(f'{rank_path} record count mismatch')
    masked = sum(row['masked_tokens'] for row in rank_rows)
    if masked != rank_summary['total_masked_tokens']:
      raise ValueError(f'{rank_path} masked-token count mismatch')
    rows.extend(rank_rows)
    total_masked += masked

  if len(rows) != payload['num_records'] \
      or total_masked != payload['total_masked_tokens']:
    raise ValueError(f'{manifest_path} aggregate counts are inconsistent')
  if len(rows) != expected_num_records:
    raise ValueError(
      f'{manifest_path} has {len(rows)} windows; frozen job requires '
      f'{expected_num_records}')
  return rows


def _validate_dataset_provenance(
    path: Path,
    *,
    require_disjoint_proof: bool,
) -> dict[str, Any]:
  with path.open() as handle:
    payload = json.load(handle)
  if not isinstance(payload, dict):
    raise ValueError(f'{path} must contain a JSON object')
  committed_hash = payload.get('manifest_sha256')
  body = dict(payload)
  body.pop('manifest_sha256', None)
  if committed_hash != canonical_sha256(body):
    raise ValueError(f'dataset provenance hash mismatch: {path}')
  if payload.get('schema_version') != 1 \
      or payload.get('artifact') != 'pinned_text_dataset_provenance':
    raise ValueError(f'invalid dataset provenance identity: {path}')
  specification = payload.get('specification')
  if not isinstance(specification, Mapping):
    raise ValueError(f'invalid dataset provenance specification: {path}')
  if payload.get('specification_sha256') != canonical_sha256(specification):
    raise ValueError(f'dataset specification hash mismatch: {path}')
  proof = specification.get('disjoint_window_proof')
  if require_disjoint_proof:
    if not isinstance(proof, Mapping) or proof.get('overlap_num_rows') != 0:
      raise ValueError(f'{path} lacks a zero-overlap data-window proof')
    proof_body = dict(proof)
    proof_hash = proof_body.pop('proof_sha256', None)
    if proof_hash != canonical_sha256(proof_body):
      raise ValueError(f'{path} disjoint-window proof hash mismatch')
  return payload


def _validate_provenance_matches_data_config(
    provenance: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> None:
  specification = provenance['specification']
  expected = {
    'logical_dataset_name': data_config['valid'],
    'dataset_name_or_path': data_config['valid_dataset_name_or_path'],
    'dataset_config_name': data_config['valid_dataset_config_name'],
    'source_split': data_config['valid_source_split'],
    'source_revision': data_config['valid_revision'],
    'source_num_rows': data_config['valid_expected_source_num_rows'],
    'source_window': data_config['valid_source_window'],
    'text_field': data_config['valid_text_field'],
    'document_boundary_mode': data_config[
      'valid_document_boundary_mode'],
    'trust_remote_code': data_config['valid_trust_remote_code'],
    'tokenizer_name_or_path': data_config['tokenizer_name_or_path'],
    'tokenizer_revision': data_config['tokenizer_revision'],
  }
  mismatches = {
    field: {'expected': value, 'observed': specification.get(field)}
    for field, value in expected.items()
    if specification.get(field) != value}
  if mismatches:
    raise ValueError(
      f'dataset provenance differs from compiled data config: {mismatches}')


def load_plan_records(
    plan_dir: Path,
    *,
    manifest_path: Path,
    suite_name: str,
    comparison_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  plan, jobs = _load_plan(plan_dir)
  _validate_repository_checkout(plan)
  manifest_path = manifest_path.expanduser().resolve()
  if sha256_file(manifest_path) != plan['source_manifest_sha256']:
    raise ValueError('protocol manifest SHA256 differs from compiled plan')
  manifest = load_and_validate_manifest(manifest_path)
  if suite_name not in plan['selected_suites']:
    raise ValueError(f'suite {suite_name} is not in the compiled plan')
  comparison = manifest['evaluation']['comparisons'].get(comparison_name)
  if comparison is None:
    raise ValueError(f'unknown comparison: {comparison_name}')
  suite = manifest['suites'][suite_name]
  arms = {comparison['baseline'], comparison['treatment']}
  if not arms.issubset(set(suite['controls'])):
    raise ValueError(
      f'comparison {comparison_name} arms are not both in suite '
      f'{suite_name}')
  selected_jobs = [
    job for job in jobs.values()
    if (job['kind'] == 'eval' and suite_name in job['suites']
        and job['identity']['control'] in arms)]
  if not selected_jobs:
    raise ValueError('compiled plan contains no selected evaluation jobs')

  expected_cells = {
    (arm, train_seed, dataset, corruption_seed, float(mask_rate), candidate_k)
    for arm in arms
    for train_seed in suite['train_seeds']
    for dataset in suite['datasets']
    for corruption_seed in suite['corruption_seeds']
    for mask_rate in suite['mask_rates']
    for candidate_k in suite['candidate_ks']}
  observed_cells = {
    (job['identity']['control'], job['identity']['train_seed'],
     job['identity']['dataset'], job['identity']['corruption_seed'],
     float(job['identity']['mask_rate']), job['identity']['candidate_k'])
    for job in selected_jobs}
  if observed_cells != expected_cells:
    raise ValueError(
      'compiled evaluation grid is incomplete for the requested suite and '
      'comparison')

  validated_dependencies: set[str] = set()

  def validate_dependency_tree(job_id: str) -> None:
    if job_id in validated_dependencies:
      return
    dependency_job = jobs[job_id]
    for dependency_id in dependency_job['dependencies']:
      validate_dependency_tree(dependency_id)
    _validated_marker(dependency_job, required=True)
    validated_dependencies.add(job_id)

  rows = []
  for job in selected_jobs:
    validate_dependency_tree(job['job_id'])
    marker = _validated_marker(job, required=True)
    assert marker is not None
    outputs = {item['name']: item for item in marker['outputs']}
    for required in (
        'conditional_record_manifest', 'conditional_records',
        'dataset_provenance', 'pairing_digest'):
      if required not in outputs:
        raise ValueError(
          f'completed job {job["job_id"]} lacks {required}')
    run_dir = Path(marker['run_dir']).resolve()
    record_manifest_path = (
      run_dir / outputs['conditional_record_manifest']['relative_path'])
    provenance_path = (
      run_dir / outputs['dataset_provenance']['relative_path'])
    pairing_path = (
      run_dir / outputs['pairing_digest']['relative_path'])
    with pairing_path.open() as handle:
      pairing_payload = json.load(handle)
    if not isinstance(pairing_payload, dict):
      raise ValueError(f'{pairing_path} must contain a JSON object')
    data_config_path = (
      REPO_ROOT / 'configs/data'
      / f'{manifest["datasets"][job["identity"]["dataset"]]["data_config"]}.yaml')
    import yaml
    with data_config_path.open() as handle:
      data_config = yaml.safe_load(handle)
    provenance = _validate_dataset_provenance(
      provenance_path,
      require_disjoint_proof=bool(
        job['identity']['require_disjoint_training_proof']))
    _validate_provenance_matches_data_config(provenance, data_config)
    expected_metadata = {
      'protocol_id': manifest['protocol_id'],
      'job_id': job['job_id'],
      'arm': job['identity']['control'],
      'train_seed': job['identity']['train_seed'],
      'corruption_seed': job['identity']['corruption_seed'],
      'dataset': data_config['valid'],
      'dataset_revision': data_config['valid_revision'],
      'mask_rate': float(job['identity']['mask_rate']),
      'candidate_k': job['identity']['candidate_k'],
    }
    target_windows = (
      job['identity']['validation_batches']
      * manifest['evaluation']['batch_size'])
    available_windows = provenance.get('observed', {}).get(
      'processed_num_sequences')
    if (not isinstance(available_windows, int)
        or isinstance(available_windows, bool) or available_windows <= 0):
      raise ValueError(
        f'{provenance_path} does not commit a positive processed window '
        'count')
    # Use the frozen target when feasible and the complete processed split
    # otherwise (notably WikiText-103 validation is smaller than 2,000
    # document-local 1,024-token windows).
    expected_windows = min(target_windows, available_windows)
    rows.extend(_load_record_bundle(
      record_manifest_path,
      expected_metadata=expected_metadata,
      expected_pairing_digest=pairing_payload,
      expected_num_records=expected_windows))
  return rows, {
    'plan': plan,
    'manifest': manifest,
    'suite': suite,
    'comparison': comparison,
    'comparison_name': comparison_name,
  }


def _paired_document_matrices(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline_arm: str,
    treatment_arm: str,
) -> tuple[dict[tuple[str, str, float, int], dict[int, np.ndarray]],
           dict[str, Any]]:
  records = [validate_record(record) for record in records]
  by_window = {}
  for row in records:
    key = (
      row['arm'], row['train_seed'], row['corruption_seed'],
      row['dataset'], row['dataset_revision'], row['mask_rate'],
      row['candidate_k'], row['document_id'], row['document_index'],
      row['document_sha256'], row['chunk_index'])
    if key in by_window:
      raise ValueError(f'duplicate conditional record key: {key}')
    by_window[key] = row
  observed_arms = {row['arm'] for row in records}
  if observed_arms != {baseline_arm, treatment_arm}:
    raise ValueError(
      f'record arms must be exactly baseline/treatment; got '
      f'{sorted(observed_arms)}')

  # Whole-run cryptographic commitments must agree across arms and training
  # seeds for each dataset/mask/K/corruption cell.
  digests: dict[tuple[Any, ...], set[str]] = {}
  for row in records:
    key = (
      row['corruption_seed'], row['dataset'], row['dataset_revision'],
      row['mask_rate'], row['candidate_k'])
    digests.setdefault(key, set()).add(row['pairing_digest_sha256'])
  bad_digests = [key for key, values in digests.items() if len(values) != 1]
  if bad_digests:
    raise ValueError(
      f'pairing digests differ across arms/train seeds for '
      f'{bad_digests[0]}')

  # Sum chunks within each source document before computing per-token NLL.
  document_totals: dict[tuple[Any, ...], dict[str, float]] = {}
  document_hashes: dict[tuple[Any, ...], str] = {}
  for row in records:
    key = (
      row['arm'], row['train_seed'], row['corruption_seed'],
      row['dataset'], row['dataset_revision'], row['mask_rate'],
      row['candidate_k'], row['document_id'], row['document_index'])
    previous_hash = document_hashes.setdefault(key, row['document_sha256'])
    if previous_hash != row['document_sha256']:
      raise ValueError(f'document hash changed across chunks for {key}')
    totals = document_totals.setdefault(key, {
      'nll_sum': 0.0, 'masked_tokens': 0.0,
      'candidate_hits': 0.0, 'retained_mass_sum': 0.0,
      'num_windows': 0.0,
    })
    for field in (
        'nll_sum', 'masked_tokens', 'candidate_hits',
        'retained_mass_sum'):
      totals[field] += float(row[field])
    totals['num_windows'] += 1.0

  strata_keys = sorted({
    (row['dataset'], row['dataset_revision'], row['mask_rate'],
     row['candidate_k']) for row in records})
  train_seeds = sorted({row['train_seed'] for row in records})
  eval_seeds = sorted({row['corruption_seed'] for row in records})
  matrices = {}
  diagnostics = {}
  for stratum in strata_keys:
    dataset, revision, mask_rate, candidate_k = stratum
    window_sets = []
    document_sets = []
    for arm in (baseline_arm, treatment_arm):
      for train_seed in train_seeds:
        for eval_seed in eval_seeds:
          windows = {
            (key[-4], key[-3], key[-2], key[-1])
            for key in by_window
            if (key[0] == arm and key[1] == train_seed
                and key[2] == eval_seed and key[3] == dataset
                and key[4] == revision and key[5] == mask_rate
                and key[6] == candidate_k)}
          if not windows:
            raise ValueError(
              f'empty factorial cell for {stratum}, {arm}, '
              f'train={train_seed}, corruption={eval_seed}')
          window_sets.append(windows)
          documents = {
            (key[-2], key[-1])
            for key in document_totals
            if (key[0] == arm and key[1] == train_seed
                and key[2] == eval_seed and key[3] == dataset
                and key[4] == revision and key[5] == mask_rate
                and key[6] == candidate_k)}
          document_sets.append(documents)
    reference_windows = window_sets[0]
    if any(windows != reference_windows for windows in window_sets):
      raise ValueError(f'window identities differ within stratum {stratum}')
    reference_documents = document_sets[0]
    if any(documents != reference_documents for documents in document_sets):
      raise ValueError(f'document sets differ within stratum {stratum}')
    ordered_documents = sorted(reference_documents)
    per_train = {}
    for train_seed in train_seeds:
      values = []
      for document_id, document_index in ordered_documents:
        replicated = []
        for eval_seed in eval_seeds:
          pair = []
          for arm in (baseline_arm, treatment_arm):
            key = (
              arm, train_seed, eval_seed, dataset, revision, mask_rate,
              candidate_k, document_id, document_index)
            totals = document_totals[key]
            pair.append(totals['nll_sum'] / totals['masked_tokens'])
          replicated.append(pair[0] - pair[1])
        values.append(math.fsum(replicated) / len(replicated))
      per_train[train_seed] = np.asarray(values, dtype=np.float64)
    matrices[stratum] = per_train

    arm_diagnostics = {}
    for arm in (baseline_arm, treatment_arm):
      selected = [
        totals for key, totals in document_totals.items()
        if (key[0] == arm and key[3] == dataset and key[4] == revision
            and key[5] == mask_rate and key[6] == candidate_k)]
      masked = math.fsum(item['masked_tokens'] for item in selected)
      arm_diagnostics[arm] = {
        'conditional_nll_per_masked_token': (
          math.fsum(item['nll_sum'] for item in selected) / masked),
        'candidate_recall': (
          math.fsum(item['candidate_hits'] for item in selected) / masked),
        'retained_unary_mass': (
          math.fsum(item['retained_mass_sum'] for item in selected)
          / masked),
      }
    diagnostics[str(stratum)] = {
      'num_documents': len(ordered_documents),
      'num_train_seeds': len(train_seeds),
      'num_corruption_seeds': len(eval_seeds),
      'arms': arm_diagnostics,
    }
  return matrices, diagnostics


def _percentile_interval(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
  tail = (1.0 - confidence_level) / 2.0
  return tuple(float(value) for value in np.quantile(
    values, [tail, 1.0 - tail], method='linear'))


def _bootstrap_collection(
    matrices: Mapping[tuple[str, str, float, int], Mapping[int, np.ndarray]],
    *,
    num_resamples: int,
    rng_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
  if not matrices:
    raise ValueError('hierarchical bootstrap has no strata')
  first = next(iter(matrices.values()))
  train_seeds = sorted(first)
  if any(sorted(per_train) != train_seeds for per_train in matrices.values()):
    raise ValueError('bootstrap strata have different training seeds')
  if num_resamples <= 0:
    raise ValueError('num_resamples must be positive')
  rng = np.random.default_rng(rng_seed)
  stratum_distributions = {
    stratum: np.empty(num_resamples, dtype=np.float64)
    for stratum in matrices}
  pooled = np.empty(num_resamples, dtype=np.float64)
  train_count = len(train_seeds)
  chunk_size = 256
  for start in range(0, num_resamples, chunk_size):
    stop = min(start + chunk_size, num_resamples)
    count = stop - start
    train_indices = rng.integers(
      0, train_count, size=(count, train_count))
    pooled_chunk = np.zeros(count, dtype=np.float64)
    for stratum, per_train in matrices.items():
      matrix = np.stack([per_train[seed] for seed in train_seeds])
      document_count = matrix.shape[1]
      document_indices = rng.integers(
        0, document_count,
        size=(count, train_count, document_count))
      sampled = matrix[
        train_indices[:, :, None], document_indices]
      means = sampled.mean(axis=(1, 2))
      stratum_distributions[stratum][start:stop] = means
      pooled_chunk += means
    pooled[start:stop] = pooled_chunk / len(matrices)

  condition_results = {}
  for stratum, distribution in stratum_distributions.items():
    per_train = matrices[stratum]
    point = math.fsum(
      float(values.mean()) for values in per_train.values()) / len(per_train)
    lower, upper = _percentile_interval(distribution, confidence_level)
    dataset, revision, mask_rate, candidate_k = stratum
    key = f'{dataset}|mask={mask_rate:.6f}|k={candidate_k}'
    condition_results[key] = {
      'dataset': dataset,
      'dataset_revision': revision,
      'mask_rate': mask_rate,
      'candidate_k': candidate_k,
      'mean_improvement': point,
      'ci_lower': lower,
      'ci_upper': upper,
    }
  pooled_point = math.fsum(
    math.fsum(float(values.mean()) for values in per_train.values())
    / len(per_train)
    for per_train in matrices.values()) / len(matrices)
  pooled_lower, pooled_upper = _percentile_interval(
    pooled, confidence_level)
  return {
    'method': 'hierarchical_paired_percentile_bootstrap',
    'improvement_definition': 'baseline conditional NLL minus treatment',
    'nesting': [
      'average corruption replications within source document',
      'resample training seeds with replacement',
      'resample source documents within sampled training seed',
      'equal-weight frozen dataset x mask-rate strata',
    ],
    'num_train_seeds': len(train_seeds),
    'num_strata': len(matrices),
    'num_resamples': num_resamples,
    'rng': 'NumPy Generator(PCG64)',
    'rng_seed': rng_seed,
    'confidence_level': confidence_level,
    'pooled': {
      'mean_improvement': pooled_point,
      'ci_lower': pooled_lower,
      'ci_upper': pooled_upper,
    },
    'conditions': condition_results,
  }


def aggregate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline_arm: str,
    treatment_arm: str,
    protocol_id: str,
    suite_name: str,
    comparison_name: str,
    num_resamples: int = 20_000,
    rng_seed: int = 1701,
    confidence_level: float = 0.95,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
  matrices, diagnostics = _paired_document_matrices(
    records, baseline_arm=baseline_arm, treatment_arm=treatment_arm)
  by_k = {}
  for candidate_k in sorted({key[3] for key in matrices}):
    selected = {
      key: value for key, value in matrices.items()
      if key[3] == candidate_k}
    by_k[str(candidate_k)] = _bootstrap_collection(
      selected,
      num_resamples=num_resamples,
      rng_seed=rng_seed + candidate_k,
      confidence_level=confidence_level)
  return {
    'schema_version': 1,
    'artifact': 'hierarchical_conditional_denoising_analysis',
    'created_utc': timestamp_utc or dt.datetime.now(
      dt.timezone.utc).isoformat(),
    'protocol_id': protocol_id,
    'suite': suite_name,
    'comparison': comparison_name,
    'arms': {'baseline': baseline_arm, 'treatment': treatment_arm},
    'objective': 'conditional_denoising_nll_per_masked_token',
    'scope_note': (
      'Conditional denoising only; no diffusion ELBO, likelihood, '
      'perplexity, or generation-quality quantity is inferred.'),
    'by_candidate_k': by_k,
    'diagnostics': diagnostics,
  }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan-dir', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument('--suite', required=True)
  parser.add_argument('--comparison', required=True)
  parser.add_argument('--bootstrap-resamples', type=int)
  parser.add_argument('--bootstrap-seed', type=int)
  parser.add_argument('--confidence-level', type=float)
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args(argv)


def main() -> int:
  args = _parse_args()
  records, context = load_plan_records(
    args.plan_dir,
    manifest_path=args.manifest,
    suite_name=args.suite,
    comparison_name=args.comparison)
  analysis_cfg = context['manifest']['analysis']
  comparison = context['comparison']
  result = aggregate_records(
    records,
    baseline_arm=comparison['baseline'],
    treatment_arm=comparison['treatment'],
    protocol_id=context['manifest']['protocol_id'],
    suite_name=args.suite,
    comparison_name=args.comparison,
    num_resamples=(
      args.bootstrap_resamples or analysis_cfg['bootstrap_resamples']),
    rng_seed=(args.bootstrap_seed or analysis_cfg['bootstrap_seed']),
    confidence_level=(
      args.confidence_level or analysis_cfg['confidence_level']))
  result['compiled_plan'] = {
    'plan_id': context['plan']['plan_id'],
    'source_manifest_sha256': context['plan']['source_manifest_sha256'],
  }
  output = args.output.expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  if output.exists():
    raise FileExistsError(output)
  temporary = output.with_name(f'.{output.name}.tmp')
  temporary.write_text(json.dumps(
    result, indent=2, sort_keys=True, allow_nan=False) + '\n')
  temporary.replace(output)
  print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
