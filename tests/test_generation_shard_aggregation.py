import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.generation_harness import summarize_group
from evaluation.generation_metrics import paired_token_metrics, repetition_rate
from evaluation.generation_shard_aggregation import (
  REFERENCE_LM_EXP_REL_TOL,
  _reference_lm_identity,
  _validate_reference_lm_score,
  aggregate_generation_shards,
  canonical_sha256,
  paired_bootstrap_intervals,
  pairing_digest,
)
from evaluation.generation_metrics import REFERENCE_LM_SEQUENCE_POLICY


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_specs():
  result = []
  per_prompt = {'prompt-0': 0, 'prompt-1': 0}
  for sample_index in range(4):
    prompt_id = f'prompt-{sample_index % 2}'
    replicate = per_prompt[prompt_id]
    per_prompt[prompt_id] += 1
    observed = 10 + sample_index % 2
    result.append({
      'sample_index': sample_index,
      'pair_key': f'{prompt_id}/replicate-{replicate:04d}',
      'pair_seed': 700 + sample_index,
      'prompt_id': prompt_id,
      'prompt_metadata': {'source_document': prompt_id},
      'initial_token_ids': [observed, 99, 99],
      'active_mask': [False, True, True],
      'reference_token_ids': [observed, 2, 3],
    })
  return result


def _record(
    pair,
    *,
    mode,
    budget,
    shard_index,
    global_digest,
    shard_digest,
    batch_seed,
    batch_size,
):
  if mode == 'structured_joint':
    sampled = [pair['initial_token_ids'][0], 2, 3]
  elif pair['sample_index'] % 2:
    sampled = [pair['initial_token_ids'][0], 5, 3]
  else:
    sampled = [pair['initial_token_ids'][0], 2, 4]
  active_values = sampled[1:]
  reference_active = pair['reference_token_ids'][1:]
  metrics = {
    'repetition_rate': {
      str(n): repetition_rate(active_values, n=n) for n in (1, 2, 4)
    },
    **paired_token_metrics(active_values, reference_active),
  }
  elapsed = 2.0 + 0.1 * budget + (0.2 if mode == 'structured_joint' else 0.0)
  timing = {
    'batch_seed': batch_seed,
    'batch_size': batch_size,
    'requested_nfe_budget': budget,
    'measured_nfe': budget,
    'wall_clock_seconds': elapsed,
    'active_tokens': 4,
    'active_tokens_per_second': 4 / elapsed,
    'sequence_tokens_per_second': 6 / elapsed,
    'peak_memory_bytes': 100 + budget,
    'unresolved_mask_tokens': 0,
  }
  return {
    'schema_version': 1,
    **pair,
    'sampling_mode': mode,
    'requested_nfe_budget': budget,
    'measured_nfe': budget,
    'batch_seed': timing['batch_seed'],
    'sample_token_ids': sampled,
    'sample_active_token_ids': active_values,
    'text': ' '.join(str(value) for value in sampled),
    'metrics': metrics,
    'timing': timing,
    'global_pairing_digest': global_digest,
    'shard_pairing_digest': shard_digest,
    'num_shards': 2,
    'shard_index': shard_index,
  }


def _write_shards(root: Path):
  root.mkdir(parents=True, exist_ok=True)
  pair_specs = _pair_specs()
  global_digest = pairing_digest(pair_specs)
  shard_dirs = []
  for shard_index in range(2):
    shard_dir = root / f'shard-{shard_index}'
    shard_dir.mkdir()
    shard_pairs = [
      pair for pair in pair_specs
      if pair['sample_index'] % 2 == shard_index
    ]
    batch_seed = int(canonical_sha256([
      {'pair_key': pair['pair_key'], 'pair_seed': pair['pair_seed']}
      for pair in shard_pairs
    ])[:16], 16) % (2 ** 63 - 1)
    shard_digest = pairing_digest(shard_pairs)
    records = []
    groups = []
    for mode in ('factorized', 'structured_joint'):
      for budget in (2, 4):
        group = [
          _record(
            pair, mode=mode, budget=budget, shard_index=shard_index,
            global_digest=global_digest, shard_digest=shard_digest,
            batch_seed=batch_seed, batch_size=len(shard_pairs))
          for pair in shard_pairs
        ]
        summary = summarize_group(group)
        summary['input_pairing_digest'] = shard_digest
        groups.append(summary)
        records.extend(group)

    samples_path = shard_dir / 'samples.jsonl'
    samples_path.write_text(''.join(
      json.dumps(record, sort_keys=True) + '\n' for record in records))
    summary_payload = {
      'schema_version': 1,
      'experiment': 'paired_contextual_forest_generation_pilot',
      'global_pairing_digest': global_digest,
      'input_pairing_digest': shard_digest,
      'global_num_paired_samples': 4,
      'num_paired_samples': 2,
      'shard_index': shard_index,
      'num_shards': 2,
      'groups': groups,
      'reference_lm': None,
    }
    summary_path = shard_dir / 'summary.json'
    summary_path.write_text(json.dumps(
      summary_payload, indent=2, sort_keys=True) + '\n')
    config_path = shard_dir / 'resolved_config.yaml'
    config_path.write_text('model: test\nlength: 3\n')
    manifest = {
      'schema_version': 1,
      'experiment': 'paired_contextual_forest_generation_pilot',
      'scientific_scope': 'test fixture',
      'command': ['test', '--shard-index', str(shard_index)],
      'start_time_utc': '2026-08-30T00:00:00+00:00',
      'end_time_utc': '2026-08-30T00:01:00+00:00',
      'duration_seconds': 60.0,
      'host': {
        'hostname': f'worker-{shard_index}',
        'platform': 'Linux-6.8-x86_64',
        'python': '3.10.14',
        'torch': '2.5.1+cu121',
        'cuda_runtime': '12.1',
        'device': 'cuda:0',
        'gpu': 'NVIDIA L4',
        'parameter_dtypes': ['torch.float32'],
        'precision_policy': (
          'checkpoint dtype; DIT-managed bf16 autocast on CUDA'),
        'packages': {
          'numpy': '2.1.3',
          'safetensors': '0.4.5',
          'tokenizers': '0.15.2',
          'transformers': '4.38.2',
        },
      },
      'repository': {
        'git_sha': 'a' * 40,
        'dirty': False,
        'status_porcelain': [],
        'tracked_diff_sha256': hashlib.sha256(b'').hexdigest(),
        'untracked_files': [],
        'dirty_content_sha256': hashlib.sha256(b'').hexdigest(),
      },
      'artifacts': {
        'backbone_checkpoint': {
          'path': '/models/backbone.pt',
          'sha256': 'b' * 64,
          'size_bytes': 1000,
        },
        'structured_adapter': {
          'path': '/models/adapter.safetensors',
          'sha256': 'c' * 64,
          'size_bytes': 200,
          'manifest_path': '/models/adapter.manifest.json',
          'manifest_sha256': 'e' * 64,
          'identity_sha256': 'f' * 64,
        },
      },
      'prompts': {
        'source': 'jsonl',
        'path': '/inputs/prompts.jsonl',
        'sha256': 'd' * 64,
        'num_prompt_records': 2,
      },
      'pairing': {
        'digest_algorithm': 'sha256-canonical-json-v1',
        'global_pairing_digest': global_digest,
        'shard_pairing_digest': shard_digest,
        'base_seed': 700,
        # The two modulo-assigned records form a final partial batch.
        'batch_size': 3,
        'global_num_samples': 4,
        'shard_num_samples': 2,
        'num_shards': 2,
        'shard_index': shard_index,
        'sequence_length': 3,
      },
      'spot_interruption_policy': {'resume_supported': False},
      'matrix': {
        'sampling_modes': ['factorized', 'structured_joint'],
        'nfe_budgets': [2, 4],
        'num_output_records': 8,
      },
      'outputs': {
        'samples_jsonl': {
          'path': 'samples.jsonl',
          'sha256': _sha256(samples_path),
          'num_records': 8,
        },
        'summary_json': {
          'path': 'summary.json',
          'sha256': _sha256(summary_path),
        },
        'resolved_config': {
          'path': 'resolved_config.yaml',
          'sha256': _sha256(config_path),
        },
      },
      'reference_lm': None,
    }
    (shard_dir / 'manifest.json').write_text(json.dumps(
      manifest, indent=2, sort_keys=True) + '\n')
    shard_dirs.append(shard_dir)
  return shard_dirs


def _read_manifest(shard_dir: Path):
  return json.loads((shard_dir / 'manifest.json').read_text())


def _write_manifest(shard_dir: Path, manifest):
  (shard_dir / 'manifest.json').write_text(json.dumps(
    manifest, indent=2, sort_keys=True) + '\n')


def _rewrite_samples(shard_dir: Path, records):
  samples_path = shard_dir / 'samples.jsonl'
  samples_path.write_text(''.join(
    json.dumps(record, sort_keys=True) + '\n' for record in records))
  manifest = _read_manifest(shard_dir)
  manifest['outputs']['samples_jsonl']['sha256'] = _sha256(samples_path)
  _write_manifest(shard_dir, manifest)


class GenerationShardAggregationTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.shards = _write_shards(self.root)

  def tearDown(self):
    self.temporary.cleanup()

  def test_complete_union_recomputes_metrics_and_clusters_repeated_prompts(self):
    result = aggregate_generation_shards(
      self.shards,
      bootstrap_resamples=200,
      bootstrap_seed=19,
      timestamp_utc='2026-08-30T02:00:00+00:00')

    coverage = result['coverage']
    self.assertEqual(coverage['global_num_paired_draws'], 4)
    self.assertEqual(coverage['num_unique_prompts'], 2)
    self.assertEqual(
      coverage['paired_draws_per_prompt'], {'prompt-0': 2, 'prompt-1': 2})
    self.assertEqual(coverage['verified_output_records'], 16)
    self.assertEqual(
      result['identity']['runtime']['cuda_runtime'], '12.1')
    self.assertEqual(
      result['identity']['runtime']['platform'], 'Linux-6.8-x86_64')
    self.assertEqual(
      result['identity']['runtime']['packages']['transformers'], '4.38.2')
    self.assertEqual(
      result['timing_policy']['hardware_identity']['gpu'], 'NVIDIA L4')
    self.assertEqual(len(result['groups']), 4)
    structured = next(
      group for group in result['groups']
      if group['sampling_mode'] == 'structured_joint'
      and group['requested_nfe_budget'] == 2)
    self.assertEqual(structured['reference']['token_accuracy'], 1.0)
    self.assertEqual(
      structured['timing']['inferential_status'], 'descriptive_only')
    comparison = next(
      item for item in result['comparisons']
      if item['comparison_kind'] == 'sampling_mode_at_fixed_nfe'
      and item['baseline']['requested_nfe_budget'] == 2)
    accuracy = comparison['endpoints']['reference_token_accuracy']
    self.assertAlmostEqual(
      accuracy['paired_draws']['point_estimate'], 0.5)
    self.assertEqual(accuracy['paired_draws']['num_paired_draws'], 4)
    self.assertEqual(
      accuracy['prompt_clusters']['num_prompt_clusters'], 2)
    self.assertEqual(
      accuracy['prompt_clusters']['draws_per_prompt'],
      {'prompt-0': 2, 'prompt-1': 2})

  def test_rejects_missing_or_duplicate_shard_coverage(self):
    with self.assertRaisesRegex(ValueError, 'incomplete shard coverage'):
      aggregate_generation_shards(
        self.shards[:1], bootstrap_resamples=5)
    with self.assertRaisesRegex(ValueError, 'duplicate shard path'):
      aggregate_generation_shards(
        [self.shards[0], self.shards[0]], bootstrap_resamples=5)

  def test_rejects_clean_repository_and_cross_shard_identity_drift(self):
    manifest = _read_manifest(self.shards[1])
    manifest['repository']['dirty'] = True
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'clean repository'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'second-fixture')
    manifest = _read_manifest(self.shards[1])
    manifest['artifacts']['structured_adapter']['sha256'] = 'e' * 64
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'adapter-manifest-fixture')
    manifest = _read_manifest(self.shards[1])
    manifest['artifacts']['structured_adapter']['identity_sha256'] = '0' * 64
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_runtime_drift_and_gpu_timing_pooling(self):
    manifest = _read_manifest(self.shards[1])
    manifest['host']['torch'] = '2.6.0+cu124'
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'gpu-drift-fixture')
    manifest = _read_manifest(self.shards[1])
    manifest['host']['gpu'] = 'NVIDIA A100-SXM4-80GB'
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'refusing to pool timing'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_platform_or_critical_package_runtime_drift(self):
    manifest = _read_manifest(self.shards[1])
    manifest['host']['platform'] = 'Windows-11-x86_64'
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'package-drift-fixture')
    manifest = _read_manifest(self.shards[1])
    manifest['host']['packages']['transformers'] = '4.50.0'
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_resolved_config_and_prompt_identity_drift(self):
    config_path = self.shards[1] / 'resolved_config.yaml'
    config_path.write_text('model: different\nlength: 3\n')
    manifest = _read_manifest(self.shards[1])
    manifest['outputs']['resolved_config']['sha256'] = _sha256(config_path)
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_adapter_manifest_path_is_not_part_of_cryptographic_identity(self):
    manifest = _read_manifest(self.shards[1])
    manifest['artifacts']['structured_adapter']['manifest_path'] = (
      '/a/different/mount/adapter.manifest.json')
    _write_manifest(self.shards[1], manifest)
    result = aggregate_generation_shards(
      self.shards, bootstrap_resamples=5)
    self.assertNotIn(
      'manifest_path', result['identity']['artifacts']['structured_adapter'])

  def test_rejects_even_tiny_deterministic_metric_mismatches(self):
    summary_path = self.shards[1] / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['groups'][0]['reference']['token_accuracy'] += 5e-7
    summary_path.write_text(json.dumps(
      summary, indent=2, sort_keys=True) + '\n')
    manifest = _read_manifest(self.shards[1])
    manifest['outputs']['summary_json']['sha256'] = _sha256(summary_path)
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'differs from recomputed'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'record-metric-fixture')
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[0]['metrics']['reference_token_accuracy'] += 5e-7
    _rewrite_samples(self.shards[0], records)
    with self.assertRaisesRegex(ValueError, 'differs from recomputed'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'prompt-identity-fixture')
    manifest = _read_manifest(self.shards[1])
    manifest['prompts']['sha256'] = 'f' * 64
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_hash_count_and_pairing_corruption(self):
    samples_path = self.shards[0] / 'samples.jsonl'
    samples_path.write_text(samples_path.read_text() + '{}\n')
    with self.assertRaisesRegex(ValueError, 'SHA256'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'third-fixture')
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[-1]['pair_key'] = 'wrong-pair'
    samples_path.write_text(''.join(
      json.dumps(record, sort_keys=True) + '\n' for record in records))
    manifest = _read_manifest(self.shards[0])
    manifest['outputs']['samples_jsonl']['sha256'] = _sha256(samples_path)
    _write_manifest(self.shards[0], manifest)
    with self.assertRaisesRegex(ValueError, 'pairing digest mismatch'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_unresolved_masks_even_with_updated_file_hash(self):
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[0]['sample_token_ids'][1] = 99
    records[0]['sample_active_token_ids'][0] = 99
    samples_path.write_text(''.join(
      json.dumps(record, sort_keys=True) + '\n' for record in records))
    manifest = _read_manifest(self.shards[0])
    manifest['outputs']['samples_jsonl']['sha256'] = _sha256(samples_path)
    _write_manifest(self.shards[0], manifest)
    with self.assertRaisesRegex(ValueError, 'unresolved mask'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_forged_batch_seed_and_noncanonical_batch_order(self):
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[0]['batch_seed'] += 1
    records[0]['timing']['batch_seed'] += 1
    _rewrite_samples(self.shards[0], records)
    with self.assertRaisesRegex(ValueError, 'does not commit'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'batch-order-fixture')
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[0], records[1] = records[1], records[0]
    _rewrite_samples(self.shards[0], records)
    with self.assertRaisesRegex(ValueError, 'ascending modulo-assigned'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'partial-batch-fixture')
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[0]['timing']['batch_size'] = 3
    _rewrite_samples(self.shards[0], records)
    with self.assertRaisesRegex(ValueError, 'final partial batch'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_wrong_reference_lm_sequence_policy_exactly(self):
    score = {
      'model_name_or_path': 'gpt2-large',
      'revision': 'a' * 40,
      'sequence_policy': 'some_other_nonempty_policy',
      'token_count': 1,
      'mean_nll_nats': 1.0,
      'perplexity': 2.718281828459045,
    }
    with self.assertRaisesRegex(ValueError, 'sequence_policy must equal'):
      _validate_reference_lm_score(score, context='score')
    with self.assertRaisesRegex(ValueError, 'sequence_policy must equal'):
      _reference_lm_identity(score, context='manifest.reference_lm')

    score['sequence_policy'] = REFERENCE_LM_SEQUENCE_POLICY
    self.assertEqual(
      _validate_reference_lm_score(score, context='score')['sequence_policy'],
      REFERENCE_LM_SEQUENCE_POLICY)

    # The only numeric tolerance is the explicitly bounded exp(mean NLL)
    # redundancy check for a reference-LM score.
    score['perplexity'] *= 1.0 + REFERENCE_LM_EXP_REL_TOL / 2.0
    _validate_reference_lm_score(score, context='score')
    score['perplexity'] *= 1.0 + REFERENCE_LM_EXP_REL_TOL * 4.0
    with self.assertRaisesRegex(ValueError, 'inconsistent with mean NLL'):
      _validate_reference_lm_score(score, context='score')

  def test_prompt_cluster_bootstrap_reports_draws_separately(self):
    intervals = paired_bootstrap_intervals(
      {0: 0.0, 1: 2.0, 2: 0.0, 3: 2.0},
      {0: 'a', 1: 'b', 2: 'a', 3: 'b'},
      num_resamples=100,
      rng_seed=3)
    self.assertEqual(intervals['paired_draws']['num_paired_draws'], 4)
    self.assertEqual(intervals['prompt_clusters']['num_prompt_clusters'], 2)
    self.assertEqual(
      intervals['prompt_clusters']['draws_per_prompt'], {'a': 2, 'b': 2})
    self.assertAlmostEqual(
      intervals['prompt_clusters']['point_estimate'], 1.0)


if __name__ == '__main__':
  unittest.main()
