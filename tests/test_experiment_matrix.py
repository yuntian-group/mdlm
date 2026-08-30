import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from scripts.compile_experiment_matrix import (
  DEFAULT_MANIFEST,
  REPO_ROOT,
  compile_matrix,
  sha256_file,
)
from scripts.run_compiled_job import run_job


class ExperimentMatrixTest(unittest.TestCase):

  def test_pilot_compiles_canonical_k64_jobs(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      plan, jobs, output = compile_matrix(
        DEFAULT_MANIFEST,
        selected_suites=['pilot'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root,
        output_dir=artifact_root / 'plan')

    self.assertEqual(plan['job_counts'], {
      'eval': 48, 'export': 2, 'train': 2})
    self.assertEqual(plan['num_jobs'], 52)
    self.assertEqual(output, (artifact_root / 'plan').resolve())
    train = jobs['train--dynamic_dynamic--s001--k064']
    self.assertIn('model.structured_decoder.top_k=64', train['argv'])
    self.assertEqual(train['execution_mode'], 'fresh_attempt')
    self.assertIn('checkpointing.resume_from_ckpt=false', train['argv'])
    self.assertFalse(any(
      token.startswith('checkpointing.resume_ckpt_path=')
      for token in train['argv']))
    self.assertIn(
      'checkpointing.save_dir={artifact_dir}', train['argv'])
    self.assertEqual(
      train['identity']['corruption_rng_policy'],
      'paired_private_torch_generator_v1_seeded_by_train_seed_epoch_rank')
    self.assertEqual(
      train['identity']['topology_teacher_rng_policy'],
      'domain_separated_private_torch_generator_v1_offset_4294967291')
    self.assertEqual(
      train['identity']['cross_control_pairing_policy'],
      'identical_corruption_stream_for_equal_train_seed_epoch_rank')
    self.assertEqual(
      {item['name'] for item in train['required_outputs']}, {
        'checkpoint', 'training_data_provenance',
        'training_validation_data_provenance'})
    training_provenance = next(
      item for item in train['required_outputs']
      if item['name'] == 'training_data_provenance')
    self.assertEqual(
      training_provenance['pattern'], 'data_provenance/train-*.json')
    evaluation = next(
      job for job in jobs.values() if job['kind'] == 'eval')
    self.assertEqual(evaluation['dependencies'], [
      f'export--{evaluation["identity"]["control"]}--s001--k064'])
    self.assertEqual(evaluation['identity']['validation_batches'], 128)
    self.assertIn(
      'eval.conditional_records.enabled=true', evaluation['argv'])
    self.assertIn(
      f'eval.conditional_records.job_id={evaluation["job_id"]}',
      evaluation['argv'])
    self.assertEqual(
      {item['name'] for item in evaluation['required_outputs']}, {
        'pairing_digest', 'conditional_records',
        'conditional_record_manifest', 'dataset_provenance'})
    self.assertTrue(
      str(evaluation['artifact_dir']).startswith(str(artifact_root.resolve())))

  def test_gated_suite_requires_matching_promotion_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      with self.assertRaisesRegex(ValueError, 'is gated on pilot'):
        compile_matrix(
          DEFAULT_MANIFEST,
          selected_suites=['candidate_k_128_pilot'],
          allowed_artifact_root=root,
          artifact_root_override=artifact_root,
          output_dir=artifact_root / 'plan')

      evidence = root / 'promotion.json'
      evidence.write_text(json.dumps({
        'schema_version': 1,
        'artifact': 'experiment_suite_promotion_decision',
        'protocol_id': 'contextual-forest-expansion-v1',
        'source_manifest_sha256': sha256_file(DEFAULT_MANIFEST),
        'source_suite': 'pilot',
        'promoted_suite': 'candidate_k_128_pilot',
        'decision': 'promote',
        'criteria': {
          'pairing_verified': True,
          'quality_gate_passed': True,
          'candidate_coverage_gate_passed': True,
        },
        'created_utc': '2026-08-30T00:00:00+00:00',
      }, sort_keys=True))
      plan, jobs, _ = compile_matrix(
        DEFAULT_MANIFEST,
        selected_suites=['candidate_k_128_pilot'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root,
        output_dir=artifact_root / 'plan',
        promotion_evidence={'candidate_k_128_pilot': evidence})

    self.assertIn('candidate_k_128_pilot', plan['promotion_evidence'])
    self.assertEqual(plan['job_counts'], {
      'eval': 16, 'export': 2, 'train': 2})
    self.assertTrue(all(
      job['identity'].get('candidate_k') == 128
      for job in jobs.values()))

  def test_resume_rejects_compiled_job_drift(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      _, jobs, output = compile_matrix(
        DEFAULT_MANIFEST,
        selected_suites=['pilot'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root,
        output_dir=artifact_root / 'plan')
      compile_matrix(
        DEFAULT_MANIFEST,
        selected_suites=['pilot'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root,
        output_dir=artifact_root / 'plan',
        resume=True)
      job_id = next(iter(jobs))
      job_path = output / 'jobs' / f'{job_id}.json'
      job_path.write_text(job_path.read_text() + ' ')
      with self.assertRaisesRegex(ValueError, 'drifted'):
        compile_matrix(
          DEFAULT_MANIFEST,
          selected_suites=['pilot'],
          allowed_artifact_root=root,
          artifact_root_override=artifact_root,
          output_dir=artifact_root / 'plan',
          resume=True)

  def test_manifest_unknown_fields_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      payload = yaml.safe_load(DEFAULT_MANIFEST.read_text())
      payload['unreviewed_override'] = True
      manifest = root / 'manifest.yaml'
      manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
      with self.assertRaisesRegex(ValueError, 'unknown=.*unreviewed_override'):
        compile_matrix(
          manifest,
          selected_suites=['pilot'],
          allowed_artifact_root=root,
          artifact_root_override=root / 'artifacts',
          output_dir=root / 'artifacts/plan',
          repo_root=REPO_ROOT)

  def test_manifest_rejects_unpaired_training_rng_policy(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      payload = yaml.safe_load(DEFAULT_MANIFEST.read_text())
      payload['training']['cross_control_pairing_policy'] = 'unpaired'
      manifest = root / 'manifest.yaml'
      manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
      with self.assertRaisesRegex(
          ValueError, 'cross_control_pairing_policy must equal'):
        compile_matrix(
          manifest,
          selected_suites=['pilot'],
          allowed_artifact_root=root,
          artifact_root_override=root / 'artifacts',
          output_dir=root / 'artifacts/plan',
          repo_root=REPO_ROOT)


class CompiledJobRunnerTest(unittest.TestCase):

  def test_success_marker_is_hash_verified_and_resumable(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      run_root = artifact_root / 'runs' / 'fake-job'
      plan = {
        'artifact_root': str(artifact_root),
        'plan_id': 'a' * 64,
      }
      job = {
        'schema_version': 1,
        'protocol_id': 'test-protocol',
        'source_manifest_sha256': 'b' * 64,
        'plan_id': 'a' * 64,
        'job_id': 'fake-job',
        'kind': 'eval',
        'artifact_dir': str(run_root),
        'suites': ['pilot'],
        'dependencies': [],
        'identity': {},
        'argv': ['{python}', '--output', '{artifact_dir}/result.json'],
        'execution_mode': 'fresh_attempt',
        'external_inputs': [],
        'required_outputs': [{
          'name': 'result', 'pattern': 'result.json', 'exactly_one': True}],
      }

      def fake_run(argv, *, cwd, check):
        self.assertEqual(cwd, REPO_ROOT)
        self.assertTrue(check)
        output = Path(argv[argv.index('--output') + 1])
        output.write_text('{"ok": true}\n')

      with mock.patch('scripts.run_compiled_job.subprocess.run', fake_run):
        first = run_job(
          'fake-job', plan=plan, jobs={'fake-job': job})
        second = run_job(
          'fake-job', plan=plan, jobs={'fake-job': job})
      self.assertEqual(first, 'completed')
      self.assertEqual(second, 'skipped')
      promoted_job = dict(
        job, plan_id='c' * 64, suites=['confirmation'])
      promoted_plan = dict(plan, plan_id='c' * 64)
      self.assertEqual(run_job(
        'fake-job', plan=promoted_plan,
        jobs={'fake-job': promoted_job}), 'skipped')
      marker = json.loads((run_root / '_job_success.json').read_text())
      output_record = marker['outputs'][0]
      result_path = Path(marker['run_dir']) / output_record['relative_path']
      self.assertEqual(
        output_record['sha256'], hashlib.sha256(result_path.read_bytes()).hexdigest())

      result_path.write_text('{"ok": false}\n')
      with self.assertRaisesRegex(ValueError, 'outputs drifted'):
        run_job('fake-job', plan=plan, jobs={'fake-job': job})


if __name__ == '__main__':
  unittest.main()
