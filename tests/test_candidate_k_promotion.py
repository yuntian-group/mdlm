import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.compile_experiment_matrix import (
  DEFAULT_MANIFEST,
  build_jobs,
  load_and_validate_manifest,
  sha256_file,
)
from scripts.evaluate_candidate_k_promotion import evaluate_candidate_k_analysis
from scripts.evaluate_experiment_promotion import canonical_sha256
from scripts.finalize_candidate_k_policy import (
  DEFAULT_TEMPLATE,
  _atomic_write,
  _read_mapping,
  _validate_template,
  finalize_policy,
)


def _write_json(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(
    payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _candidate_plan(root):
  template = _validate_template(
    _read_mapping(DEFAULT_TEMPLATE, context='template'),
    manifest_path=DEFAULT_MANIFEST)
  plan_dir = root / 'plan'
  artifact_root = root / 'artifacts'
  parent_path = root / 'parent-promotion.json'
  _write_json(parent_path, {})
  parent = {
    'path': str(parent_path.resolve()),
    'sha256': sha256_file(parent_path),
    'source_suite': 'pilot',
    'route_name': 'k128',
    'canonical_decision_sha256': 'b' * 64,
    'source_compiled_plan_sha256': 'c' * 64,
  }
  repository = {'sha': 'd' * 40, 'dirty': False}
  promotion_evidence = {'candidate_k_128_pilot': parent}
  identity = {
    'protocol_id': template['protocol_id'],
    'source_manifest_sha256': template['source_manifest_sha256'],
    'repository': repository,
    'artifact_root': str(artifact_root.resolve()),
    'selected_suites': ['candidate_k_128_pilot'],
    'promotion_evidence': promotion_evidence,
  }
  plan_id = canonical_sha256(identity)
  manifest = load_and_validate_manifest(DEFAULT_MANIFEST)
  jobs = build_jobs(
    manifest,
    selected_suites=['candidate_k_128_pilot'],
    artifact_root=artifact_root.resolve(),
    source_manifest_sha256=template['source_manifest_sha256'],
    source_repository_sha=repository['sha'],
    plan_id=plan_id)
  counts = {}
  for job in jobs.values():
    counts[job['kind']] = counts.get(job['kind'], 0) + 1
  plan = {
    'schema_version': 2,
    **identity,
    'plan_id': plan_id,
    'manifest_protocol_status': manifest['protocol_status'],
    'scientific_scope': manifest['scientific_scope'],
    'job_counts': dict(sorted(counts.items())),
    'num_jobs': len(jobs),
    'job_ids': list(jobs),
    'job_spec_sha256': {
      job_id: canonical_sha256(job) for job_id, job in jobs.items()},
  }
  for job_id, job in jobs.items():
    _write_json(plan_dir / 'jobs' / f'{job_id}.json', job)
  _write_json(plan_dir / 'compiled-plan.json', plan)
  return plan_dir, plan, jobs, template


def _parent_verification(plan):
  parent = plan['promotion_evidence']['candidate_k_128_pilot']
  return {
    'route_name': 'k128',
    'commitments': {
      'canonical_decision_sha256': parent['canonical_decision_sha256'],
      'source_compiled_plan_sha256': parent[
        'source_compiled_plan_sha256'],
    },
  }


def _policy_bundle(root):
  plan_dir, plan, jobs, template = _candidate_plan(root)
  with mock.patch(
      'scripts.evaluate_experiment_promotion.verify_compiler_evidence',
      return_value=_parent_verification(plan)), mock.patch(
      'scripts.finalize_candidate_k_policy._git_metadata',
      return_value=plan['repository']):
    policy = finalize_policy(
      DEFAULT_TEMPLATE,
      plan_dir,
      manifest_path=DEFAULT_MANIFEST,
      frozen_utc='2026-08-30T16:00:00+00:00')
  return policy, plan_dir / 'compiled-plan.json', plan, jobs, template


def _source_integrity(policy, plan_path):
  contract = policy['analysis_contract']
  output_sha = 'e' * 64
  body = {
    'schema_version': 1,
    'source_compiled_plan_path': str(plan_path.resolve()),
    'source_compiled_plan_sha256': contract['source_compiled_plan_sha256'],
    'source_plan_id': contract['source_plan_id'],
    'source_manifest_sha256': policy['source_manifest_sha256'],
    'source_repository_sha': contract['source_repository_sha'],
    'source_repository_clean': True,
    'validated_job_ids': ['eval--fixture'],
    'jobs': {
      'eval--fixture': {
        'job_spec_sha256': 'a' * 64,
        'job_execution_sha256': 'b' * 64,
        'success_marker_path': str(
          (plan_path.parent / 'run' / '_job_success.json').resolve()),
        'success_marker_sha256': 'c' * 64,
        'outputs': [{
          'name': 'conditional_records',
          'relative_path': 'conditional-records.jsonl',
          'size_bytes': 1,
          'sha256': output_sha,
        }],
        'scientific_output_sha256': {
          'conditional_records': output_sha,
        },
      },
    },
  }
  return {**body, 'commitment_sha256': canonical_sha256(body)}


def _analysis(
    policy,
    plan_path,
    *,
    improvement=0.02,
    pooled_ci_lower=0.005,
    candidate_recall=0.80,
    retained_mass=0.80,
):
  contract = policy['analysis_contract']
  conditions = {}
  diagnostics = {}
  for dataset, dataset_spec in contract['datasets'].items():
    revision = dataset_spec['revision']
    for mask_rate in contract['mask_rates']:
      condition_key = (
        f'{dataset}|mask={mask_rate:.6f}|k={contract["candidate_k"]}')
      conditions[condition_key] = {
        'dataset': dataset,
        'dataset_revision': revision,
        'mask_rate': mask_rate,
        'candidate_k': contract['candidate_k'],
        'mean_improvement': improvement,
        'ci_lower': improvement - 0.01,
        'ci_upper': improvement + 0.01,
      }
      diagnostic_key = str((
        dataset, revision, mask_rate, contract['candidate_k']))
      diagnostics[diagnostic_key] = {
        'num_documents': 20,
        'num_train_seeds': len(contract['train_seeds']),
        'num_corruption_seeds': len(contract['corruption_seeds']),
        'arms': {
          contract['baseline_arm']: {
            'conditional_nll_per_masked_token': 3.0,
            'candidate_recall': candidate_recall,
            'retained_unary_mass': retained_mass,
          },
          contract['treatment_arm']: {
            'conditional_nll_per_masked_token': 3.0 - improvement,
            'candidate_recall': candidate_recall,
            'retained_unary_mass': retained_mass,
          },
        },
      }
  bootstrap = contract['bootstrap']
  integrity = _source_integrity(policy, plan_path)
  return {
    'schema_version': contract['analysis_schema_version'],
    'artifact': contract['analysis_artifact'],
    'created_utc': '2026-08-30T17:00:00+00:00',
    'protocol_id': policy['protocol_id'],
    'suite': policy['source_suite'],
    'comparison': contract['comparison'],
    'arms': {
      'baseline': contract['baseline_arm'],
      'treatment': contract['treatment_arm'],
    },
    'objective': contract['objective'],
    'scope_note': (
      'Conditional denoising only; no diffusion ELBO, likelihood, '
      'perplexity, or generation-quality quantity is inferred.'),
    'by_candidate_k': {
      str(contract['candidate_k']): {
        'method': bootstrap['method'],
        'improvement_definition': 'baseline conditional NLL minus treatment',
        'nesting': [
          'average corruption replications within source document',
          'resample training seeds with replacement',
          'resample source documents within sampled training seed',
          'equal-weight frozen dataset x mask-rate strata',
        ],
        'num_train_seeds': len(contract['train_seeds']),
        'num_strata': contract['expected_num_strata'],
        'num_resamples': bootstrap['num_resamples'],
        'rng': 'NumPy Generator(PCG64)',
        'rng_seed': bootstrap['rng_seed'],
        'confidence_level': bootstrap['confidence_level'],
        'pooled': {
          'mean_improvement': improvement,
          'ci_lower': pooled_ci_lower,
          'ci_upper': improvement + 0.01,
        },
        'conditions': conditions,
      },
    },
    'diagnostics': diagnostics,
    'compiled_plan': {
      'plan_id': contract['source_plan_id'],
      'source_manifest_sha256': policy['source_manifest_sha256'],
      'source_compiled_plan_sha256': contract[
        'source_compiled_plan_sha256'],
      'source_repository_sha': contract['source_repository_sha'],
      'source_repository_clean': True,
      'job_artifact_commitment_sha256': integrity['commitment_sha256'],
    },
    'source_integrity': integrity,
  }


class CandidateKPromotionTest(unittest.TestCase):

  def _evaluate(self, root, **analysis_kwargs):
    policy, plan_path, plan, _, _ = _policy_bundle(root)
    analysis = _analysis(policy, plan_path, **analysis_kwargs)
    with mock.patch(
        'scripts.evaluate_candidate_k_promotion.'
        '_verify_authoritative_analysis',
        return_value=analysis['source_integrity']), mock.patch(
        'scripts.evaluate_candidate_k_promotion.'
        '_verify_policy_predates_source_jobs'):
      decision = evaluate_candidate_k_analysis(
        policy,
        analysis,
        policy_sha256='a' * 64,
        analysis_sha256='b' * 64,
        source_plan=plan,
        source_plan_path=plan_path,
        source_plan_sha256=sha256_file(plan_path),
        manifest_path=DEFAULT_MANIFEST,
        created_utc='2026-08-30T18:00:00+00:00')
    return decision

  def test_template_is_bound_to_current_manifest(self):
    template = _validate_template(
      _read_mapping(DEFAULT_TEMPLATE, context='template'),
      manifest_path=DEFAULT_MANIFEST)
    self.assertEqual(template['analysis_contract']['candidate_k'], 128)
    self.assertEqual(
      template['source_manifest_sha256'], sha256_file(DEFAULT_MANIFEST))

  def test_quality_positive_support_limited_promotes_both_routes(self):
    with tempfile.TemporaryDirectory() as directory:
      decision = self._evaluate(Path(directory))
    self.assertEqual(
      decision['outcome'], 'promote_k128_confirmation_and_k256')
    self.assertTrue(decision['routes']['confirmation']['promote'])
    self.assertTrue(decision['routes']['k256']['promote'])

  def test_quality_negative_support_limited_promotes_only_k256(self):
    with tempfile.TemporaryDirectory() as directory:
      decision = self._evaluate(
        Path(directory), improvement=-0.02, pooled_ci_lower=-0.04)
    self.assertEqual(decision['outcome'], 'promote_k256_only')
    self.assertFalse(decision['routes']['confirmation']['promote'])
    self.assertTrue(decision['routes']['k256']['promote'])

  def test_quality_positive_support_sufficient_promotes_confirmation_only(self):
    with tempfile.TemporaryDirectory() as directory:
      decision = self._evaluate(
        Path(directory), candidate_recall=0.995, retained_mass=0.97)
    self.assertEqual(decision['outcome'], 'promote_k128_confirmation_only')
    self.assertTrue(decision['routes']['confirmation']['promote'])
    self.assertFalse(decision['routes']['k256']['promote'])

  def test_finalizer_refuses_nonempty_source_artifact_directory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, plan, jobs, _ = _candidate_plan(root)
      job = next(iter(jobs.values()))
      artifact_dir = Path(job['artifact_dir'])
      artifact_dir.mkdir(parents=True)
      (artifact_dir / 'started.txt').write_text('started\n')
      with mock.patch(
          'scripts.evaluate_experiment_promotion.verify_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.finalize_candidate_k_policy._git_metadata',
          return_value=plan['repository']), self.assertRaisesRegex(
          RuntimeError, 'execution began'):
        finalize_policy(DEFAULT_TEMPLATE, plan_dir)

  def test_finalizer_rejects_tampered_factorial_job(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, plan, jobs, _ = _candidate_plan(root)
      job_id, job = next(iter(jobs.items()))
      job['identity']['candidate_k'] = 256
      _write_json(plan_dir / 'jobs' / f'{job_id}.json', job)
      with mock.patch(
          'scripts.evaluate_experiment_promotion.verify_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.finalize_candidate_k_policy._git_metadata',
          return_value=plan['repository']), self.assertRaisesRegex(
          ValueError, 'exact frozen K128 factorial'):
        finalize_policy(DEFAULT_TEMPLATE, plan_dir)

  def test_finalizer_refuses_timestamp_before_template(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, plan, _, _ = _candidate_plan(root)
      with mock.patch(
          'scripts.evaluate_experiment_promotion.verify_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.finalize_candidate_k_policy._git_metadata',
          return_value=plan['repository']), self.assertRaisesRegex(
          ValueError, 'predate the policy template'):
        finalize_policy(
          DEFAULT_TEMPLATE,
          plan_dir,
          frozen_utc='2026-08-30T15:54:59+00:00')

  def test_finalizer_rejects_dirty_checkout(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, plan, _, _ = _candidate_plan(root)
      with mock.patch(
          'scripts.finalize_candidate_k_policy._git_metadata',
          return_value={
            'sha': plan['repository']['sha'], 'dirty': True,
          }), self.assertRaisesRegex(ValueError, 'checkout differs'):
        finalize_policy(DEFAULT_TEMPLATE, plan_dir)

  def test_finalizer_rejects_different_checkout_revision(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, _, _, _ = _candidate_plan(root)
      with mock.patch(
          'scripts.finalize_candidate_k_policy._git_metadata',
          return_value={'sha': 'e' * 40, 'dirty': False}), \
          self.assertRaisesRegex(ValueError, 'checkout differs'):
        finalize_policy(DEFAULT_TEMPLATE, plan_dir)

  def _assert_publication_race_preserves_competing_artifact(self, name):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / name
      original_link = os.link

      def competing_publish(source, target):
        Path(target).write_text('competing artifact\n')
        return original_link(source, target)

      with mock.patch(
          'scripts.finalize_candidate_k_policy.os.link',
          side_effect=competing_publish), self.assertRaises(FileExistsError):
        _atomic_write(destination, {'ours': True})
      self.assertEqual(destination.read_text(), 'competing artifact\n')
      self.assertEqual(
        list(destination.parent.glob(f'.{destination.name}.tmp-*')), [])

  def test_candidate_decision_race_never_overwrites_destination(self):
    self._assert_publication_race_preserves_competing_artifact(
      'routing-decision.json')

  def test_candidate_evidence_race_never_overwrites_destination(self):
    self._assert_publication_race_preserves_competing_artifact(
      'candidate-k-promotion.json')

  def test_evaluator_rejects_source_plan_copy_at_different_path(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      policy, plan_path, plan, _, _ = _policy_bundle(root)
      copied = root / 'copied' / 'compiled-plan.json'
      _write_json(copied, plan)
      analysis = _analysis(policy, plan_path)
      with self.assertRaisesRegex(
          ValueError, 'path differs from policy'):
        evaluate_candidate_k_analysis(
          policy,
          analysis,
          policy_sha256='a' * 64,
          analysis_sha256='b' * 64,
          source_plan=plan,
          source_plan_path=copied,
          source_plan_sha256=sha256_file(copied))


if __name__ == '__main__':
  unittest.main()
