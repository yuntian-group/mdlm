#!/usr/bin/env python3
"""Run a paired 256-sample generation/infilling pilot and persist every row.

The default experiment compares factorized, exact structured-marginal, and
exact structured-joint token sampling using the same prompt order and seed
commitment.  No output is overwritten, and both model artifacts must be
identified by caller-supplied SHA256 digests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from evaluation.generation_harness import (  # noqa: E402
  SAMPLING_MODES,
  expand_paired_samples,
  iter_batches,
  pairing_digest,
  prompt_from_record,
  run_sampling_group,
  summarize_group,
  unconditional_prompt,
)
from evaluation.generation_metrics import (  # noqa: E402
  TransformersReferenceLMScorer,
  validate_reference_lm_spec,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    while True:
      chunk = handle.read(chunk_size)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


def verify_artifact(path: Path, expected_sha256: str, role: str) -> str:
  path = path.resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  expected = expected_sha256.lower()
  if len(expected) != 64 or any(value not in '0123456789abcdef'
                                for value in expected):
    raise ValueError(f'{role} SHA256 must be 64 hexadecimal characters')
  actual = sha256_file(path)
  if actual != expected:
    raise ValueError(
      f'{role} SHA256 mismatch: expected {expected}, found {actual}')
  return actual


def _register_resolvers() -> None:
  from omegaconf import OmegaConf

  resolvers = {
    'cwd': os.getcwd,
    'device_count': torch.cuda.device_count,
    'eval': eval,
    'div_up': lambda x, y: (x + y - 1) // y,
  }
  for name, resolver in resolvers.items():
    if not OmegaConf.has_resolver(name):
      OmegaConf.register_new_resolver(name, resolver)


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Paired end-to-end contextual-forest generation pilot.')
  parser.add_argument('--backbone-checkpoint', type=Path, required=True)
  parser.add_argument('--backbone-sha256', required=True)
  parser.add_argument('--adapter', type=Path, required=True)
  parser.add_argument('--adapter-sha256', required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--prompt-jsonl', type=Path)
  parser.add_argument('--num-samples', type=int, default=256)
  parser.add_argument(
    '--num-shards', type=int, default=1,
    help='Split the global paired sample set into independent Spot-safe runs.')
  parser.add_argument(
    '--shard-index', type=int, default=0,
    help='Zero-based shard to run; each shard requires a fresh output dir.')
  parser.add_argument('--sequence-length', type=int, default=256)
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--base-seed', type=int, default=91001)
  parser.add_argument(
    '--modes', nargs='+', choices=SAMPLING_MODES,
    default=list(SAMPLING_MODES))
  parser.add_argument('--nfe-budgets', nargs='+', type=int, default=[32, 64])
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--model-config', default='contextual-forest-small')
  parser.add_argument('--data-config', default='openwebtext-streaming')
  parser.add_argument(
    '--override', action='append', default=[],
    help='Additional Hydra override; repeat for multiple overrides.')
  parser.add_argument('--reference-lm')
  parser.add_argument('--reference-lm-revision')
  parser.add_argument('--reference-lm-device', default='cuda')
  parser.add_argument('--reference-lm-batch-size', type=int, default=8)
  parser.add_argument(
    '--allow-dirty', action='store_true',
    help='Allow a dirty Git tree; dirty content is committed in the manifest.')
  return parser.parse_args(argv)


def _load_prompt_records(
    path: Path | None,
    *,
    tokenizer,
    mask_token_id: int,
    sequence_length: int,
) -> tuple[list, dict[str, Any]]:
  if path is None:
    return [unconditional_prompt(
      mask_token_id=mask_token_id,
      sequence_length=sequence_length)], {
        'source': 'generated_unconditional_prompt',
        'path': None,
        'sha256': None,
        'num_prompt_records': 1,
      }
  path = path.resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  prompts = []
  with path.open() as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError as error:
        raise ValueError(
          f'invalid JSON in {path} line {line_number}: {error}') from error
      if not isinstance(record, dict):
        raise ValueError(f'{path} line {line_number} is not a JSON object')
      prompts.append(prompt_from_record(
        record,
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        sequence_length=sequence_length,
        line_number=line_number))
  if not prompts:
    raise ValueError(f'{path} contains no prompt records')
  return prompts, {
    'source': 'jsonl',
    'path': str(path),
    'sha256': sha256_file(path),
    'num_prompt_records': len(prompts),
  }


def _git_provenance() -> dict[str, Any]:
  def command(*args: str) -> str:
    return subprocess.check_output(
      ['git', *args], cwd=REPO_ROOT, text=True,
      stderr=subprocess.DEVNULL)

  try:
    sha = command('rev-parse', 'HEAD').strip()
    status = command('status', '--porcelain=v1')
    diff = subprocess.check_output(
      ['git', 'diff', '--binary', 'HEAD'], cwd=REPO_ROOT)
    untracked_output = subprocess.check_output(
      ['git', 'ls-files', '--others', '--exclude-standard', '-z'],
      cwd=REPO_ROOT)
  except (OSError, subprocess.CalledProcessError):
    return {
      'git_sha': None,
      'dirty': None,
      'status_porcelain': None,
      'tracked_diff_sha256': None,
      'untracked_files': None,
      'dirty_content_sha256': None,
    }
  untracked_files = []
  for raw_path in untracked_output.split(b'\0'):
    if not raw_path:
      continue
    relative_path = raw_path.decode('utf-8', errors='strict')
    path = REPO_ROOT / relative_path
    if path.is_file():
      untracked_files.append({
        'path': relative_path,
        'sha256': sha256_file(path),
        'size_bytes': path.stat().st_size,
      })
  tracked_diff_sha256 = hashlib.sha256(diff).hexdigest()
  dirty_content_sha256 = hashlib.sha256(
    diff + json.dumps(
      untracked_files, sort_keys=True,
      separators=(',', ':')).encode('utf-8')).hexdigest()
  return {
    'git_sha': sha,
    'dirty': bool(status),
    'status_porcelain': status.splitlines(),
    'tracked_diff_sha256': tracked_diff_sha256,
    'untracked_files': untracked_files,
    'dirty_content_sha256': dirty_content_sha256,
  }


def _compose_config(args: argparse.Namespace):
  import hydra
  from omegaconf import open_dict

  _register_resolvers()
  overrides = [
    f'model={args.model_config}',
    f'data={args.data_config}',
    *args.override,
  ]
  with hydra.initialize_config_dir(
      config_dir=str(REPO_ROOT / 'configs'), version_base=None):
    config = hydra.compose(config_name='config', overrides=overrides)
  with open_dict(config):
    config.mode = 'sample_eval'
    config.model.length = int(args.sequence_length)
    config.model.structured_decoder.training.backbone_checkpoint = str(
      args.backbone_checkpoint.resolve())
    config.model.structured_decoder.training.use_ema_backbone = False
    config.model.structured_decoder.training.strict_backbone_checkpoint = True
    config.model.structured_decoder.sampling.mode = 'factorized'
    config.eval.checkpoint_path = ''
    config.eval.adapter_checkpoint = str(args.adapter.resolve())
    config.eval.adapter_sha256 = args.adapter_sha256.lower()
    config.eval.disable_ema = True
    config.eval.compute_generative_perplexity = False
    config.eval.generate_samples = False
    config.training.ema = 0.0
    config.checkpointing.resume_from_ckpt = False
    config.sampling.predictor = 'ddpm'
    config.sampling.noise_removal = True
    config.sampling.semi_ar = False
    config.loader.eval_batch_size = int(args.batch_size)
    config.loader.eval_global_batch_size = int(args.batch_size)
  return config


def _output_paths(output_dir: Path) -> dict[str, Path]:
  output_dir = output_dir.resolve()
  if output_dir.exists() and any(output_dir.iterdir()):
    raise FileExistsError(
      f'refusing to reuse non-empty output directory {output_dir}; '
      'rerun an interrupted Spot shard in a fresh directory')
  paths = {
    'samples': output_dir / 'samples.jsonl',
    'summary': output_dir / 'summary.json',
    'manifest': output_dir / 'manifest.json',
    'config': output_dir / 'resolved_config.yaml',
  }
  existing = [path for path in paths.values() if path.exists()]
  if existing:
    raise FileExistsError(
      f'refusing to overwrite existing experiment artifact {existing[0]}')
  output_dir.mkdir(parents=True, exist_ok=True)
  return paths


def _atomic_write_text(path: Path, content: str) -> None:
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
  try:
    temporary.write_text(content)
    os.replace(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def _attach_reference_lm_scores(
    records: list[dict[str, Any]],
    *,
    model_name_or_path: str,
    revision: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
  scorer = TransformersReferenceLMScorer(
    model_name_or_path,
    revision=revision,
    device=device,
    batch_size=batch_size)
  scores = scorer.score([record['text'] for record in records])
  total_nll = 0.0
  total_tokens = 0
  for record, score in zip(records, scores):
    payload = {
      'model_name_or_path': model_name_or_path,
      'revision': revision,
      'token_count': score.token_count,
      'mean_nll_nats': score.mean_nll_nats,
      'perplexity': score.perplexity,
    }
    record['reference_lm'] = payload
    if score.mean_nll_nats is not None:
      total_nll += score.mean_nll_nats * score.token_count
      total_tokens += score.token_count
  mean_nll = total_nll / total_tokens if total_tokens else None
  return {
    'model_name_or_path': model_name_or_path,
    'revision': revision,
    'device': device,
    'num_scored_sequences': len(scores),
    'num_scored_tokens': total_tokens,
    'mean_nll_nats': mean_nll,
    'perplexity': (
      float(torch.exp(torch.tensor(min(mean_nll, 80.0))).item())
      if mean_nll is not None else None),
  }


def _summarize_attached_reference_lm(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
  total_nll = 0.0
  total_tokens = 0
  for record in records:
    score = record['reference_lm']
    if score['mean_nll_nats'] is not None:
      total_nll += score['mean_nll_nats'] * score['token_count']
      total_tokens += score['token_count']
  mean_nll = total_nll / total_tokens if total_tokens else None
  identities = {
    (record['reference_lm']['model_name_or_path'],
     record['reference_lm']['revision'])
    for record in records
  }
  if len(identities) != 1:
    raise ValueError('reference-LM identities differ within one summary')
  model_name_or_path, revision = next(iter(identities))
  return {
    'model_name_or_path': model_name_or_path,
    'revision': revision,
    'num_scored_sequences': len(records),
    'num_scored_tokens': total_tokens,
    'mean_nll_nats': mean_nll,
    'perplexity': (
      float(torch.exp(torch.tensor(min(mean_nll, 80.0))).item())
      if mean_nll is not None else None),
  }


def main(argv=None) -> int:
  args = _parse_args(argv)
  args.reference_lm, args.reference_lm_revision = validate_reference_lm_spec(
    args.reference_lm, args.reference_lm_revision)

  import dataloader
  import diffusion
  from omegaconf import OmegaConf

  if args.num_samples <= 0:
    raise ValueError('--num-samples must be positive')
  if args.sequence_length <= 0:
    raise ValueError('--sequence-length must be positive')
  if args.batch_size <= 0:
    raise ValueError('--batch-size must be positive')
  if args.num_shards <= 0:
    raise ValueError('--num-shards must be positive')
  if not 0 <= args.shard_index < args.num_shards:
    raise ValueError('--shard-index must lie in [0, num-shards)')
  if len(set(args.modes)) != len(args.modes):
    raise ValueError('--modes contains duplicates')
  if any(budget < 2 for budget in args.nfe_budgets):
    raise ValueError('--nfe-budgets must be at least 2')
  if len(set(args.nfe_budgets)) != len(args.nfe_budgets):
    raise ValueError('--nfe-budgets contains duplicates')

  start_time = dt.datetime.now(dt.timezone.utc)
  backbone_sha256 = verify_artifact(
    args.backbone_checkpoint, args.backbone_sha256, 'backbone')
  adapter_sha256 = verify_artifact(
    args.adapter, args.adapter_sha256, 'adapter')
  repository_provenance = _git_provenance()
  if repository_provenance['dirty'] and not args.allow_dirty:
    raise RuntimeError(
      'refusing to run from a dirty Git tree; commit the experiment code or '
      'pass --allow-dirty to record a content commitment explicitly')
  paths = _output_paths(args.output_dir)
  config = _compose_config(args)
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion(config, tokenizer=tokenizer)
  device = torch.device(args.device)
  if device.type == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable')
  # Keep stored parameters in their checkpoint dtype.  DIT owns its tested
  # bf16 CUDA autocast regions internally; eagerly casting all parameters
  # would make its pre-autocast timestep MLP receive mismatched input dtypes.
  model = model.to(device=device).eval()
  model.backbone.eval()
  model.structured_head.eval()
  model.noise.eval()
  if model.ema is not None:
    raise AssertionError('pilot requires EMA to be disabled explicitly')
  parameter_dtypes = sorted({
    str(parameter.dtype) for parameter in model.parameters()
  })

  prompts, prompt_provenance = _load_prompt_records(
    args.prompt_jsonl,
    tokenizer=tokenizer,
    mask_token_id=model.mask_index,
    sequence_length=args.sequence_length)
  global_paired_samples = expand_paired_samples(
    prompts,
    num_samples=args.num_samples,
    base_seed=args.base_seed)
  global_pairing_digest = pairing_digest(global_paired_samples)
  paired_samples = [
    sample for sample in global_paired_samples
    if sample.sample_index % args.num_shards == args.shard_index
  ]
  if not paired_samples:
    raise ValueError(
      f'shard {args.shard_index}/{args.num_shards} contains no samples; '
      'use no more shards than global samples')
  input_pairing_digest = pairing_digest(paired_samples)

  all_records: list[dict[str, Any]] = []
  summaries = []
  for mode in args.modes:
    for nfe_budget in args.nfe_budgets:
      group_records = []
      for batch in iter_batches(paired_samples, args.batch_size):
        records, batch_metadata = run_sampling_group(
          model,
          batch,
          sampling_mode=mode,
          nfe_budget=nfe_budget,
          tokenizer=tokenizer,
          device=device)
        for record in records:
          record.update({
            'global_pairing_digest': global_pairing_digest,
            'shard_pairing_digest': input_pairing_digest,
            'num_shards': args.num_shards,
            'shard_index': args.shard_index,
          })
        group_records.extend(records)
        print(json.dumps({
          'event': 'generation_batch_complete',
          'sampling_mode': mode,
          'nfe_budget': nfe_budget,
          'completed_samples': len(group_records),
          'total_samples': len(paired_samples),
          'batch': batch_metadata,
        }, sort_keys=True), flush=True)
      summary = summarize_group(group_records)
      if summary['pairing_digest'] != input_pairing_digest:
        raise AssertionError(
          'mode/NFE group does not match the committed shard inputs')
      summary['input_pairing_digest'] = input_pairing_digest
      summaries.append(summary)
      all_records.extend(group_records)

  reference_lm = None
  if args.reference_lm:
    del model
    if device.type == 'cuda':
      torch.cuda.empty_cache()
    reference_lm = _attach_reference_lm_scores(
      all_records,
      model_name_or_path=args.reference_lm,
      revision=args.reference_lm_revision,
      device=args.reference_lm_device,
      batch_size=args.reference_lm_batch_size)
    for summary in summaries:
      group_records = [
        record for record in all_records
        if (record['sampling_mode'] == summary['sampling_mode']
            and record['requested_nfe_budget']
            == summary['requested_nfe_budget'])
      ]
      summary['reference_lm'] = _summarize_attached_reference_lm(
        group_records)

  end_time = dt.datetime.now(dt.timezone.utc)
  summary_payload = {
    'schema_version': 1,
    'experiment': 'paired_contextual_forest_generation_pilot',
    'global_pairing_digest': global_pairing_digest,
    'input_pairing_digest': input_pairing_digest,
    'global_num_paired_samples': args.num_samples,
    'num_paired_samples': len(paired_samples),
    'shard_index': args.shard_index,
    'num_shards': args.num_shards,
    'groups': summaries,
    'reference_lm': reference_lm,
  }
  jsonl = ''.join(
    json.dumps(record, sort_keys=True) + '\n' for record in all_records)
  _atomic_write_text(paths['samples'], jsonl)
  samples_sha256 = sha256_file(paths['samples'])
  _atomic_write_text(
    paths['summary'],
    json.dumps(summary_payload, indent=2, sort_keys=True) + '\n')
  summary_sha256 = sha256_file(paths['summary'])
  _atomic_write_text(paths['config'], OmegaConf.to_yaml(config, resolve=True))
  config_sha256 = sha256_file(paths['config'])

  manifest = {
    'schema_version': 1,
    'experiment': 'paired_contextual_forest_generation_pilot',
    'scientific_scope': (
      'end-to-end sampling pilot; quality metrics are descriptive and do not '
      'estimate the diffusion ELBO or likelihood'),
    'command': sys.argv if argv is None else [sys.argv[0], *argv],
    'start_time_utc': start_time.isoformat(),
    'end_time_utc': end_time.isoformat(),
    'duration_seconds': (end_time - start_time).total_seconds(),
    'host': {
      'hostname': platform.node(),
      'platform': platform.platform(),
      'python': platform.python_version(),
      'torch': torch.__version__,
      'device': str(device),
      'gpu': (torch.cuda.get_device_name(device)
              if device.type == 'cuda' else None),
      'parameter_dtypes': parameter_dtypes,
      'precision_policy': (
        'checkpoint dtype; DIT-managed bf16 autocast on CUDA'),
    },
    'repository': repository_provenance,
    'artifacts': {
      'backbone_checkpoint': {
        'path': str(args.backbone_checkpoint.resolve()),
        'sha256': backbone_sha256,
        'size_bytes': args.backbone_checkpoint.resolve().stat().st_size,
      },
      'structured_adapter': {
        'path': str(args.adapter.resolve()),
        'sha256': adapter_sha256,
        'size_bytes': args.adapter.resolve().stat().st_size,
      },
    },
    'prompts': prompt_provenance,
    'pairing': {
      'digest_algorithm': 'sha256-canonical-json-v1',
      'global_pairing_digest': global_pairing_digest,
      'shard_pairing_digest': input_pairing_digest,
      'base_seed': args.base_seed,
      'batch_size': args.batch_size,
      'global_num_samples': args.num_samples,
      'shard_num_samples': len(paired_samples),
      'num_shards': args.num_shards,
      'shard_index': args.shard_index,
      'sequence_length': args.sequence_length,
    },
    'spot_interruption_policy': {
      'resume_supported': False,
      'partial_output_append_supported': False,
      'policy': (
        'Outputs are committed atomically only after all matrix cells in this '
        'shard finish. Run small independent shards with --num-shards and '
        '--shard-index; rerun an interrupted shard in a fresh output dir.'),
    },
    'matrix': {
      'sampling_modes': args.modes,
      'nfe_budgets': args.nfe_budgets,
      'num_output_records': len(all_records),
    },
    'outputs': {
      'samples_jsonl': {
        'path': paths['samples'].name,
        'sha256': samples_sha256,
        'num_records': len(all_records),
      },
      'summary_json': {
        'path': paths['summary'].name,
        'sha256': summary_sha256,
      },
      'resolved_config': {
        'path': paths['config'].name,
        'sha256': config_sha256,
      },
    },
    'reference_lm': reference_lm,
  }
  _atomic_write_text(
    paths['manifest'], json.dumps(manifest, indent=2, sort_keys=True) + '\n')
  print(json.dumps({
    'event': 'generation_pilot_complete',
    'output_dir': str(args.output_dir.resolve()),
    'num_output_records': len(all_records),
    'input_pairing_digest': input_pairing_digest,
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
