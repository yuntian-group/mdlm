"""Local-assets, single-run additive-head training with durable continuation.

This portable dense-reference trainer optimizes the candidate-set revealed-block
surrogate, NOT held-out data PPL. It neither downloads assets nor selects a
generation coefficient. Every checkpoint destination is explicitly supplied.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time

import numpy as np
import torch

from . import assets
from .adaptation import adaptation_step
from .backbone import (CHECKPOINT_REVISION, TOKENIZER_REPO, TOKENIZER_REVISION,
                       file_sha256, load_bd3lm)
from .counts import SparseCountPotential
from .heads import ContextualAdditiveHead, GlobalAdditiveHead
from .prepare_counts import validate_manifest as validate_count_manifest

METRIC = 'candidate_set_revealed_block_nll_per_selected_token_not_data_ppl'
SOURCE_FILES = ('train_cli.py', 'adaptation.py', 'assets.py', 'backbone.py',
                'prepare_counts.py', 'counts.py', 'heads.py', 'composition.py',
                'training.py', 'chain.py', 'fast_candidates.py', 'candidates.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_identity():
    root = Path(__file__).resolve().parent
    return {name: file_sha256(root / name) for name in SOURCE_FILES}


@dataclass(frozen=True)
class TrainingConfig:
    kind: str = 'global'
    rank: int = 64
    context_size: int = 128
    count_strength: float = .1
    global_strength: float = 0.
    top_k: int = 500
    top_p: float = .9
    schedule_steps: int = 32
    learning_rate: float = .001
    max_updates: int = 5000
    min_updates: int = 1000
    eval_every: int = 250
    dev_blocks: int = 32
    patience: int = 4
    checkpoint_every: int = 25

    def validate(self):
        require(self.kind in ('global', 'contextual'), 'Unknown stage')
        for key in ('rank', 'context_size', 'top_k', 'schedule_steps', 'max_updates',
                    'min_updates', 'eval_every', 'dev_blocks', 'patience', 'checkpoint_every'):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0,
                    'Positive integer required: ' + key)
        require(self.rank <= 128 and self.top_k <= 2048,
                'Head interchange supports rank<=128 and candidate cap<=2048')
        require(self.min_updates <= self.max_updates, 'min_updates exceeds max_updates')
        for name in ('count_strength', 'global_strength'):
            assets.number(getattr(self, name), name)
        assets.number(self.learning_rate, 'learning_rate', positive=True)
        require(type(self.top_p) in (float, int) and 0 < self.top_p <= 1, 'Invalid top_p')
        require(self.kind != 'global' or self.global_strength == 0,
                'Global stage has no inherited global coefficient')


class PackedRows:
    """SHA-bound .npy windows, with explicit train/dev declarations and no test API.

    The manifests authenticate bytes, not semantic document disjointness. The
    latter remains the data preparer's responsibility. No BOS/EOS is inserted.
    """
    def __init__(self, path, digest, *, split):
        path = Path(path).resolve()
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, 'Packed manifest SHA256 mismatch')
        manifest = json.loads(raw)
        require(manifest.get('schema') == 'packed_chain_adaptation_v1'
                and manifest.get('split') == split and split in ('train', 'dev'),
                'Explicit separate train/dev adaptation manifests required')
        # Reuse the documented packed-window/count input invariants, not its
        # train-only declaration. The original manifest is retained unchanged.
        validate_count_manifest(dict(manifest, schema='packed_count_training_v1', split='train'))
        self.arrays, self.ends, self.paths = [], [], set()
        total = 0
        for shard in manifest['shards']:
            source = (path.parent / shard['path']).resolve()
            require(source not in self.paths, 'Duplicate resolved shard')
            require(file_sha256(source) == shard['sha256'], 'Packed shard SHA256 mismatch')
            array = np.load(source, mmap_mode='r', allow_pickle=False)
            require(isinstance(array, np.ndarray) and array.dtype.kind in 'iu'
                    and array.shape == (shard['rows'], manifest['window_length']),
                    'Expected integer [rows,window_length] .npy windows')
            for start in range(0, len(array), 256):
                chunk = array[start:start + 256]
                require(bool(np.all((chunk >= 0) & (chunk < manifest['mask_id']))),
                        'Packed rows contain MASK/PAD or out-of-vocabulary IDs')
            require(file_sha256(source) == shard['sha256'], 'Packed shard changed during validation')
            self.paths.add(source)
            self.arrays.append(array)
            total += len(array)
            self.ends.append(total)
        self.manifest = manifest
        self.identity = {'manifest_sha256': digest, 'manifest': manifest,
                         'rows': total, 'document_disjointness_independently_verified': False}

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        import bisect
        require(type(index) is int and 0 <= index < len(self), 'Packed row outside declared split')
        group = bisect.bisect_right(self.ends, index)
        offset = index - (self.ends[group - 1] if group else 0)
        return torch.from_numpy(np.array(self.arrays[group][offset], dtype=np.int64, copy=True))


def validate_splits(training, development):
    require(not (training.paths & development.paths), 'Train/dev share a shard path')
    left = {s['sha256'] for s in training.manifest['shards']}
    right = {s['sha256'] for s in development.manifest['shards']}
    require(not (left & right), 'Train/dev share identical shard bytes')
    for key in ('vocab_size', 'window_length', 'mask_id', 'eos_id', 'tokenizer',
                'document_eos_retained', 'window_bos_eos_added'):
        require(training.manifest[key] == development.manifest[key],
                'Train/dev tokenization or packing differs: ' + key)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def tree_equal(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and torch.equal(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(tree_equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(tree_equal(x, y) for x, y in zip(a, b)))
    return type(a) is type(b) and a == b


def atomic(path, writer):
    """Replace one complete file; a failed writer preserves the prior checkpoint."""
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix='training-', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        writer(Path(name))
        with open(name, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(name, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    atomic(path, lambda temporary: temporary.write_text(canonical(value) + '\n'))


def immutable_head(path, payload):
    if Path(path).exists():
        require(tree_equal(torch.load(path, map_location='cpu', weights_only=True), payload),
                'Immutable head snapshot differs')
    else:
        atomic(path, lambda temporary: torch.save(payload, temporary))
    return {'immutable_head_path': str(Path(path).resolve()),
            'immutable_head_sha256': file_sha256(path)}


@contextmanager
def output_lock(output, resume):
    """One writer; stale lock files are harmless after a process exits (POSIX)."""
    import fcntl
    output = Path(output)
    if not resume:
        require(not output.exists() or not any(output.iterdir()),
                'New run needs an empty output directory; use --resume for continuation')
    else:
        require((output / 'last.pt').is_file(), '--resume needs a durable last.pt')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.training.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another process owns this training output') from error
        if not resume:
            require(not any(p.name != '.training.lock' for p in output.iterdir()),
                    'Output acquired by another run before initialization')
        yield


def pending_development(state, config):
    step = state['step']
    recorded = [row['step'] for row in state['development']]
    require(len(recorded) == len(set(recorded)) and all(0 < s <= step for s in recorded),
            'Invalid or duplicate development history')
    require(all(s in recorded for s in range(config.eval_every, step, config.eval_every)),
            'Older missing development event cannot be reconstructed from current weights')
    return step > 0 and (step % config.eval_every == 0 or step == config.max_updates) and step not in recorded


def commit_development(state, config, evaluate, persist_best, save):
    """Snapshot first, then record selection; interrupted dev is safely replayable."""
    if not pending_development(state, config):
        return None
    row = dict(evaluate(), step=state['step'])
    require(math.isfinite(row['mean_nll']), 'Nonfinite development objective')
    best, stale = state['best'], state['stale']
    if best is None or row['mean_nll'] < best['mean_nll'] - 1e-5:
        best = dict(step=state['step'], mean_nll=row['mean_nll'], **persist_best())
        stale = 0
    else:
        stale += 1
    state.update(best=best, stale=stale, development=[*state['development'], row],
                 complete=(state['step'] >= config.max_updates or
                           (state['step'] >= config.min_updates and stale >= config.patience)))
    save()
    return row


def evaluate_rows(development, frozen, count_potential, head, config, global_head=None):
    generator = torch.Generator(device=next(head.parameters()).device).manual_seed(104729)
    training = head.training
    rows, total, selected = [], 0., 0
    head.eval()
    try:
        with torch.no_grad():
            for i in range(config.dev_blocks):
                result = adaptation_step(development[i][None].to(next(head.parameters()).device),
                    frozen=frozen, count_potential=count_potential, head=head, generator=generator,
                    global_head=global_head, global_strength=config.global_strength,
                    top_k=config.top_k, nucleus_p=config.top_p, schedule_steps=config.schedule_steps)
                rows.append(result.metrics)
                total += result.metrics.get('block_nll', 0.)
                selected += result.metrics['revealed_tokens']
    finally:
        head.train(training)
    require(selected > 0, 'Empty development target set')
    return dict(metric=METRIC, block_nll=total, revealed_tokens=selected, mean_nll=total / selected,
                rows=rows, development_blocks=config.dev_blocks, selected_on_test=False)


def head_configuration(head):
    result = {'vocab_size': head.vocab_size, 'rank': head.rank}
    if isinstance(head, ContextualAdditiveHead):
        result.update(hidden_size=head.hidden_size, context_size=head.context_size)
    return result


def make_identity(config, frozen, head, training, development, count_identity,
                  inherited, order, device):
    return dict(schema='additive_ladder_training_identity_v1', kind=config.kind,
        head_config=head_configuration(head), model=frozen.identity,
        data={'training': training.identity, 'development': development.identity, 'test_rows_exposed': False},
        sources=source_identity(), counts=count_identity, count_strength=config.count_strength,
        inherited_global=inherited, global_strength=config.global_strength,
        cap=config.top_k, top_p=config.top_p, schedule_steps=config.schedule_steps,
        optimizer='AdamW', learning_rate=config.learning_rate, weight_decay=0., gradient_clip_norm=1.,
        batch_size=1, max_updates=config.max_updates, min_updates=config.min_updates,
        eval_every=config.eval_every, dev_blocks=config.dev_blocks, patience=config.patience,
        improvement_threshold=1e-5, checkpoint_every=config.checkpoint_every,
        objective='joint reveal logZ-logZ_clamped; candidate-set surrogate; NOT data PPL',
        support='frozen top-p/top-K with gold injection only at selected targets',
        residual_training_coefficient=1., count_mode='conditional', smoothing=.1,
        order_sha256=value_sha(order), fixed_rng={'initialization': 1, 'corruption': 7919,
            'order': 15485863, 'development': 104729}, one_training_run=True, no_seed_sweep=True,
        implementation='portable dense reference, not historical CUDA-training bitwise/runtime reproduction',
        runtime={'torch': str(torch.__version__), 'device': str(device), 'cuda': torch.version.cuda,
                 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32})


def run_training(*, training, development, frozen, count_potential, count_identity,
                 head, config, output, global_head=None, inherited=None, resume=False,
                 phase='train', max_new_updates=None, seconds_limit=36000., stop_requested=lambda: False):
    """Train one configuration. Operational slice limits do not change its identity.

    Unexpected failures preserve the most recent complete checkpoint; they never
    save a half-applied optimizer transaction. Graceful signals stop at a row
    boundary and persist the current state. Development failures occur after an
    explicit checkpoint and are replayed before any further optimizer update.
    """
    config.validate()
    require(phase in ('train', 'preflight') and math.isfinite(seconds_limit) and seconds_limit > 0,
            'Finite positive budget and valid phase required')
    require(max_new_updates is None or (type(max_new_updates) is int and max_new_updates > 0),
            'max_new_updates must be positive')
    require(len(training) > 0 and len(development) >= config.dev_blocks, 'Insufficient train/dev rows')
    require(isinstance(head, GlobalAdditiveHead if config.kind == 'global' else ContextualAdditiveHead)
            and head.rank == config.rank, 'Head/config mismatch')
    require((global_head is not None) == (config.kind == 'contextual')
            and (inherited is not None) == (config.kind == 'contextual'), 'Wrong frozen predecessor')
    if global_head is not None:
        require(not global_head.training and all(not p.requires_grad for p in global_head.parameters()),
                'Inherited global head must be frozen/eval')
    device = next(head.parameters()).device
    output = Path(output).resolve()
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(15485863)).tolist()
    identity = make_identity(config, frozen, head, training, development, count_identity, inherited, order, device)
    digest = value_sha(identity)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=0.)
    generator = torch.Generator(device=device).manual_seed(7919)
    state = dict(schema='portable_additive_train_state_v1', identity_sha256=digest, step=0, cursor=0,
                 best=None, stale=0, complete=False, initial_development=None, training=[], development=[])
    deadline = time.monotonic() + seconds_limit

    def export():
        return dict(schema='additive_ladder_head_v1', kind=config.kind, head_config=head_configuration(head),
                    head_state_dict=cpu_tree(head.state_dict()), model_revision=CHECKPOINT_REVISION,
                    training_identity=identity, training_step=state['step'])

    def save():
        payload = dict(state, identity=identity, head_state_dict=cpu_tree(head.state_dict()),
            optimizer=cpu_tree(optimizer.state_dict()), generator=generator.get_state().cpu(),
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None)
        atomic(output / 'last.pt', lambda path: torch.save(payload, path))
        # last.pt is authoritative. These convenience files may lag after a crash.
        write_json(output / 'identity.json', identity)
        write_json(output / 'progress.json', {k: state[k] for k in
            ('step', 'cursor', 'best', 'stale', 'complete', 'initial_development')})

    def persist_best():
        payload = export()
        receipt = immutable_head(output / f"head-step{state['step']:06d}.pt", payload)
        atomic(output / 'best-head.pt', lambda path: torch.save(payload, path))
        return receipt

    def evaluate():
        return evaluate_rows(development, frozen, count_potential, head, config, global_head)

    def gradients(loss, require_nonzero=False):
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.))
        require(math.isfinite(norm) and (not require_nonzero or norm > 0), 'Invalid head gradient')
        if hasattr(frozen, 'model'):
            require(all(p.grad is None and not p.requires_grad for p in frozen.model.parameters()),
                    'Backbone must remain frozen without gradients')
        if global_head is not None:
            require(all(p.grad is None for p in global_head.parameters()), 'Inherited head acquired gradients')
        return norm

    with output_lock(output, resume):
        if resume:
            restored = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
            require(restored.get('schema') == state['schema'] and restored.get('identity') == identity
                    and restored.get('identity_sha256') == digest, 'Resume identity mismatch')
            if (output / 'identity.json').exists():
                require(json.loads((output / 'identity.json').read_text()) == identity, 'Saved identity mismatch')
            head.load_state_dict(restored.pop('head_state_dict'), strict=True)
            optimizer.load_state_dict(restored.pop('optimizer'))
            generator.set_state(restored.pop('generator'))
            torch.set_rng_state(restored.pop('cpu_rng'))
            cuda_rng = restored.pop('cuda_rng')
            if device.type == 'cuda':
                torch.cuda.set_rng_state(cuda_rng, device)
            restored.pop('identity')
            state = restored
            pending_development(state, config)  # Reject historical gaps before any update.
            if state['best'] is not None:
                best = state['best']
                require(file_sha256(best['immutable_head_path']) == best['immutable_head_sha256'],
                        'Saved best head is missing or changed')
        else:
            save()  # Durable zero-step initialization before forward/preflight.
        if phase == 'preflight':
            require(state['step'] == 0, 'Preflight requires the unchanged zero-step initialization')
            before = generator.get_state()
            try:
                result = adaptation_step(training[order[0]][None].to(device), frozen=frozen,
                    count_potential=count_potential, head=head, generator=generator, global_head=global_head,
                    global_strength=config.global_strength, top_k=config.top_k, nucleus_p=config.top_p,
                    schedule_steps=config.schedule_steps)
                require(result.loss is not None, 'Preflight drew no targets; do not force a gold target')
                norm = gradients(result.loss, require_nonzero=True)
                proof = dict(passed=True, gradient_norm=norm, metrics=result.metrics,
                             identity_sha256=digest, optimizer_updates=0)
                write_json(output / 'preflight.json', proof)
                return proof
            finally:
                generator.set_state(before)
                optimizer.zero_grad(set_to_none=True)
        head.train()
        if state['initial_development'] is None:
            state['initial_development'] = evaluate()
            save()
        start = state['step']
        while not state['complete'] or pending_development(state, config):
            if stop_requested() or time.monotonic() >= deadline:
                break
            if pending_development(state, config):
                save()
                commit_development(state, config, evaluate, persist_best, save)
                continue
            if state['step'] >= config.max_updates or (max_new_updates is not None and state['step'] - start >= max_new_updates):
                break
            index = order[state['cursor'] % len(order)]
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            tick = time.monotonic()
            result = adaptation_step(training[index][None].to(device), frozen=frozen,
                count_potential=count_potential, head=head, generator=generator, global_head=global_head,
                global_strength=config.global_strength, top_k=config.top_k, nucleus_p=config.top_p,
                schedule_steps=config.schedule_steps)
            if result.loss is None:
                state['cursor'] += 1
                continue
            norm = gradients(result.loss)
            optimizer.step()
            state['cursor'] += 1
            state['step'] += 1
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            state['training'].append(dict(step=state['step'], row_index=index, gradient_norm=norm,
                seconds=time.monotonic() - tick, **result.metrics))
            if pending_development(state, config) or state['step'] % config.checkpoint_every == 0:
                save()
        save()
        write_json(output / 'training-history.json', state)
        require(source_identity() == identity['sources'], 'Training sources changed during execution')
        return state


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('upstream', 'checkpoint', 'tokenizer', 'train-manifest', 'dev-manifest', 'count-file', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('train-manifest', 'dev-manifest', 'count'):
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--kind', choices=('global', 'contextual'), required=True)
    p.add_argument('--global-head', type=Path)
    p.add_argument('--global-sha256')
    p.add_argument('--global-strength', type=float, default=0.)
    p.add_argument('--count-strength', type=float, default=.1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--phase', choices=('preflight', 'train'), default='train')
    p.add_argument('--seconds-limit', type=float, required=True)
    p.add_argument('--max-new-updates', type=int)
    for name in ('rank', 'context-size', 'top-k', 'schedule-steps', 'max-updates', 'min-updates',
                 'eval-every', 'dev-blocks', 'patience', 'checkpoint-every'):
        p.add_argument('--' + name, type=int, default=getattr(TrainingConfig(), name.replace('-', '_')))
    p.add_argument('--top-p', type=float, default=.9)
    p.add_argument('--learning-rate', type=float, default=.001)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = TrainingConfig(**{name: getattr(args, name) for name in TrainingConfig.__dataclass_fields__})
    config.validate()
    require((args.global_head is not None) == (args.global_sha256 is not None), 'Supply both global head path/SHA')
    require((args.global_head is not None) == (config.kind == 'contextual'), 'Only contextual training inherits a global head')
    training = PackedRows(args.train_manifest, args.train_manifest_sha256, split='train')
    development = PackedRows(args.dev_manifest, args.dev_manifest_sha256, split='dev')
    validate_splits(training, development)
    m = training.manifest
    require(m['vocab_size'] == 50258 and m['mask_id'] == 50257 and m['eos_id'] == 50256
            and m['window_length'] == 1024 and m['tokenizer'] == {'id': TOKENIZER_REPO, 'revision': TOKENIZER_REVISION},
            'Packed data differs from the pinned MDLM tokenizer/window convention')
    require(torch.device(args.device).type in ('cpu', 'cuda'), 'Expected CPU or CUDA device')
    torch.manual_seed(1)
    frozen = load_bd3lm(args.upstream, args.checkpoint, args.tokenizer,
                        device=args.device, steps=config.schedule_steps, top_p=config.top_p)
    counts = assets.load_count_asset(args.count_file, args.count_sha256, vocab_size=frozen.mask_id + 1)
    potential = SparseCountPotential.from_counts(**{k: v.to(args.device) for k, v in counts.statistics.items()},
        strength=config.count_strength, mode='conditional', smoothing=.1)
    global_head, inherited = None, None
    if config.kind == 'global':
        head = GlobalAdditiveHead(frozen.mask_id + 1, rank=config.rank).to(args.device)
    else:
        asset = assets.load_head_asset(args.global_head, args.global_sha256,
                                      kind='global', device=args.device, vocab_size=frozen.mask_id + 1)
        assets.validate_training_predecessor(asset, count_strength=config.count_strength,
            cap=config.top_k, top_p=config.top_p, steps=config.schedule_steps,
            counts_sha256=counts.identity['sha256'])
        global_head = asset.head
        inherited = dict(path=asset.identity['path'], sha256=asset.identity['sha256'], step=asset.identity['training_step'])
        table = frozen.model.backbone.backbone.vocab_embed.embedding.detach()
        head = ContextualAdditiveHead(table, hidden_size=table.shape[1], rank=config.rank,
                                     context_size=config.context_size).to(args.device)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = run_training(training=training, development=development, frozen=frozen,
            count_potential=potential, count_identity=counts.identity, head=head, config=config,
            output=args.output, global_head=global_head, inherited=inherited, resume=args.resume,
            phase=args.phase, max_new_updates=args.max_new_updates, seconds_limit=args.seconds_limit,
            stop_requested=lambda: stopped)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(canonical({'output': str(args.output.resolve()), 'phase': args.phase,
                     'step': result.get('step', 0), 'complete': result.get('complete', False),
                     'best': result.get('best'), 'metric': METRIC}))
    return 0


if __name__ == '__main__':
    main()
