"""Generate saved samples from existing local MDLM/count/head assets.

Run ``python -m additive_chain.generate_cli --help``. This is a finite,
single-window command, not a durable checkpoint/resume experiment scheduler.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from . import assets
from .backbone import load_bd3lm, file_sha256
from .generation import ChainDecoder, generate_saved


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--upstream', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--tokenizer', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--samples', required=True, type=int)
    p.add_argument('--population', choices=('accepted', 'raw'), default='accepted',
                   help='Stop at this many accepted outputs, or exactly this many raw attempts')
    p.add_argument('--stream-namespace', required=True)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--top-k', type=int, default=500)
    p.add_argument('--top-p', type=float, default=.9)
    p.add_argument('--device', default='cuda')
    p.add_argument('--count-file', type=Path)
    p.add_argument('--count-sha256')
    p.add_argument('--count-strength', type=float, default=0.)
    p.add_argument('--count-mode', choices=('conditional', 'pmi'), default='conditional')
    p.add_argument('--smoothing', type=float, default=.1)
    for name in ('global', 'contextual'):
        p.add_argument('--' + name + '-head', type=Path)
        p.add_argument('--' + name + '-sha256')
        p.add_argument('--' + name + '-strength', type=float, default=0.)
    p.add_argument('--allow-recalibration', action='store_true')
    p.add_argument('--sampling', choices=('joint', 'own-marginal'), default='joint')
    p.add_argument('--backend', choices=('fused', 'static'), default='fused')
    p.add_argument('--fast-candidates', action='store_true')
    p.add_argument('--no-prune-irrelevant', action='store_true')
    p.add_argument('--strict-unary-validation', action='store_true')
    p.add_argument('--segment-batch-size', type=int, default=32)
    p.add_argument('--no-warmup', action='store_true')
    p.add_argument('--table-gib', type=float, default=12.)
    p.add_argument('--scratch-mib', type=float, default=128.)
    p.add_argument('--reserve-gib', type=float, default=2.)
    return p


def validate_options(a):
    assets.require(a.population in ('accepted', 'raw'), 'Unknown generation population')
    assets.require(a.samples > 0 and a.steps > 0 and 1 <= a.top_k <= 2048
        and 0 < a.top_p <= 1 and 0 < a.smoothing <= 1 and a.segment_batch_size > 0,
        'Invalid samples, schedule, support or smoothing')
    assets.require(bool(a.stream_namespace.strip()), 'Declare a nonempty stream namespace')
    for name in ('count', 'global', 'contextual'):
        strength = getattr(a, name + '_strength')
        assets.number(strength, name + ' strength')
        path = getattr(a, name + ('_file' if name == 'count' else '_head'))
        digest = getattr(a, name + '_sha256')
        assets.require((path is None) == (digest is None), 'Supply both ' + name + ' path and SHA256')
        assets.require(not strength or path is not None, 'Positive ' + name + ' strength requires an asset')
    assets.require(not a.contextual_strength or a.global_head is not None,
        'Contextual inference needs the exact inherited global asset, even if its coefficient is zero')
    for key in ('table_gib', 'scratch_mib', 'reserve_gib'):
        assets.number(getattr(a, key), key, positive=key != 'reserve_gib')
    a.output = a.output.resolve()
    assets.require(not a.output.exists() and a.output.parent.is_dir(),
        'Output must be a new file in an existing directory; never overwrite results')
    device = torch.device(a.device)
    assets.require(device.type in ('cpu', 'cuda'), 'This adapter supports CPU baseline or NVIDIA CUDA')
    assets.require(device.type == 'cuda' or not any((a.count_strength, a.global_strength, a.contextual_strength)),
        'Active chain inference requires NVIDIA CUDA/Triton; no CPU inference fallback')
    if device.type == 'cuda':
        assets.require(torch.cuda.is_available(), 'CUDA is unavailable')
    return a


def source_identity():
    # Public scientific modules only; no repository, test, or operational files.
    root = Path(__file__).resolve().parent
    return {p.name: file_sha256(p) for p in sorted(root.glob('*.py'))}


def run(a):
    a = validate_options(a)
    sources = source_identity()
    frozen = load_bd3lm(a.upstream, a.checkpoint, a.tokenizer,
                        device=a.device, steps=a.steps, top_p=a.top_p)
    vocab = frozen.mask_id + 1
    count = None if a.count_file is None else assets.load_count_asset(
        a.count_file, a.count_sha256, vocab_size=vocab)
    heads = {kind: None if getattr(a, kind + '_head') is None else assets.load_head_asset(
        getattr(a, kind + '_head'), getattr(a, kind + '_sha256'),
        kind=kind, device=a.device, vocab_size=vocab) for kind in ('global', 'contextual')}
    changes = assets.validate_composition(heads['global'], heads['contextual'],
        count_strength=a.count_strength, global_strength=a.global_strength,
        contextual_strength=a.contextual_strength, count_mode=a.count_mode, smoothing=a.smoothing,
        cap=a.top_k, top_p=a.top_p, steps=a.steps, allow_recalibration=a.allow_recalibration)
    if a.contextual_strength:
        local = heads['contextual'].head
        embedding = frozen.model.backbone.backbone.vocab_embed.embedding.detach()
        assets.require(local.hidden_size == embedding.shape[1]
            and torch.equal(local.token_embeddings, embedding.float()),
            'Contextual frozen embedding table differs from this actual backbone')
    decoder = ChainDecoder(mask_id=frozen.mask_id, device=a.device,
        count_statistics=None if count is None else count.statistics,
        count_strength=a.count_strength, count_mode=a.count_mode, smoothing=a.smoothing,
        global_head=None if heads['global'] is None else heads['global'].head,
        global_strength=a.global_strength,
        contextual_head=None if heads['contextual'] is None else heads['contextual'].head,
        contextual_strength=a.contextual_strength, sampling=a.sampling, backend=a.backend,
        top_k=a.top_k, nucleus_p=a.top_p, fast_candidates=a.fast_candidates,
        prune_irrelevant=not a.no_prune_irrelevant, segment_batch_size=a.segment_batch_size,
        strict_unary_validation=a.strict_unary_validation)
    try:
        budget = {} if a.backend == 'fused' else dict(
            max_table_bytes=int(a.table_gib * 1024**3),
            max_temporary_bytes=int(a.scratch_mib * 1024**2), reserve_bytes=int(a.reserve_gib * 1024**3))
        setup = decoder.prepare(**budget)
        result = generate_saved(frozen.forward, decoder, frozen.model.tokenizer,
            samples=a.samples, steps=a.steps, length=1024,
            stream_namespace=a.stream_namespace, warmup=not a.no_warmup,
            population=a.population)
    finally:
        decoder.close()
    assets.require(source_identity() == sources, 'Scientific source changed during execution')
    result['assets'] = {'backbone': frozen.identity,
        'counts': None if count is None else count.identity,
        'heads': {kind: None if asset is None else asset.identity for kind, asset in heads.items()},
        'training_recalibration': changes,
        'count_configuration': {'requested_mode': a.count_mode, 'requested_strength': a.count_strength,
            'requested_smoothing': a.smoothing, 'stored_strength_applied': False}}
    result['source_sha256'] = sources
    result['setup'] = setup
    result['validation_scope'] = {'entrypoint': 'portable Python API and local asset imports',
        'actual_checkpoint_cuda_parity_asserted': False, 'historical_reproduction_asserted': False,
        'note': 'A successful command produces new samples; it is not an archived same-GPU backend proof.'}
    # Serialization is outside generation latency; x mode protects concurrent files too.
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        value = run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        p.exit(1, str(error) + '\n')
    accepted = sum(row['accepted'] for row in value['attempts'])
    print(json.dumps({'output': str(args.output), 'status': value['status'],
                      'accepted': accepted, 'raw_attempts': len(value['attempts'])}))
    return 0 if value['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
