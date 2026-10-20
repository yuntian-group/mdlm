"""Full-development selection followed by a frozen IWSLT test comparison."""
import argparse
import json
import math
from pathlib import Path
import random

from .backbone import file_sha256

STEPS = (8, 32)
TEST_SEEDS = (101, 202, 303)
COUNTS = {'dev': 7283, 'test': 6750}
SOURCE_FILES = ('iwslt_confirm.py', 'iwslt_run.py', 'translation.py', 'generation.py',
                'chain.py', 'counts.py', 'candidates.py', 'backbone.py')


def source_hashes():
    return {name: file_sha256(Path(__file__).parent / name) for name in SOURCE_FILES}


def read(path):
    return json.loads(path.read_text())


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def dev_cells(steps):
    if steps not in STEPS:
        raise ValueError('Unregistered step count')
    cells = [dict(steps=steps, temperature=t, strength=0., sampling='joint', seed=101)
             for t in (.85, 1.)]
    cells += [dict(steps=steps, temperature=1., strength=s, sampling=m, seed=101)
              for s in (.1, .2) for m in ('joint', 'own-marginal')]
    return cells


def cell_name(cell):
    return f"T{cell['steps']}-tau{cell['temperature']}-lambda{cell['strength']}-{cell['sampling']}-seed{cell['seed']}"


def select_settings(cells):
    """Choose by full-development BLEU, without reading any test output."""
    expected = [c for steps in STEPS for c in dev_cells(steps)]
    if len(cells) != len(expected) or {cell_name(c) for c in cells} != {cell_name(c) for c in expected}:
        raise ValueError('Incomplete or duplicate development grid')
    if any(not math.isfinite(c['bleu']) for c in cells):
        raise ValueError('Nonfinite development score')
    selected = {}
    for steps in STEPS:
        group = [c for c in cells if c['steps'] == steps]
        # Fixed tie breaks: lower temperature or weaker pair correction.
        baseline = max((c for c in group if c['strength'] == 0),
                       key=lambda c: (c['bleu'], -c['temperature']))
        scores = {s: sum(c['bleu'] for c in group if c['strength'] == s) / 2 for s in (.1, .2)}
        strength = max(scores, key=lambda s: (scores[s], -s))
        selected[str(steps)] = dict(baseline_temperature=baseline['temperature'],
            strength=strength, crf_temperature=1., baseline_dev_bleu=baseline['bleu'],
            mean_crf_dev_bleu=scores[strength])
    return selected


def test_cells(selection, steps):
    if steps not in STEPS or selection['test_seeds'] != list(TEST_SEEDS):
        raise ValueError('Unregistered test configuration')
    setting = selection['selected'][str(steps)]
    cells = []
    for seed in TEST_SEEDS:
        cells.append(dict(steps=steps, temperature=setting['baseline_temperature'],
                          strength=0., sampling='joint', seed=seed))
        cells.extend(dict(steps=steps, temperature=setting['crf_temperature'],
                          strength=setting['strength'], sampling=m, seed=seed)
                     for m in ('joint', 'own-marginal'))
    return cells


def ordered_rows(data, split):
    from .iwslt_run import dataset
    rows = dataset(data, split)
    if len(rows) != COUNTS[split] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Full registered split required')
    random.Random(31415).shuffle(rows)
    return rows


def check_output(root, cell, identity):
    name = cell_name(cell)
    receipt = read(root / (name + '-identity.json'))
    if receipt['identity'] != identity or receipt['output_sha256'] != file_sha256(root / (name + '.json')):
        raise ValueError('Output or registered settings changed: ' + name)
    output = read(root / (name + '.json'))
    rows = output['predictions']
    if [r['id'] for r in rows] != identity['ids'] or len(rows) != output['summary']['examples']:
        raise ValueError('Incomplete or misordered predictions')
    return output


def freeze(args):
    from sacrebleu.metrics import BLEU
    from .iwslt_run import run_lock
    with run_lock(args.output):
        rows = ordered_rows(args.data, 'dev')
        ids = [r['id'] for r in rows]
        data_sha = file_sha256(args.data / 'manifest.json')
        model_sha = file_sha256(args.trained)
        sources = source_hashes()
        cells, artifacts = [], {}
        for steps in STEPS:
            root = args.development / f'step-{steps}'
            summary = read(root / 'summary.json')
            if summary['completed'] != summary['expected'] or summary['expected'] != 6:
                raise ValueError('Development decoding is unfinished')
            for cell in dev_cells(steps):
                identity = dict(cell, split='dev', ids=ids, model_sha256=model_sha,
                                data_sha256=data_sha, source_sha256=sources, selection_sha256=None)
                result = check_output(root, cell, identity)
                pred = result['predictions']
                score = BLEU(lowercase=True, tokenize='13a').corpus_score(
                    [r['hypothesis'] for r in pred], [[r['reference'] for r in pred]]).score
                if abs(score - result['summary']['bleu']) > 1e-10:
                    raise ValueError('Development BLEU cannot be reproduced')
                cells.append(dict(cell, bleu=score))
                for suffix in ('.json', '-identity.json'):
                    path = root / (cell_name(cell) + suffix)
                    artifacts[str(path.relative_to(args.development))] = file_sha256(path)
        selection = dict(selected=select_settings(cells), development_cells=cells,
            model_sha256=model_sha, data_sha256=data_sha, source_sha256=sources,
            development_artifacts=artifacts, development_examples=len(rows),
            test_seeds=list(TEST_SEEDS), steps=list(STEPS),
            policy='Baseline temperature maximizes full-development BLEU; common CRF maximizes mean joint/own-marginal full-development BLEU at each step count. Test settings are not retuned.')
        path = args.output / 'selection.json'
        if path.exists() and read(path) != selection:
            raise ValueError('Refusing to change frozen selection')
        atomic_json(path, selection)
        print(json.dumps(selection['selected'], indent=2), flush=True)


def decode(args):
    import torch
    from .generation import ChainDecoder
    from .iwslt_run import log, run_lock, score_translations
    from .translation import LengthPredictor, load_conditional, MASK_ID
    with run_lock(args.output):
        model_sha = file_sha256(args.trained)
        data_sha = file_sha256(args.data / 'manifest.json')
        sources = source_hashes()
        selection_sha = None
        if args.split == 'test':
            if not args.selection:
                raise ValueError('Test decoding requires frozen development selection')
            selection = read(args.selection)
            if (selection['model_sha256'], selection['data_sha256'], selection['source_sha256']) != (model_sha, data_sha, sources):
                raise ValueError('Test model/data/code differ from frozen selection')
            if select_settings(selection['development_cells']) != selection['selected']:
                raise ValueError('Frozen selection does not follow development scores')
            selection_sha = file_sha256(args.selection)
            cells = test_cells(selection, args.steps)
        else:
            if args.selection:
                raise ValueError('Development does not consume test selection')
            cells = dev_cells(args.steps)
        rows = ordered_rows(args.data, args.split)
        ids = [r['id'] for r in rows]
        counts_path = args.data / 'counts.pt'
        if file_sha256(counts_path) != read(args.data / 'manifest.json')['counts_sha256']:
            raise ValueError('Training counts changed')
        model, tokenizer = load_conditional(args.checkpoint, args.tokenizer)
        saved = torch.load(args.trained, map_location='cuda', weights_only=False)
        model.load_state_dict(saved['model'])
        head = LengthPredictor(model.pretrained.config.hidden_dim).cuda()
        head.load_state_dict(saved['length_head'])
        del saved
        model.eval()
        head.eval()
        counts = torch.load(counts_path, weights_only=True)
        completed = []
        log('confirmation_started', split=args.split, steps=args.steps, examples=len(rows),
            cells=len(cells), gpu=torch.cuda.get_device_name(), selection_sha256=selection_sha)
        for cell in cells:
            name = cell_name(cell)
            output, receipt = args.output / (name + '.json'), args.output / (name + '-identity.json')
            identity = dict(cell, split=args.split, ids=ids, model_sha256=model_sha,
                            data_sha256=data_sha, source_sha256=sources, selection_sha256=selection_sha)
            if receipt.exists():
                summary = check_output(args.output, cell, identity)['summary']
            else:
                if output.exists():
                    raise ValueError('Output lacks completion receipt: ' + name)
                decoder = ChainDecoder(mask_id=MASK_ID, device='cuda', fast_candidates=True,
                    count_statistics=counts, count_strength=cell['strength'],
                    sampling=cell['sampling'], temperature=cell['temperature'])
                decoder.prepare()
                pending = output.with_suffix('.pending.json')
                try:
                    summary = score_translations(model, tokenizer, head, rows, decoder, pending,
                                                 steps=cell['steps'], seed=cell['seed'])
                finally:
                    decoder.close()
                if file_sha256(args.trained) != model_sha:
                    raise ValueError('Checkpoint changed during decoding')
                if args.selection and file_sha256(args.selection) != selection_sha:
                    raise ValueError('Frozen selection changed during decoding')
                pending.replace(output)
                atomic_json(receipt, dict(identity=identity, output_sha256=file_sha256(output)))
            completed.append(dict(cell, **summary))
            atomic_json(args.output / 'summary.json', dict(completed=len(completed), expected=len(cells),
                cells=completed, split=args.split, model_sha256=model_sha, data_sha256=data_sha,
                selection_sha256=selection_sha))
            log('confirmation_cell', split=args.split, **cell, **summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('decode')
    for name in ('checkpoint', 'tokenizer', 'data', 'trained', 'output'):
        d.add_argument('--' + name, type=Path, required=True)
    d.add_argument('--split', choices=['dev', 'test'], required=True)
    d.add_argument('--steps', type=int, choices=STEPS, required=True)
    d.add_argument('--selection', type=Path)
    f = sub.add_parser('freeze')
    for name in ('data', 'trained', 'development', 'output'):
        f.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    (decode if args.command == 'decode' else freeze)(args)


if __name__ == '__main__':
    main()
