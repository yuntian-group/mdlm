"""Paired corpus metrics for the frozen, three-seed IWSLT test comparison."""
import argparse
from pathlib import Path

import numpy as np

from .backbone import file_sha256
from .iwslt_confirm import (STEPS, TEST_SEEDS, atomic_json, check_output, ordered_rows,
                           read, select_settings, source_hashes, test_cells)


def paired_seed_metric(left, right, metric, *, replicates=2000, seed=1718):
    """Resample sentences jointly across methods AND decoding seeds.

    Report the mean of corpus scores over fixed decoding seeds, not a corpus
    formed by treating repeated translations of a sentence as new sentences.
    Cache additive metric statistics to avoid retokenizing each bootstrap.
    """
    if not left or len(left) != len(right) or not left[0] or replicates < 1:
        raise ValueError('Nonempty paired seeds and positive replicate count required')
    keys = [(r['id'], r['reference']) for r in left[0]]
    if len(set(k[0] for k in keys)) != len(keys):
        raise ValueError('Duplicate test sentence')
    for rows in left + right:
        if [(r['id'], r['reference']) for r in rows] != keys:
            raise ValueError('All methods and seeds require identical sentence ordering and references')
    refs = [[r['reference'] for r in left[0]]]
    stats = [np.asarray([metric._extract_corpus_statistics(
        [r['hypothesis'] for r in rows], refs) for rows in group]) for group in (left, right)]

    def score(s):
        return np.array([metric._compute_score_from_stats(x.tolist()).score for x in s.sum(axis=1)])

    a, b = [score(s) for s in stats]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        indices = rng.integers(0, len(keys), len(keys))
        aa, bb = [score(s[:, indices, :]) for s in stats]
        differences.append(float((aa - bb).mean()))
    return dict(left_by_seed=a.tolist(), right_by_seed=b.tolist(),
        left_mean=float(a.mean()), right_mean=float(b.mean()), delta=float((a-b).mean()),
        interval_95=np.quantile(differences, [.025, .975]).tolist(),
        examples=len(keys), decoding_seeds=len(left), replicates=replicates, bootstrap_seed=seed)


def run(args):
    from sacrebleu.metrics import BLEU, CHRF
    selection = read(args.selection)
    if selection['source_sha256'] != source_hashes() or select_settings(selection['development_cells']) != selection['selected']:
        raise ValueError('Selection or decoder source changed')
    if file_sha256(args.data / 'manifest.json') != selection['data_sha256']:
        raise ValueError('Data differ from frozen selection')
    ids = [r['id'] for r in ordered_rows(args.data, 'test')]
    selection_sha = file_sha256(args.selection)
    result = dict(selection_sha256=selection_sha, model_sha256=selection['model_sha256'],
        data_sha256=selection['data_sha256'], test_examples=len(ids), test_seeds=list(TEST_SEEDS),
        uncertainty='Paired sentence bootstrap with the same resampled indices for all methods and all three fixed decoding seeds; mean corpus score over seeds.',
        output_sha256={}, steps={})
    for steps in STEPS:
        root = args.input / f'step-{steps}'
        summary = read(root / 'summary.json')
        if summary['completed'] != summary['expected'] or summary['expected'] != 9:
            raise ValueError('Test comparison incomplete')
        groups = {'baseline': [], 'joint': [], 'own-marginal': []}
        for cell in test_cells(selection, steps):
            identity = dict(cell, split='test', ids=ids, model_sha256=selection['model_sha256'],
                data_sha256=selection['data_sha256'], source_sha256=selection['source_sha256'],
                selection_sha256=selection_sha)
            output = check_output(root, cell, identity)
            method = 'baseline' if cell['strength'] == 0 else cell['sampling']
            groups[method].append(output['predictions'])
            from .iwslt_confirm import cell_name
            path = root / (cell_name(cell) + '.json')
            result['output_sha256'][str(path.relative_to(args.input))] = file_sha256(path)
            for name, metric in [('bleu', BLEU(lowercase=True, tokenize='13a')), ('chrf', CHRF())]:
                rows = output['predictions']
                score = metric.corpus_score([r['hypothesis'] for r in rows], [[r['reference'] for r in rows]]).score
                if abs(score - output['summary'][name]) > 1e-10:
                    raise ValueError('Saved test score cannot be reproduced')
        for k in range(len(TEST_SEEDS)):
            lengths = [[r['predicted_length'] for r in groups[m][k]] for m in groups]
            if any(x != lengths[0] for x in lengths[1:]):
                raise ValueError('Methods used different predicted lengths')
        contrasts = {}
        for left, right in [('joint', 'own-marginal'), ('joint', 'baseline'), ('own-marginal', 'baseline')]:
            values = {}
            for name, metric in [('bleu', BLEU(lowercase=True, tokenize='13a')), ('chrf', CHRF())]:
                values[name] = paired_seed_metric(groups[left], groups[right], metric, replicates=args.replicates)
                values[name]['signature'] = str(metric.get_signature())
            contrasts[left + '_minus_' + right] = values
        result['steps'][str(steps)] = contrasts
        print('Finished paired test readout for ' + str(steps) + ' steps', flush=True)
    if file_sha256(args.selection) != selection_sha:
        raise ValueError('Selection changed during readout')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'data', 'selection', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--replicates', type=int, default=2000)
    run(p.parse_args())


if __name__ == '__main__':
    main()
