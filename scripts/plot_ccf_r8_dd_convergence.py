"""Plot existing training records without modifying source CSV/logs."""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    args = parser.parse_args()
    folder = args.data_dir
    with (folder / 'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    elapsed = {}
    for step, stamp in re.findall(r'Epoch 0:.*?(\d+)/\?\s+\[([\d:]+)<', (folder / 'training.out').read_text()):
        seconds = 0
        for component in stamp.split(':'):
            seconds = seconds * 60 + int(component)
        elapsed[int(step)] = max(seconds, elapsed.get(int(step), 0))
    # Config/source: loss = structured NLL + 0.1 * topology loss; aux weight = 0.
    training = [(int(r['step']) + 1,
                 float(r['train/structured/loss_step']) - 0.1 * float(r['train/structured/topology_loss_step']))
                for r in rows if r['train/structured/loss_step']]
    key = 'val/structured/conditional_nll_per_masked_token'
    validation = [(int(r['step']) + 1, float(r[key])) for r in rows if r[key]]
    x, y = np.array(training).T
    vx, vy = np.array(validation).T
    assert len(training) == 700 and len(validation) == 14
    assert all(int(s) in elapsed for s in vx)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(vy))
    # Trailing mean of the 50 logged minibatches (logs every 10 updates).
    smooth = np.convolve(y, np.ones(50) / 50, mode='valid')
    clock_steps = np.array(sorted(elapsed))
    clock_minutes = np.array([elapsed[int(s)] / 60 for s in clock_steps])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.2, 5.8))
    fig.suptitle('Separate rank-8 dynamic/dynamic: training convergence',
                 x=0.07, ha='left', fontsize=17, fontweight='bold', y=0.98)
    a.plot(x, y, color='#BBC7D5', lw=0.7, alpha=0.7, label='Logged minibatches (reconstructed NLL)')
    a.plot(x[49:], smooth, color='#24599E', lw=2.1, label='Trailing mean: 50 logged minibatches')
    a.set_title('Training NLL', loc='left', pad=15)
    a.set_ylabel('Conditional NLL (nats / masked token)')
    a.legend(loc='lower left', fontsize=9, frameon=False)
    b.plot(vx, vy, '-o', color='#B55327', lw=1.8, ms=5)
    best = int(np.argmin(vy))
    b.annotate(f'Best logged: {int(vx[best]):,} steps\nNLL {vy[best]:.3f}',
               xy=(vx[best], vy[best]), xytext=(2900, 5.665), fontsize=10,
               arrowprops={'arrowstyle': '-', 'color': '#B55327'})
    b.set_ylim(5.64, 5.94)
    b.set_title('Held-out validation NLL (zoomed scale)', loc='left', pad=15)
    b.set_ylabel('Conditional NLL (nats / masked token)')
    for ax in (a, b):
        ax.set_xlim(0, 7100)
        ax.set_xlabel('Training updates')
        ax.grid(axis='y', alpha=0.18)
        ax.axvline(2000, color='#777777', linestyle=':', lw=1)
    time_axis = b.secondary_xaxis('top', functions=(
        lambda s: np.interp(s, clock_steps, clock_minutes),
        lambda t: np.interp(t, clock_minutes, clock_steps)))
    time_axis.set_xlabel('Recorded training elapsed time (minutes)', labelpad=9)
    time_axis.set_xticks([0, 15, 30, 45, 60, 75])
    b.set_title('Held-out validation NLL (zoomed scale)', loc='left', y=1.27)
    fig.text(0.07, 0.075, 'Training: loss − 0.1 × topology loss; sampled every 10 updates. Validation: direct token-weighted NLL, every 500 updates.', fontsize=9)
    fig.text(0.07, 0.04, 'Elapsed time comes from progress logs (includes validation/checkpoint pauses). Dotted line: step 2,000. These are not GPT-2 generation NLLs.', fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.19, top=0.74, wspace=0.24)
    fig.savefig(folder / 'training-convergence.png', dpi=180, facecolor='white')
    fig.savefig(folder / 'training-convergence.svg', facecolor='white')
    summary = {
        'training_run': 'ccf_separate_r8_dynamic_dynamic_s001_step7000_run-separate-r8-dd',
        'training_points': len(training), 'validation_points': len(validation),
        'first_1000_training_logged_batch_mean': float(y[x <= 1000].mean()),
        'last_1000_training_logged_batch_mean': float(y[x > 6000].mean()),
        'first_3_validation_mean': float(vy[:3].mean()),
        'last_3_validation_mean': float(vy[-3:].mean()),
        'final_epoch_token_weighted_training_nll': next(float(r['train/structured/conditional_nll_per_masked_token']) for r in rows if r['train/structured/conditional_nll_per_masked_token']),
        'validation': [{'step': int(s), 'nll': float(n), 'elapsed_minutes': elapsed[int(s)] / 60} for s, n in validation],
        'caveats': ['Training curve reconstructed from logged minibatches, not all updates or an epoch token-weighted curve.',
                    'Training OpenWebText; validation pinned WikiText-103, 32 batches of 4.',
                    'eval.corruption_seed is null: validation corruption is not fixed across checkpoints.',
                    'Conditional denoising NLL is not diffusion ELBO, ordinary text likelihood, or GPT-2 generation NLL.',
                    'Validation minima on this small subset do not establish an optimal generation checkpoint.'],
    }
    (folder / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
