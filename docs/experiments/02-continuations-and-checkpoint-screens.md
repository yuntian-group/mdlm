# Continuations and pre-optimization checkpoint screens

Historical September 2026 results, preceding the optimized evaluation series.
Training continues the same shared-R16 four arms through 3k, 6k and 10k total
updates, restoring full checkpoints rather than restarting the optimizer.
Architecture, K=128, frozen backbone, training seed 1, length 1024 and batch 4
remain matched. Source checkpoint directories must be mapped to the portable
archive aliases in each script before reproducing an old continuation.

## Paired 6k versus 7k screen

| Arm | 6k PPL | 7k PPL | Generated samples per checkpoint | Sampling implementation |
|---|---:|---:|---:|---|
| SS | 84.591 | 75.939 | 2 | O: unoptimized |
| FD | 60.841 | 56.451 | 2 | O: unoptimized |
| DF | 70.503 | 64.646 | 2 | O: unoptimized |
| DD | 49.880 | 48.511 | 2 | O: unoptimized |

Sixteen generated samples total: four arms × two checkpoints × two seeds.
Each sample uses 1,000 reverse transitions plus cleanup, length 1024 and batch
1. GPT-2-large scores through the first nonleading EOS. PPL aggregates token
NLL across samples. The two samples are seeds 91001 and 91002; these are small
screens, not an uncertainty-controlled demonstration of improvement.

The paired summarizer supports larger seed sets. Its current default of five
does not change this historical result's sample count: invoke it with
`--num-seeds 2` for a two-seed archive.

## Earlier one-sample checkpoint screen

Every value below represents **one generated sample**, using sampler O and
the same 1,000-transition generation/scoring protocol. These values are not
pooled with the two-sample table above or with the later five-sample study.

| Arm | 3k | 6k | 7k | 8k | 10k |
|---|---:|---:|---:|---:|---:|
| SS | 111.725 | 85.392 | 85.392 | 81.631 | 148.739 |
| FD | 69.946 | 72.664 | 61.351 | 88.931 | 81.515 |
| DF | 137.529 | 71.753 | 81.108 | 78.142 | 77.524 |
| DD | 68.582 | 46.393 | 42.073 | 59.539 | 89.897 |

The identical SS values at 6k and 7k are retained as recorded, rather than
interpreted as evidence that nothing changed. Sampling variability and earlier
checkpoint selection prevent strong conclusions from these single samples.

## Reference evaluations

This milestone also preserves the original MDLM generation and held-out
evaluation entry points. Keep external GPT-2 generative PPL separate from
held-out MDLM likelihood/ELBO metrics. No additional baseline result is inferred
merely from a script's existence. Later notes record completed matched MDLM
baseline measurements with their actual sample counts.

Private login paths, email addresses and machine restrictions have been
removed. Activate the environment before submission, set `CCF_CACHE_ROOT`,
and supply your own scheduler notification address/options. Archived run
directory aliases preserve the distinctions between source checkpoints; they
do not identify a login or host. Historical results above were recovered from
saved summaries and were not rerun by this commit.
