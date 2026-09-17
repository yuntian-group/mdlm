# Matched four-arm CCF baseline

Historical September 2026 experiment, before the sampling speed optimizations.
These results were recovered from saved experiment summaries; this commit does
not rerun training or generation. The exact original submission date is not
preserved in the summary, so it is ordered before its continuation/variant work.

All arms use a frozen released MDLM backbone, shared rank-16 factors, top-K 128,
length 1024, training seed 1 and batch size 4. SS means fixed topology/factors;
FD means fixed topology/contextual factors; DF means contextual topology/fixed
factors; DD means contextual topology/factors.

| Arm | Checkpoint 500 PPL | Checkpoint 1000 PPL | Generated samples per cell | Sampler |
|---|---:|---:|---:|---|
| SS | 108.038 | 112.500 | 10 | O: original, before CCF speed fixes |
| FD | 135.469 | 109.117 | 10 | O: original, before CCF speed fixes |
| DF | 117.766 | 97.406 | 10 | O: original, before CCF speed fixes |
| DD | 87.636 | 77.674 | 10 | O: original, before CCF speed fixes |

Generation uses 1,000 reverse transitions plus the final cleanup allowance,
batch size 1 and seeds starting at 91001. PPL is GPT-2-large evaluator perplexity
of generated text, not MDLM likelihood or training NLL. Multi-sample PPL is
`exp(sum(token_count * mean_nll) / sum(token_count))`; it is not the arithmetic
mean of individual sample PPLs. The EOS scoring policy is the first nonleading
EOS, so scored lengths can differ. These small screens are exploratory.

The 500-update column comes from the historical step-500 evaluation; the
1,000-update column comes from the legacy compiled table. This commit retains
their recorded values and sample counts without claiming a new raw-data audit.

## Running the scripts on another machine

Activate an environment with the repository's dependencies before submission.
Set `CCF_CACHE_ROOT` to a directory containing `checkpoints/`, `huggingface/`,
and `runs/`. Explicitly export `CCF_CODE_ROOT` before `sbatch` so Slurm’s
spooled script uses the intended checkout or isolated snapshot. The backbone is expected at
`$CCF_CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt`.

```bash
export CCF_CODE_ROOT=/path/to/this/checkout
export CCF_CACHE_ROOT=/path/to/ccf-cache
sbatch scripts/train_four_ccf_matched_1k.sh

# Use the resulting export directory; adjacent hash files are required.
export ADAPTER_DIR=/path/to/training-output/exported_adapters
sbatch scripts/generate_four_ccf_matched.sh

# Evaluate the original training directory's verified step-500 best.ckpt.
export CCF_SOURCE_ROOT=/path/to/training-output
sbatch scripts/evaluate_four_ccf_step500.sh
```

Pass your own `--mail-user` and any site-specific scheduler options to `sbatch`.
The scripts retain notification event types, but embed no email address, login,
host restriction or private home path. Slurm logs go to the submission directory.
The archived adapter-hash fallback applies only to the historical exports;
new exports use their adjacent hash files.

This milestone uses the existing [MDLM](https://arxiv.org/abs/2406.07524) and
forest generation/scoring infrastructure. The text-loading utility does not
introduce a new model or scoring algorithm.
