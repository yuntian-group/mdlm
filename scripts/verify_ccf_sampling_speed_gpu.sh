#!/bin/bash
#SBATCH --job-name=ccf-speed-verify
#SBATCH --time=00:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS

set -euo pipefail
# Activate the Python environment before submitting; supply site options to sbatch.
CCF_CODE_ROOT="${CCF_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${CCF_CACHE_ROOT:?Set CCF_CACHE_ROOT to your checkpoints/cache/runs directory}"
export CCF_CODE_ROOT CCF_CACHE_ROOT
cd "${CCF_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export HF_HUB_CACHE=${CCF_CACHE_ROOT}/huggingface
export TOKENIZERS_PARALLELISM=false
RUN_ROOT="${CCF_CACHE_ROOT}/runs/ccf_speed_verify_job${SLURM_JOB_ID:?}"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/.run.lock"
flock -n 9 || exit 2
ATTEMPT=$(mktemp -d "$RUN_ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
git rev-parse HEAD > "$ATTEMPT/git-commit.txt"
printf '%s\n' "${CCF_LOCAL_FIX_COMMIT:?Provide local fix commit provenance}" > "$ATTEMPT/local-fix-commit.txt"
sha256sum structured_objective.py structured_utils.py evaluation/generation_harness.py \
  scripts/verify_ccf_sampling_optimization.py > "$ATTEMPT/code-sha256.txt"
srun --ntasks=1 python -u scripts/verify_ccf_sampling_optimization.py \
  --device cuda --benchmark --real-step-check > "$ATTEMPT/report.log"
echo "All exact token/RNG checks passed. Report: $ATTEMPT/report.log"
