#!/bin/bash
#SBATCH --job-name=ccf-speed-v2
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
cd "${CCF_CODE_ROOT:?}"
export HF_HUB_CACHE=${CCF_CACHE_ROOT}/huggingface
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
RUN_ROOT="${CCF_CACHE_ROOT}/runs/ccf_speed_v2_job${SLURM_JOB_ID:?}"
mkdir -p "$RUN_ROOT"
ATTEMPT=$(mktemp -d "$RUN_ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
printf '%s\n' "${CCF_LOCAL_FIX_COMMIT:?}" > "$ATTEMPT/local-fix-commit.txt"
sha256sum structured_utils.py structured_objective.py \
  scripts/verify_ccf_sampling_v2.py tests/test_ccf_sampling_optimization.py \
  > "$ATTEMPT/code-sha256.txt"
srun --ntasks=1 python -m unittest tests.test_ccf_sampling_optimization -v \
  > "$ATTEMPT/tests.log" 2>&1
srun --ntasks=1 python -u scripts/verify_ccf_sampling_v2.py --device cuda \
  --real-step-check > "$ATTEMPT/report.json"
echo "V2 verification passed. Results: $ATTEMPT"
