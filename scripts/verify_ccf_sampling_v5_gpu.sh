#!/bin/bash
#SBATCH --job-name=ccf-traversal-draw-test
#SBATCH --array=0-2%3
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
export TOKENIZERS_PARALLELISM=false
MODES=(tensor_traversal buffered_noise aligned_noise)
MODE="${MODES[${SLURM_ARRAY_TASK_ID:?}]}"
ROOT="${CCF_CACHE_ROOT}/runs/ccf_v5_speed_job${SLURM_ARRAY_JOB_ID}/$MODE"
mkdir -p "$ROOT"
ATTEMPT=$(mktemp -d "$ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
printf '%s\n' "${CCF_LOCAL_TEST_COMMIT:?}" > "$ATTEMPT/local-test-commit.txt"
sha256sum structured_utils.py structured_objective.py models/structured_decoder.py diffusion.py \
  evaluation/generation_harness.py scripts/audit_ccf_sampling_v3.py scripts/audit_ccf_sampling_v4.py \
  scripts/verify_ccf_sampling_v3_gpu.py > "$ATTEMPT/code-sha256.txt"
srun --ntasks=1 python -u scripts/verify_ccf_sampling_v3_gpu.py --mode "$MODE" \
  --baseline-mode unchecked_batched_roots_kruskal --output-dir "$ATTEMPT/results"
