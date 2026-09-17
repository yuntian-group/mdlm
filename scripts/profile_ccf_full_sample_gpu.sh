#!/bin/bash
#SBATCH --job-name=ccf-full-profile
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
RUN_ROOT="${CCF_CACHE_ROOT}/runs/ccf_full_profile_job${SLURM_JOB_ID:?}"
mkdir -p "$RUN_ROOT"
ATTEMPT=$(mktemp -d "$RUN_ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
printf '%s\n' "${CCF_LOCAL_PROFILE_COMMIT:?}" > "$ATTEMPT/local-profile-commit.txt"
sha256sum structured_utils.py structured_objective.py models/structured_decoder.py \
  diffusion.py evaluation/generation_harness.py scripts/profile_ccf_full_sample.py \
  > "$ATTEMPT/code-sha256.txt"
srun --ntasks=1 python -u scripts/profile_ccf_full_sample.py --output-dir "$ATTEMPT/profile"
echo "Profile output: $ATTEMPT/profile/breakdown.json"
