#!/bin/bash
#SBATCH --job-name=ccf-r8-dd2k-seeds
#SBATCH --array=0-8%3
#SBATCH --time=00:20:00
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
CACHE_ROOT=${CCF_CACHE_ROOT}
cd "$CACHE_ROOT/code/ccf_sampling_speed_v2_checked.iZJBw9"
SEED=$((91002 + ${SLURM_ARRAY_TASK_ID:?Submit as array}))
SOURCE="$CACHE_ROOT/runs/ccf_separate_checkpoints_run-separate-r8-checkpoints/dynamic_dynamic/step2000/attempt-0.sGG8eC"
ADAPTER="$SOURCE/dynamic_dynamic.safetensors"
MANIFEST="$SOURCE/dynamic_dynamic.manifest.json"
BACKBONE="$CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt"
ROOT="$CACHE_ROOT/runs/ccf_r8_dd_step2000_more_seeds_job${SLURM_ARRAY_JOB_ID}/seed$SEED"
mkdir -p "$ROOT"
ATTEMPT=$(mktemp -d "$ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
export HF_HUB_CACHE="$CACHE_ROOT/huggingface"
export TOKENIZERS_PARALLELISM=false
sha256sum structured_utils.py structured_objective.py diffusion.py models/structured_decoder.py evaluation/generation_harness.py scripts/run_generation_pilot.py "$ADAPTER" "$MANIFEST" > "$ATTEMPT/source-sha256.txt"
printf '%s\n' "${CCF_EVAL_COMMIT:?Record local script commit}" > "$ATTEMPT/local-eval-commit.txt"
printf '%s\n' 'Production v2 commit 4fc49fb7a82290d235aad332d64e44bb1e4e8fd7; GPU verification 1544303 passed. Independent one-sample jobs: pair key replicate-0000, seeds 91002–91010.' > "$ATTEMPT/provenance.txt"
srun --ntasks=1 python -u scripts/run_generation_pilot.py \
  --backbone-checkpoint "$BACKBONE" --backbone-sha256 7508daae475e7c0aa39dd7014e786fa9788fe1fc37f040c076e2df021e45f605 \
  --adapter "$ADAPTER" --adapter-sha256 e72e981045ba19c0867e313e74a36cc1066fbe9da6c3a8f54a55c7dbf0901255 \
  --adapter-manifest "$MANIFEST" --adapter-manifest-sha256 "$(sha256sum "$MANIFEST" | awk '{print $1}')" \
  --output-dir "$ATTEMPT/generation" --num-samples 1 --sequence-length 1024 \
  --batch-size 1 --base-seed "$SEED" --modes structured_joint --nfe-budgets 1001 \
  --device cuda --model-config contextual-forest-small \
  --data-config train_openwebtext_pinned --allow-dirty \
  --reference-lm gpt2-large --reference-lm-revision 32b71b12589c2f8d625668d2335a01cac3249519 \
  --reference-lm-device cuda --reference-lm-batch-size 1 --reference-lm-max-length 1024 --reference-lm-dtype float32 \
  --override "data.cache_dir=$HF_HUB_CACHE" \
  --override model.structured_decoder.top_k=128 --override model.structured_decoder.rank=8 \
  --override ++model.structured_decoder.factor_embedding_mode=separate \
  --override ++model.structured_decoder.factor_conditioner_hidden_dim=0 \
  --override model.structured_decoder.topology_mode=dynamic --override model.structured_decoder.factor_mode=dynamic \
  --override model.structured_decoder.independent_mode=false \
  --override model.structured_decoder.training.topology_weight=0.1 \
  --override "checkpointing.save_dir=$ATTEMPT"
echo "Complete seed $SEED: $ATTEMPT/generation"
