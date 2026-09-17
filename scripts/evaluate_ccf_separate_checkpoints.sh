#!/bin/bash
#SBATCH --job-name=ccf-separate-ckpt-ppl
#SBATCH --array=0-5%2
#SBATCH --time=02:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS

# #1 separate rank8: three checkpoints for each of the two dynamic-factor arms.
# Require afterok on GPU speed verification when using the optimized snapshot.
# Existing 7k samples are reused; original training outputs are read-only.
set -euo pipefail
# Activate the Python environment before submitting; supply site options to sbatch.
CCF_CODE_ROOT="${CCF_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${CCF_CACHE_ROOT:?Set CCF_CACHE_ROOT to your checkpoints/cache/runs directory}"
export CCF_CODE_ROOT CCF_CACHE_ROOT
cd "${CCF_CODE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ ! "$IDX" =~ ^[0-5]$ ]]; then
  echo "Expected task index 0 through 5"
  exit 2
fi
STEPS=(2000 4000 6000)
ARMS=(fixed_dynamic dynamic_dynamic)
TOPOLOGIES=(fixed dynamic)
WEIGHTS=(0.0 0.1)
SOURCE_RUNS=(ccf_separate_r8_fixed_dynamic_s001_step7000_run-separate-r8-fd ccf_separate_r8_dynamic_dynamic_s001_step7000_run-separate-r8-dd)
ARM_INDEX=$((IDX / 3))
STEP="${STEPS[$((IDX % 3))]}"
ARM="${ARMS[$ARM_INDEX]}"
CACHE_ROOT=${CCF_CACHE_ROOT}
CHECKPOINT="$CACHE_ROOT/runs/${SOURCE_RUNS[$ARM_INDEX]}/training/checkpoints/0-$STEP.ckpt"
BACKBONE="$CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt"
HF_CACHE="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false
RUN_ROOT="$CACHE_ROOT/runs/ccf_separate_checkpoints_job${SLURM_ARRAY_JOB_ID:?Submit as Slurm array}/$ARM/step$STEP"
test -f "$CHECKPOINT"
test -f "$BACKBONE"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/.run.lock"
flock -n 9 || { echo "Another process owns $RUN_ROOT"; exit 2; }
if [[ -L "$RUN_ROOT/generation" && -s "$RUN_ROOT/generation/samples.jsonl" && -s "$RUN_ROOT/generation/summary.json" ]]; then
  echo "$ARM step $STEP already complete"
  exit 0
fi
ATTEMPT_DIR=$(mktemp -d "$RUN_ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
ADAPTER="$ATTEMPT_DIR/$ARM.safetensors"
MANIFEST="$ATTEMPT_DIR/$ARM.manifest.json"
GEN_DIR="$ATTEMPT_DIR/generation"
git rev-parse HEAD > "$ATTEMPT_DIR/git-base-commit.txt"
git status --short > "$ATTEMPT_DIR/git-status.txt"
printf '%s\n' "${CCF_LOCAL_FIX_COMMIT:?Record local speed fix commit}" > "$ATTEMPT_DIR/local-fix-commit.txt"
printf '%s\n' "${CCF_LOCAL_EVAL_COMMIT:?Record local evaluation script commit}" > "$ATTEMPT_DIR/local-eval-commit.txt"
sha256sum diffusion.py models/structured_decoder.py structured_objective.py \
  structured_utils.py scripts/export_structured_adapter.py \
  scripts/run_generation_pilot.py scripts/evaluate_ccf_separate_checkpoints.sh \
  evaluation/generation_harness.py evaluation/generation_metrics.py \
  > "$ATTEMPT_DIR/code-sha256.txt"
CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
BACKBONE_SHA=$(sha256sum "$BACKBONE" | awk '{print $1}')
printf '%s  %s\n%s  %s\n' "$CHECKPOINT_SHA" "$CHECKPOINT" \
  "$BACKBONE_SHA" "$BACKBONE" > "$ATTEMPT_DIR/input-sha256.txt"
echo "Arm: $ARM; training step: $STEP; output: $RUN_ROOT"

srun --ntasks=1 python -u scripts/export_structured_adapter.py \
  --checkpoint "$CHECKPOINT" --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --expected-global-step "$STEP" --output "$ADAPTER" --manifest "$MANIFEST" \
  --control-identity "$ARM" --topology-mode "${TOPOLOGIES[$ARM_INDEX]}" \
  --factor-mode dynamic --candidate-k 128 --independent-mode false \
  --topology-weight "${WEIGHTS[$ARM_INDEX]}" > "$ATTEMPT_DIR/export-report.json"
ADAPTER_SHA=$(sha256sum "$ADAPTER" | awk '{print $1}')
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')

srun --ntasks=1 python -u scripts/run_generation_pilot.py \
  --backbone-checkpoint "$BACKBONE" --backbone-sha256 "$BACKBONE_SHA" \
  --adapter "$ADAPTER" --adapter-sha256 "$ADAPTER_SHA" \
  --adapter-manifest "$MANIFEST" --adapter-manifest-sha256 "$MANIFEST_SHA" \
  --output-dir "$GEN_DIR" --num-samples 1 --sequence-length 1024 \
  --batch-size 1 --base-seed 91001 --modes structured_joint --nfe-budgets 1001 \
  --device cuda --model-config contextual-forest-small \
  --data-config train_openwebtext_pinned --allow-dirty \
  --reference-lm gpt2-large \
  --reference-lm-revision 32b71b12589c2f8d625668d2335a01cac3249519 \
  --reference-lm-device cuda --reference-lm-batch-size 1 \
  --reference-lm-max-length 1024 --reference-lm-dtype float32 \
  --override "data.cache_dir=$HF_CACHE" \
  --override model.structured_decoder.top_k=128 \
  --override model.structured_decoder.rank=8 \
  --override ++model.structured_decoder.factor_embedding_mode=separate \
  --override ++model.structured_decoder.factor_conditioner_hidden_dim=0 \
  --override "model.structured_decoder.topology_mode=${TOPOLOGIES[$ARM_INDEX]}" \
  --override model.structured_decoder.factor_mode=dynamic \
  --override model.structured_decoder.independent_mode=false \
  --override "model.structured_decoder.training.topology_weight=${WEIGHTS[$ARM_INDEX]}" \
  --override "checkpointing.save_dir=$RUN_ROOT"
ln -s "$GEN_DIR" "$RUN_ROOT/generation"
echo "Completed $ARM at $STEP: $RUN_ROOT/generation/samples.jsonl"
