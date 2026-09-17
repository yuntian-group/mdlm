#!/bin/bash
#SBATCH --job-name=mdlm-hf-1k
#SBATCH --time=04:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Generate ten adapter-free samples directly from the released Hugging Face
# MDLM checkpoint.  main.py only prints the final sample batch, so this script
# makes ten one-batch calls and preserves every sample in its own output.txt.

set -euo pipefail
# Activate the Python environment before submitting; supply site options to sbatch.
CCF_CODE_ROOT="${CCF_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${CCF_CACHE_ROOT:?Set CCF_CACHE_ROOT to your checkpoints/cache/runs directory}"
export CCF_CODE_ROOT CCF_CACHE_ROOT

cd "$CCF_CODE_ROOT"

HF_CACHE=${CCF_CACHE_ROOT}/huggingface
NUM_SAMPLES="${NUM_SAMPLES:-10}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}}"
OUT_ROOT="${CCF_CACHE_ROOT}/runs/original_mdlm_hf_l1024_t1000_${RUN_TAG}"

if [[ ! "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SAMPLES must be a positive integer; found: $NUM_SAMPLES"
  exit 2
fi
if [[ -e "$OUT_ROOT" ]]; then
  echo "Output already exists: $OUT_ROOT"
  exit 2
fi
mkdir -p "$OUT_ROOT"

export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

for ((SEED = 1; SEED <= NUM_SAMPLES; SEED++)); do
  RUN_DIR="$OUT_ROOT/seed_${SEED}"
  mkdir -p "$RUN_DIR"

  echo "=== HUGGING FACE MDLM SEED $SEED/$NUM_SAMPLES ==="
  python -u main.py \
    mode=sample_eval \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    data=openwebtext-split \
    "data.cache_dir=$HF_CACHE" \
    model.length=1024 \
    sampling.predictor=ddpm_cache \
    sampling.steps=1000 \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=1 \
    eval.compute_generative_perplexity=false \
    backbone=hf_dit \
    seed="$SEED" \
    checkpointing.resume_from_ckpt=false \
    "checkpointing.save_dir=$RUN_DIR" \
    "hydra.run.dir=$RUN_DIR/hydra" \
    2>&1 | tee "$RUN_DIR/output.txt"
done

echo "Saved all $NUM_SAMPLES Hugging Face MDLM samples under: $OUT_ROOT"
echo "Score them with:"
echo "python scripts/score_text_files.py $OUT_ROOT/seed_*/output.txt --device cuda --output $OUT_ROOT/gpt2-generation-scores.tsv"
