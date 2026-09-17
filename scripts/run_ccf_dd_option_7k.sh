#!/bin/bash
# Internal runner shared by the two separate Slurm entrypoints.
# Only this invocation's variant is changed; all other baseline settings match.
set -euo pipefail
# Activate the Python environment before submitting; supply site options to sbatch.
CCF_CODE_ROOT="${CCF_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${CCF_CACHE_ROOT:?Set CCF_CACHE_ROOT to your checkpoints/cache/runs directory}"
export CCF_CODE_ROOT CCF_CACHE_ROOT
VARIANT="${1:?Expected film_mlp128 or topk256}"
case "$VARIANT" in
  film_mlp128) TOP_K=128; FILM_WIDTH=128 ;;
  topk256) TOP_K=256; FILM_WIDTH=0 ;;
  *) echo "Unknown experiment: $VARIANT"; exit 2 ;;
esac

cd "${CCF_CODE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
CACHE_ROOT=${CCF_CACHE_ROOT}
BACKBONE="$CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt"
HF_CACHE="$CACHE_ROOT/huggingface"
TRAIN_SEED=1
ARM=dynamic_dynamic
TOPOLOGY=dynamic
FACTOR=dynamic
TOPOLOGY_WEIGHT=0.1
JOB_TAG="${RUN_TAG:-job${SLURM_JOB_ID:?Run through Slurm or supply RUN_TAG}}"
RUN_ROOT="$CACHE_ROOT/runs/ccf_dd_${VARIANT}_s001_step7000_${JOB_TAG}"
RUN_DIR="$RUN_ROOT/training"
ADAPTER="$RUN_ROOT/dynamic_dynamic.safetensors"
MANIFEST="$RUN_ROOT/dynamic_dynamic.manifest.json"
GEN_DIR="$RUN_ROOT/generation"
export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

test -f "$BACKBONE"
mkdir "$RUN_ROOT"
mkdir "$RUN_DIR"
git rev-parse HEAD > "$RUN_ROOT/git-commit.txt"
git status --short > "$RUN_ROOT/git-status.txt"
sha256sum diffusion.py models/dit.py models/structured_decoder.py \
  structured_objective.py structured_utils.py structured_training.py \
  scripts/run_ccf_dd_option_7k.sh scripts/export_structured_adapter.py \
  scripts/run_generation_pilot.py evaluation/generation_harness.py \
  evaluation/generation_metrics.py > "$RUN_ROOT/code-sha256.txt"
BACKBONE_SHA256=$(sha256sum "$BACKBONE" | awk '{print $1}')
printf '%s  %s\n' "$BACKBONE_SHA256" "$BACKBONE" > "$RUN_ROOT/backbone-sha256.txt"
echo "Variant=$VARIANT; top-K=$TOP_K; FiLM hidden width=$FILM_WIDTH; shared rank=16"
echo "Training fresh to 7000; constant LR=3e-4 after original 50-step warmup"
echo "Output: $RUN_ROOT"

# Exercise this allocated GPU/environment before committing to full training.
python -u - "$FILM_WIDTH" "$TOP_K" <<'PY'
import importlib.metadata
import sys
import torch
from models.structured_decoder import ContextualCouplingForestHead
from structured_training import structured_denoising_loss
from structured_objective import sample_structured_tokens
width, top_k = map(int, sys.argv[1:])
print("Python", sys.executable, "Torch", torch.__version__, "CUDA", torch.version.cuda, flush=True)
print("Allocated GPU", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)
for package in ("lightning", "transformers", "safetensors"):
    print(package, importlib.metadata.version(package), flush=True)
head = ContextualCouplingForestHead(
    hidden_size=8, vocab_size=300, top_k=top_k, rank=4, time_embed_dim=8,
    topology_dim=8, num_anchor_slots=2, contextual_neighbors=1,
    factor_conditioner_hidden_dim=width).cuda()
hidden = torch.randn(1, 4, 8, device="cuda")
logits = torch.randn(1, 4, 300, device="cuda")
active = torch.ones(1, 4, dtype=torch.bool, device="cuda")
output = head(hidden, logits, torch.tensor([0.5], device="cuda"), active)
loss = structured_denoising_loss(output, logits, output.candidate_ids[..., 0], active).loss
loss.backward()
with torch.no_grad():
    sample_structured_tokens(output, logits, active, num_samples=1)
torch.cuda.synchronize()
print("GPU head forward/backward/sampling preflight OK", flush=True)
PY

srun --ntasks=1 python -u main.py \
  mode=train \
  data=train_openwebtext_pinned \
  "data.cache_dir=$HF_CACHE" \
  seed="$TRAIN_SEED" \
  backbone=dit \
  parameterization=subs \
  model=contextual-forest-small \
  model.length=1024 \
  "model.structured_decoder.top_k=$TOP_K" \
  model.structured_decoder.rank=16 \
  ++model.structured_decoder.factor_embedding_mode=shared \
  "++model.structured_decoder.factor_conditioner_hidden_dim=$FILM_WIDTH" \
  "model.structured_decoder.topology_mode=$TOPOLOGY" \
  "model.structured_decoder.factor_mode=$FACTOR" \
  model.structured_decoder.independent_mode=false \
  model.structured_decoder.training.backbone_mode=frozen \
  model.structured_decoder.training.require_pretrained_backbone=true \
  model.structured_decoder.training.strict_backbone_checkpoint=true \
  "model.structured_decoder.training.backbone_checkpoint=$BACKBONE" \
  model.structured_decoder.training.use_ema_backbone=false \
  model.structured_decoder.training.deterministic_backbone=true \
  model.structured_decoder.training.backbone_lr_multiplier=0.0 \
  model.structured_decoder.training.head_lr=0.0003 \
  model.structured_decoder.training.structured_nll_weight=1.0 \
  model.structured_decoder.training.factorized_aux_weight=0.0 \
  "model.structured_decoder.training.topology_weight=$TOPOLOGY_WEIGHT" \
  training.antithetic_sampling=true \
  training.importance_sampling=false \
  training.sampling_eps=0.001 \
  training.change_of_variables=false \
  training.ema=0 \
  optim.lr=0.0003 \
  optim.weight_decay=0 \
  lr_scheduler.num_warmup_steps=50 \
  trainer.max_steps=7000 \
  trainer.val_check_interval=500 \
  trainer.limit_val_batches=32 \
  trainer.num_sanity_val_steps=0 \
  trainer.devices=1 \
  trainer.precision=bf16 \
  loader.global_batch_size=4 \
  loader.eval_global_batch_size=4 \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  loader.num_workers=0 \
  strategy.find_unused_parameters=true \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=500 \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=$RUN_DIR" \
  "hydra.run.dir=$RUN_DIR/hydra" \
  wandb=null

CHECKPOINT="$RUN_DIR/checkpoints/last.ckpt"
test -f "$CHECKPOINT"
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
python -u scripts/export_structured_adapter.py \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --expected-global-step 7000 \
  --output "$ADAPTER" \
  --manifest "$MANIFEST" \
  --control-identity dynamic_dynamic \
  --topology-mode dynamic \
  --factor-mode dynamic \
  --candidate-k "$TOP_K" \
  --independent-mode false \
  --topology-weight 0.1 \
  > "$RUN_ROOT/export-report.json"
ADAPTER_SHA256=$(sha256sum "$ADAPTER" | awk '{print $1}')
MANIFEST_SHA256=$(sha256sum "$MANIFEST" | awk '{print $1}')
printf '%s  %s\n' "$ADAPTER_SHA256" "$ADAPTER" > "$ADAPTER.sha256"
printf '%s  %s\n' "$MANIFEST_SHA256" "$MANIFEST" > "$MANIFEST.sha256"

# Exactly one 1024-token sample, seed 91001, 1000 transitions + cleanup.
srun --ntasks=1 python -u scripts/run_generation_pilot.py \
  --backbone-checkpoint "$BACKBONE" \
  --backbone-sha256 "$BACKBONE_SHA256" \
  --adapter "$ADAPTER" \
  --adapter-sha256 "$ADAPTER_SHA256" \
  --adapter-manifest "$MANIFEST" \
  --adapter-manifest-sha256 "$MANIFEST_SHA256" \
  --output-dir "$GEN_DIR" \
  --num-samples 1 --sequence-length 1024 --batch-size 1 --base-seed 91001 \
  --modes structured_joint --nfe-budgets 1001 --device cuda \
  --model-config contextual-forest-small \
  --data-config train_openwebtext_pinned --allow-dirty \
  --reference-lm gpt2-large \
  --reference-lm-revision 32b71b12589c2f8d625668d2335a01cac3249519 \
  --reference-lm-device cuda --reference-lm-batch-size 1 \
  --reference-lm-max-length 1024 --reference-lm-dtype float32 \
  --override "data.cache_dir=$HF_CACHE" \
  --override "model.structured_decoder.top_k=$TOP_K" \
  --override model.structured_decoder.rank=16 \
  --override ++model.structured_decoder.factor_embedding_mode=shared \
  --override "++model.structured_decoder.factor_conditioner_hidden_dim=$FILM_WIDTH" \
  --override model.structured_decoder.topology_mode=dynamic \
  --override model.structured_decoder.factor_mode=dynamic \
  --override model.structured_decoder.independent_mode=false \
  --override model.structured_decoder.training.topology_weight=0.1 \
  --override "checkpointing.save_dir=$RUN_ROOT"

echo "Completed $VARIANT: $GEN_DIR/samples.jsonl (includes reference_lm GPT-2 scores)"
