#!/usr/bin/env bash
# ============================================================================
# run_celebrity.sh — MUKSB celebrity (identity) forgetting pipeline
#
# Class-wise identity unlearning, an exact parallel to the Imagenette-10 setup:
# 10 celebrity classes, forget ONE, retain the other 9, pseudo-label = the next
# celebrity's caption (descriptions[(c+1)%10], like MUKSB_cls.py).
#
# Training data = the REAL Celebrity Faces Dataset (the 10 chosen identities,
# ~100 real jpg faces each). Evaluation generates images from the model and
# measures identity removal / retention; FID compares retain generations against
# the REAL retain faces.
#
# Stages:
#   1. train MUKSB on the real dataset to forget celeb FORGET_IDX (0..9)
#   2. generate SD v1.4 baseline eval images   (UA-before reference)
#   3. eval unlearned model: UA + RA + CLIP + FID (retain gens vs REAL faces)
#
# Modes:
#   SMOKE=1   -> tiny end-to-end test (few eval imgs, 1 epoch, 256px)
#   SMOKE=0   -> full run
#
# Usage:
#   SMOKE=1 GPUS="0"       FORGET_IDX=1 bash run_celebrity.sh   # forget Brad Pitt
#   SMOKE=0 GPUS="0 1 2 3" FORGET_IDX=1 bash run_celebrity.sh
# ============================================================================
set -euo pipefail

PY=${PY:-/storage/s25017/miniconda3/envs/munba3/bin/python}
SD_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SD_DIR"

# ── config (override via env) ────────────────────────────────────────────────
SMOKE=${SMOKE:-1}
GPUS=${GPUS:-0}
FORGET_IDX=${FORGET_IDX:-1}
PROMPTS=${PROMPTS:-prompts/celebrity.csv}
# Real dataset = training data AND FID reference (retain quality vs real faces).
REAL_DS=${REAL_DS:-/storage/s25017/Datasets/Celebrity_Faces_Dataset}
GEN_ROOT=${GEN_ROOT:-Evaluation/celebrity/generated}
LR=${LR:-1e-5}
BETA=${BETA:-100.0}
BATCH=${BATCH:-4}
GUID=${GUID:-7.5}
ANCHOR_MODE=${ANCHOR_MODE:-next}            # pseudo-label = next celebrity
ANCHOR_PROMPT=${ANCHOR_PROMPT:-"a photo of a person"}   # only used if ANCHOR_MODE=fixed
# SalUn saliency mask (|∇L_f|) localizes the update to forget-salient weights,
# curbing collateral leakage onto retained identities. Lower density = tighter.
TRAIN_METHOD=${TRAIN_METHOD:-xattn}          # full | xattn | noxattn | selfattn | ...
MASK_VARIANT=${MASK_VARIANT:-salun}         # salun (recommended) | None | forget_fisher | dual_fisher | random
MASK_DENSITY=${MASK_DENSITY:-0.5}           # fraction of params updated (try 0.1–0.3 for tighter localization)

if [[ "$SMOKE" == "1" ]]; then
    EPOCHS=${EPOCHS:-1}
    N_EVAL=${N_EVAL:-8}
    IMG=${IMG:-256}
    STEPS=${STEPS:-25}
    TAG=smoke
else
    EPOCHS=${EPOCHS:-5}
    N_EVAL=${N_EVAL:-50}
    IMG=${IMG:-512}
    STEPS=${STEPS:-50}
    TAG=full
fi

GPU_ARR=($GPUS)
GPU0=${GPU_ARR[0]}

LOG_DIR="$GEN_ROOT/logs"; mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/pipeline_${TAG}_${RUN_ID}.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "============================================================"
echo " MUKSB celebrity forgetting | mode=$TAG | forget_idx=$FORGET_IDX"
echo " GPUs=$GPUS  epochs=$EPOCHS  lr=$LR  beta=$BETA  anchor=$ANCHOR_MODE"
echo " REAL_DS=$REAL_DS  GEN_ROOT=$GEN_ROOT  img=$IMG steps=$STEPS"
echo " LOG -> $PIPELINE_LOG   Started: $(date)"
echo "============================================================"

[[ -d "$REAL_DS" ]] || { echo "ERROR: real dataset not found: $REAL_DS"; exit 1; }

# ── Stage 1: train MUKSB on the real dataset ─────────────────────────────────
echo "--- [1] train MUKSB to forget celeb $FORGET_IDX (real faces, pseudo-label=$ANCHOR_MODE) ---"
TRAIN_LOG="$LOG_DIR/train_celeb${FORGET_IDX}_${TAG}_${RUN_ID}.log"
$PY MUKSB_celebrity.py --class_to_forget "$FORGET_IDX" \
    --epochs "$EPOCHS" --lr "$LR" --beta "$BETA" --batch_size "$BATCH" \
    --image_size "$IMG" --ddim_steps "$STEPS" \
    --train_method "$TRAIN_METHOD" \
    --mask_variant "$MASK_VARIANT" --mask_density "$MASK_DENSITY" \
    --anchor_mode "$ANCHOR_MODE" --anchor_prompt "$ANCHOR_PROMPT" \
    --device "$GPU0" 2>&1 | tee "$TRAIN_LOG"

RUN_TAG=$(grep -oP 'RUN_TAG=\K.*' "$TRAIN_LOG" | tail -1)
DIFF_TAG=${RUN_TAG/compvis/diffusers}
CKPT="models/$RUN_TAG/${DIFF_TAG}-epoch_${EPOCHS}.pt"
echo "    RUN_TAG = $RUN_TAG"
echo "    CKPT    = $CKPT"
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

# ── Stage 2: SD v1.4 baseline eval (UA-before) ───────────────────────────────
# echo "--- [2] generate SD v1.4 baseline eval images ($N_EVAL/celeb) ---"
# $PY Evaluation/celebrity/eval_celebrity.py --model_path "" \
#     --class_to_forget "$FORGET_IDX" --output_dir "$GEN_ROOT" \
#     --prompts_csv "$PROMPTS" --ref_root "$REAL_DS" --device $GPUS \
#     --n_per_class "$N_EVAL" --guidance_scale "$GUID" \
#     --image_size "$IMG" --ddim_steps "$STEPS"

# ── Stage 3: unlearned-model eval (UA + RA + CLIP + FID vs real faces) ────────
echo "--- [3] eval unlearned model (UA + RA + CLIP + FID vs real retain faces) ---"
$PY Evaluation/celebrity/eval_celebrity.py --model_path "$CKPT" \
    --class_to_forget "$FORGET_IDX" --output_dir "$GEN_ROOT" \
    --prompts_csv "$PROMPTS" --ref_root "$REAL_DS" --device $GPUS \
    --n_per_class "$N_EVAL" --guidance_scale "$GUID" \
    --image_size "$IMG" --ddim_steps "$STEPS"

echo
echo "ALL DONE ($TAG) — $(date)"
echo "Pipeline log : $PIPELINE_LOG"
echo "Summaries    :"
find "$GEN_ROOT" -name eval_summary.json 2>/dev/null | sort | while read -r f; do
    echo "  $f"
done
