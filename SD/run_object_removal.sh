#!/usr/bin/env bash
# ============================================================================
# run_object_removal.sh — MUKSB object-concept removal pipeline
#   (dog / car / bicycle)
#
# Stages per concept:
#   0. generate forget + retain TRAINING images from vanilla SD v1.4
#      (skipped if images already exist unless REGEN=1)
#   1. train MUKSB to unlearn the concept
#   2. eval the unlearned model: UA + FID (vs --ref_dir) + CLIP
#      (run baseline eval separately with BASELINE=1 after method is verified)
#
# Modes:
#   SMOKE=1   -> tiny end-to-end test (8 imgs, 1 epoch)
#   SMOKE=0   -> full run
#   BASELINE=1 -> skip training, run baseline eval only (SD v1.4, no ckpt)
#   REGEN=1   -> force regenerate training data even if folder exists
#
# Usage:
#   SMOKE=1 GPUS="4" CONCEPTS="dog" bash run_object_removal.sh
#   SMOKE=0 GPUS="0 1 2 3" CONCEPTS="dog car bicycle" bash run_object_removal.sh
#   BASELINE=1 GPUS="0" CONCEPTS="dog" bash run_object_removal.sh
# ============================================================================
set -euo pipefail

PY=${PY:-/storage/s25017/miniconda3/envs/munba3/bin/python}
SD_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SD_DIR"

# ── config (override via env) ────────────────────────────────────────────────
SMOKE=${SMOKE:-1}
BASELINE=${BASELINE:-0}
REGEN=${REGEN:-0}
GPUS=${GPUS:-0}
CONCEPTS=${CONCEPTS:-"dog car bicycle"}
DATA_ROOT=${DATA_ROOT:-/storage/s25017/Datasets/object_removal}
GEN_ROOT=${GEN_ROOT:-Evaluation/objects/generated}
ANCHOR=${ANCHOR:-"a photo"}
LR=${LR:-1e-5}
BETA=${BETA:-100.0}
BATCH=${BATCH:-4}
GUID=${GUID:-7.5}
IMG=${IMG:-512}
STEPS=${STEPS:-50}
# If REF_DIR is set, FID is computed against it; otherwise FID is skipped
REF_DIR=${REF_DIR:-""}

if [[ "$SMOKE" == "1" ]]; then
    EPOCHS=${EPOCHS:-1}
    N_TRAIN=${N_TRAIN:-8}
    N_EVAL=${N_EVAL:-8}
    N_RETAIN_EVAL=${N_RETAIN_EVAL:-8}
    TAG=smoke
else
    EPOCHS=${EPOCHS:-5}
    N_TRAIN=${N_TRAIN:-0}   # 0 = full CSV
    N_EVAL=${N_EVAL:-0}
    N_RETAIN_EVAL=${N_RETAIN_EVAL:-0}
    TAG=full
fi

GPU_ARR=($GPUS)
GPU0=${GPU_ARR[0]}
PROMPTS=prompts/objects
TMP=prompts/objects/_subset; mkdir -p "$TMP"

# ── logging: save to GEN_ROOT so results stay together ──────────────────────
LOG_DIR="$GEN_ROOT/logs"
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/pipeline_${TAG}_${RUN_ID}.log"

# tee all output to log file
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "============================================================"
echo " MUKSB object removal | mode=$TAG | concepts: $CONCEPTS"
echo " GPUs=$GPUS  epochs=$EPOCHS  lr=$LR  beta=$BETA  anchor='$ANCHOR'"
echo " DATA_ROOT=$DATA_ROOT  GEN_ROOT=$GEN_ROOT"
echo " LOG -> $PIPELINE_LOG"
echo " BASELINE=$BASELINE  REGEN=$REGEN  REF_DIR=${REF_DIR:-<none>}"
echo " Started: $(date)"
echo "============================================================"

# head -N a CSV (keeping header). N<=0 -> return original path unchanged.
subset_csv () {
    local src=$1 n=$2 out=$3
    if [[ "$n" -le 0 ]]; then echo "$src"; return; fi
    head -n $((n + 1)) "$src" > "$out"
    echo "$out"
}

# check if a directory already has images in it
has_images () { find "$1" -maxdepth 1 -name "*.png" -o -name "*.jpg" 2>/dev/null | grep -q .; }

CONCEPT_LOGS=()

for C in $CONCEPTS; do
    echo
    echo "########## CONCEPT: $C  [$(date +%H:%M:%S)] ##########"

    FORGET_CSV=$(subset_csv "$PROMPTS/${C}_forget.csv"  "$N_TRAIN"       "$TMP/${C}_forget.csv")
    RETAIN_CSV=$(subset_csv "$PROMPTS/${C}_retain.csv"  "$N_TRAIN"       "$TMP/${C}_retain.csv")
    EVAL_CSV=$(  subset_csv "$PROMPTS/${C}_eval.csv"    "$N_EVAL"        "$TMP/${C}_eval.csv")
    QUAL_CSV=$(  subset_csv "$PROMPTS/${C}_retain.csv"  "$N_RETAIN_EVAL" "$TMP/${C}_qual.csv")

    FORGET_DIR=$DATA_ROOT/$C/forget
    RETAIN_DIR=$DATA_ROOT/$C/retain
    OUT_ROOT=$GEN_ROOT/$C

    # ── Stage 0: training data ───────────────────────────────────────────────
    if [[ "$BASELINE" == "0" ]]; then
        if has_images "$FORGET_DIR" && [[ "$REGEN" != "1" ]]; then
            echo "--- [0] forget images exist, skipping (set REGEN=1 to force) ---"
        else
            echo "--- [0] generate forget training images ---"
            $PY Evaluation/objects/generate_objects_multigpu.py \
                --prompts_csv "$FORGET_CSV" --output_dir "$FORGET_DIR" \
                --n_per_prompt 1 --gpu_ids $GPUS --model_path "" \
                --guidance_scale $GUID --image_size $IMG --ddim_steps $STEPS --batch_size $BATCH
        fi

        if has_images "$RETAIN_DIR" && [[ "$REGEN" != "1" ]]; then
            echo "--- [0] retain images exist, skipping (set REGEN=1 to force) ---"
        else
            echo "--- [0] generate retain training images ---"
            $PY Evaluation/objects/generate_objects_multigpu.py \
                --prompts_csv "$RETAIN_CSV" --output_dir "$RETAIN_DIR" \
                --n_per_prompt 1 --gpu_ids $GPUS --model_path "" \
                --guidance_scale $GUID --image_size $IMG --ddim_steps $STEPS --batch_size $BATCH
        fi

        # ── Stage 1: train MUKSB ──────────────────────────────────────────────
        echo "--- [1] train MUKSB ($C) ---"
        TRAIN_LOG="$LOG_DIR/train_${C}_${TAG}_${RUN_ID}.log"
        $PY MUKSB_object.py --concept "$C" \
            --forget_path "$FORGET_DIR" --remain_path "$RETAIN_DIR" \
            --anchor_prompt "$ANCHOR" --epochs $EPOCHS --lr $LR --beta $BETA \
            --batch_size $BATCH --image_size $IMG --device $GPU0 2>&1 | tee "$TRAIN_LOG"
        RUN_TAG=$(grep -oP 'RUN_TAG=\K.*' "$TRAIN_LOG" | tail -1)
        DIFF_TAG=${RUN_TAG/compvis/diffusers}
        CKPT="models/$RUN_TAG/${DIFF_TAG}-epoch_${EPOCHS}.pt"
        echo "    RUN_TAG = $RUN_TAG"
        echo "    CKPT    = $CKPT"
        [[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
        CONCEPT_LOGS+=("$C: $TRAIN_LOG")

        # ── Stage 2: eval unlearned model ─────────────────────────────────────
        echo "--- [2] eval unlearned model (UA + CLIP${REF_DIR:+ + FID}) ---"
        FID_ARGS=""
        [[ -n "$REF_DIR" && -d "$REF_DIR" ]] || FID_ARGS="--skip_fid"
        $PY Evaluation/objects/eval_objects.py --concept "$C" --model_path "$CKPT" \
            --device $GPUS --output_root "$OUT_ROOT" \
            --eval_prompts "$EVAL_CSV" --retain_prompts "$QUAL_CSV" \
            --ref_dir "${REF_DIR:-}" \
            --n_eval_per_prompt 1 --n_retain_per_prompt 1 \
            --guidance_scale $GUID --image_size $IMG --ddim_steps $STEPS $FID_ARGS

    else
        # ── BASELINE mode: eval vanilla SD v1.4 only ──────────────────────────
        echo "--- [baseline] eval vanilla SD v1.4 (UA-before + retain CLIP) ---"
        $PY Evaluation/objects/eval_objects.py --concept "$C" --model_path "" \
            --device $GPUS --output_root "$OUT_ROOT" \
            --eval_prompts "$EVAL_CSV" --retain_prompts "$QUAL_CSV" \
            --n_eval_per_prompt 1 --n_retain_per_prompt 1 \
            --guidance_scale $GUID --image_size $IMG --ddim_steps $STEPS --skip_fid
        echo "    Retain images -> $OUT_ROOT/sd14_baseline/retain  (use as REF_DIR for FID)"
    fi

    echo "########## DONE: $C  [$(date +%H:%M:%S)] ##########"
done

echo
echo "ALL CONCEPTS DONE ($TAG) — $(date)"
echo "Pipeline log: $PIPELINE_LOG"
echo
echo "Summaries:"
for C in $CONCEPTS; do
    find "$GEN_ROOT/$C" -name eval_summary.json 2>/dev/null | sort | while read -r f; do
        echo "  $f"
        python3 -c "import json,sys; d=json.load(open('$f')); \
            print(f\"    UA={d.get('ua_top1','?')}%  FID={d.get('retain_fid','?')}  CLIP={d.get('retain_clip','?')}\")" 2>/dev/null || true
    done
done

if [[ ${#CONCEPT_LOGS[@]} -gt 0 ]]; then
    echo
    echo "Training logs:"
    for l in "${CONCEPT_LOGS[@]}"; do echo "  $l"; done
fi
