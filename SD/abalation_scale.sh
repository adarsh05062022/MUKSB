#!/usr/bin/env bash
# ============================================================================
# abalation_scale.sh — Step-size (scale) ablation on ImageNette-10  (SD v1.4)
#
# Validates the step-size choice of MUKSB by holding the KS bisector direction
# (g_star) fixed and varying ONLY the effective_scale multiplier. Mirrors the
# CIFAR-10 ablation (Classification/run_pipeline_scale_ablation.sh, Table 7),
# but for the Stable-Diffusion ImageNette class-erasure setting.
#
# Variants (all share the identical KS direction):
#   harmonic    — 2||gr||||gf||/(||gr||+||gf||)   == MUKSB (full); ALREADY RUN
#   arithmetic  — (||gr||+||gf||)/2               (Arithmetic Mean)
#   min         — min(||gr||, ||gf||)             (Min Norm)
#   fixed       — 1.0                             (Fixed alpha=1, direction-only)
#
# By default ONLY the 3 missing variants are trained (arithmetic, min, fixed),
# since the harmonic-mean MUKSB results already exist. The harmonic arm is
# trained too only if INCLUDE_HARMONIC=1.
#
# Forgetting mode: class-wise — one ImageNette class per run, 10 classes total.
# Classes run sequentially; the variants for a class launch in parallel, one
# GPU each.
#
# Usage:
#   bash abalation_scale.sh
#
# Common overrides:
#   GPUS="0 1 2"            bash abalation_scale.sh   # one GPU per variant
#   CLASSES="0 1 2"        bash abalation_scale.sh   # subset of classes
#   VARIANTS="arithmetic min" bash abalation_scale.sh
#   INCLUDE_HARMONIC=1     bash abalation_scale.sh   # also re-run harmonic
#   SMOKE=1 CLASSES="0"    bash abalation_scale.sh   # tiny end-to-end test
# ============================================================================
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PY=${PY:-/storage/s25017/miniconda3/envs/munba3/bin/python}

# ── Variants to run (default: the 3 missing arms) ────────────────────────────
if [ -n "${VARIANTS:-}" ]; then
    read -ra VARIANT_LIST <<< "${VARIANTS}"
else
    VARIANT_LIST=(arithmetic min fixed)
    if [ "${INCLUDE_HARMONIC:-0}" = "1" ]; then
        VARIANT_LIST=(harmonic "${VARIANT_LIST[@]}")
    fi
fi

# ── Training hyperparameters (match your existing MUKSB harmonic run!) ───────
SEED=${SEED:-42}
TRAIN_METHOD=${TRAIN_METHOD:-full}
CKPT_PATH=${CKPT_PATH:-models/ldm/sd-v1-4-full-ema.ckpt}
CONFIG_PATH=${CONFIG_PATH:-configs/stable-diffusion/v1-inference_nash.yaml}
DIFFUSERS_CONFIG=${DIFFUSERS_CONFIG:-diffusers_unet_config.json}
MASK_VARIANT=${MASK_VARIANT:-None}
MASK_DENSITY=${MASK_DENSITY:-0.5}
BETA=${BETA:-1.0}
ALPHA=${ALPHA:-1e-4}
DDIM_STEPS=${DDIM_STEPS:-50}

if [ "${SMOKE:-0}" = "1" ]; then
    EPOCHS=${EPOCHS:-1}
    BATCH=${BATCH:-2}
    IMG=${IMG:-256}
    LR=${LR:-5e-6}
    TAG=smoke
else
    EPOCHS=${EPOCHS:-5}
    BATCH=${BATCH:-8}
    IMG=${IMG:-256}
    LR=${LR:-5e-6}
    TAG=full
fi

# ── Classes to forget (default: all 10 ImageNette classes) ───────────────────
NUM_CLASSES=10
if [ -n "${CLASSES:-}" ]; then
    read -ra CLASS_LIST <<< "${CLASSES}"
else
    CLASS_LIST=($(seq 0 $((NUM_CLASSES - 1))))
fi

# ── GPU assignment (round-robin one GPU per variant) ─────────────────────────
if [ -n "${GPUS:-}" ]; then
    read -ra GPU_LIST <<< "${GPUS}"
else
    GPU_LIST=(0 1 2 3)
fi
NUM_GPUS=${#GPU_LIST[@]}

# ── Logs ─────────────────────────────────────────────────────────────────────
LOG_ROOT="${SCRIPT_DIR}/logs/abalation_scale"
mkdir -p "${LOG_ROOT}"

# ── Validate checkpoint ───────────────────────────────────────────────────────
if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT_PATH}"
    exit 1
fi
if [ ! -x "${PY}" ]; then
    echo "ERROR: python interpreter not found/executable: ${PY}"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Per-(variant, class) worker
# ─────────────────────────────────────────────────────────────────────────────
launch() {
    local VARIANT=$1
    local CLASS=$2
    local GPU=$3

    local LOG_FILE="${LOG_ROOT}/${TAG}_cls${CLASS}_${VARIANT}.log"
    echo "[scale=${VARIANT} | class=${CLASS} | gpu=${GPU}] → ${LOG_FILE}"

    {
        echo "====== scale=${VARIANT} | class=${CLASS} | gpu=${GPU} | seed=${SEED} ======"
        local CMD="${PY} MUKSB_cls_scale.py \
            --scale_variant ${VARIANT} \
            --class_to_forget ${CLASS} \
            --train_method ${TRAIN_METHOD} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH} \
            --lr ${LR} \
            --image_size ${IMG} \
            --ckpt_path ${CKPT_PATH} \
            --config_path ${CONFIG_PATH} \
            --diffusers_config_path ${DIFFUSERS_CONFIG} \
            --mask_variant ${MASK_VARIANT} \
            --mask_density ${MASK_DENSITY} \
            --beta ${BETA} \
            --alpha ${ALPHA} \
            --ddim_steps ${DDIM_STEPS} \
            --device ${GPU}"

        echo "+ ${CMD}"
        eval ${CMD}
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "ERROR: scale=${VARIANT} class=${CLASS} failed (rc=${rc})"
            exit $rc
        fi
        echo "====== DONE: scale=${VARIANT} | class=${CLASS} ======"
    } 2>&1 | tee "${LOG_FILE}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Scale Ablation — ImageNette-10 class-wise erasure (SD v1.4)"
echo "  Variants : ${VARIANT_LIST[*]}"
echo "  Classes  : ${CLASS_LIST[*]}"
echo "  GPUs     : ${GPU_LIST[*]}"
echo "  Mode     : ${TAG}  (epochs=${EPOCHS} batch=${BATCH} lr=${LR} img=${IMG})"
echo "  Ckpt     : ${CKPT_PATH}"
echo "  Models   : ${SCRIPT_DIR}/models/compvis-cls_<C>-MUKSB_scale_<variant>-..."
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Classes sequentially — variants for a class run in parallel
# ─────────────────────────────────────────────────────────────────────────────
OVERALL_FAILED=0
TOTAL=${#CLASS_LIST[@]}
CLASS_IDX=0

for CLASS in "${CLASS_LIST[@]}"; do
    CLASS_IDX=$((CLASS_IDX + 1))
    echo ""
    echo "============================================================"
    echo "  Class ${CLASS}  (${CLASS_IDX}/${TOTAL}) — launching ${#VARIANT_LIST[@]} variant(s)"
    echo "============================================================"

    PIDS=()
    LABELS=()
    VI=0
    for VARIANT in "${VARIANT_LIST[@]}"; do
        GPU=${GPU_LIST[$((VI % NUM_GPUS))]}
        launch "${VARIANT}" "${CLASS}" "${GPU}" &
        PIDS+=($!)
        LABELS+=("${VARIANT}/class${CLASS}")
        VI=$((VI + 1))
    done

    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "FAILED: ${LABELS[$i]} (rc=${rc})"
            OVERALL_FAILED=$((OVERALL_FAILED + 1))
        else
            echo "OK: ${LABELS[$i]}"
        fi
    done
    echo "<<< Class ${CLASS} done"
done

echo ""
echo "============================================================"
if [ "${OVERALL_FAILED}" -eq 0 ]; then
    echo "All scale-ablation variants completed successfully."
else
    echo "${OVERALL_FAILED} run(s) failed. Check logs under: ${LOG_ROOT}"
fi
echo "Trained models: ${SCRIPT_DIR}/models/"
echo "Next: generate + evaluate each unlearned model (UA / FID / CLIP) the"
echo "same way you evaluated the harmonic MUKSB models, then assemble Table 7."
echo "============================================================"
exit ${OVERALL_FAILED}
