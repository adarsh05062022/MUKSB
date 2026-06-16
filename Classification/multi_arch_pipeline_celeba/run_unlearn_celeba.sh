#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MUKSB Classification — multi-architecture unlearning pipeline
# Generic CelebA worker: supports forget_fraction-based forgetting.
#
# This script is sourced from per-arch wrappers
# (run_resnet34_celeba.sh, ...).
#
# Required env vars (set by the caller):
#   ARCH            — e.g. resnet34
#   MODEL_PATH      — pretrained checkpoint
#   DATASET         — celeba
#   NUM_CLASSES     — 307
#   DATA_DIR        — path to dataset
#   RESULTS_ROOT    — output root
#   FORGET_FRACTION — fraction of identity classes to forget (e.g. 0.1 = 10%)
#   SEEDS / GPUS / EPOCHS — bash arrays, same length
#   METHODS         — bash array, subset of {MUKSB, retrain, FT}
#
# Unlearning hyperparams (MUKSB / FT):
#   UNLEARN_LR, GAMMA, ALPHA, BATCH_SIZE, DECREASING_LR
#
# Retrain hyperparams (should match original base-model training):
#   RETRAIN_LR, RETRAIN_EPOCHS, RETRAIN_BATCH_SIZE, RETRAIN_DECREASING_LR
#
# Usage (from a wrapper):
#   source run_unlearn_celeba.sh
# ─────────────────────────────────────────────────────────────────

set -uo pipefail

# ── Dataset defaults ───────────────────────────────────────────────
: "${DATASET:=celeba}"
: "${NUM_CLASSES:=307}"
: "${DATA_DIR:=/storage/s25017/Datasets/CELEB_HQ_307}"
: "${RESULTS_ROOT:=results_multi_arch}"

: "${CLASS_TO_REPLACE:=-1}"
: "${NUM_INDEXES:=4500}"   # required by arg_parser but ignored by CelebA loader
: "${FORGET_FRACTION:=0.1}"

# ── Unlearning hyperparams (MUKSB / FT) ───────────────────────────
# Per paper: SGD, lr=0.02, 10 epochs, bs=32 (MUKSB / FT)
: "${UNLEARN_EPOCHS:=10}"
: "${UNLEARN_LR:=0.02}"
: "${GAMMA:=0.5}"
: "${WITH_L1:=false}"
: "${ALPHA:=0.2}"
: "${BATCH_SIZE:=32}"
: "${PRINT_FREQ:=50}"
: "${DECREASING_LR:=91,136}"
: "${EXTRA:=lr_0_02}"

# ── Retrain hyperparams (match base-model training) ────────────────
# Per paper: SGD, lr=1e-3, 10 epochs, bs=8, cosine schedule.
# --imagenet_arch in impl.py switches the scheduler to cosine for retrain.
: "${RETRAIN_LR:=1e-3}"
: "${RETRAIN_EPOCHS:=10}"
: "${RETRAIN_BATCH_SIZE:=8}"
: "${RETRAIN_IMAGENET_ARCH:=true}"

FORGET_PCT=$(echo "${FORGET_FRACTION} * 100" | bc | cut -d. -f1)
FORGET_TAG="random_${FORGET_PCT}pct_bs${BATCH_SIZE}_${EXTRA}"

# ── Validate ──────────────────────────────────────────────────────
if [ -z "${ARCH:-}" ] || [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: ARCH and MODEL_PATH must be set by the caller."
    exit 1
fi

if [ ${#SEEDS[@]} -ne ${#GPUS[@]} ] || [ ${#SEEDS[@]} -ne ${#EPOCHS[@]} ]; then
    echo "ERROR: SEEDS, GPUS, and EPOCHS arrays must all be the same length."
    exit 1
fi

if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS=("MUKSB" "retrain")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CLS_DIR}"

# ─────────────────────────────────────────────────────────────────
# Per-(method, seed) worker
# ─────────────────────────────────────────────────────────────────
run_one() {
    local METHOD=$1
    local SEED=$2
    local GPU=$3
    local EPOCHS_FOR_SEED=$4
    local LR_FOR_METHOD=$5
    local BS_FOR_METHOD=$6
    local USE_IMAGENET_ARCH=$7   # "true" → pass --imagenet_arch (cosine LR for retrain)

    local SAVE_DIR="${RESULTS_ROOT}/${DATASET}/${ARCH}/${METHOD}/${FORGET_TAG}/seed${SEED}"
    local LOG_FILE="${SAVE_DIR}/run.log"

    mkdir -p "${SAVE_DIR}"

    echo "[arch=${ARCH} method=${METHOD} seed=${SEED} gpu=${GPU} forget=${FORGET_PCT}% lr=${LR_FOR_METHOD} epochs=${EPOCHS_FOR_SEED}] log: ${LOG_FILE}"

    {
        echo "====== ${METHOD} on ${ARCH} (seed=${SEED}, gpu=${GPU}, forget=${FORGET_PCT}%, lr=${LR_FOR_METHOD}, epochs=${EPOCHS_FOR_SEED}) ======"
        CMD="python main_random.py \
            --unlearn ${METHOD} \
            --unlearn_epochs ${EPOCHS_FOR_SEED} \
            --unlearn_lr ${LR_FOR_METHOD} \
            --mask ${MODEL_PATH} \
            --save_dir ${SAVE_DIR} \
            --dataset ${DATASET} \
            --arch ${ARCH} \
            --num_classes ${NUM_CLASSES} \
            --gpu ${GPU} \
            --class_to_replace ${CLASS_TO_REPLACE} \
            --num_indexes_to_replace ${NUM_INDEXES} \
            --forget_fraction ${FORGET_FRACTION} \
            --gamma ${GAMMA} \
            --alpha ${ALPHA} \
            --batch_size ${BS_FOR_METHOD} \
            --seed ${SEED} \
            --print_freq ${PRINT_FREQ} \
            --decreasing_lr ${DECREASING_LR} \
            --data ${DATA_DIR}"

        if [ "${WITH_L1}" = "true" ]; then
            CMD="${CMD} --with_l1"
        fi
        if [ "${USE_IMAGENET_ARCH}" = "true" ]; then
            CMD="${CMD} --imagenet_arch"
        fi

        eval ${CMD}
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "ERROR: ${METHOD} failed for arch=${ARCH} seed=${SEED} forget=${FORGET_PCT}% (rc=${rc})"
            exit $rc
        fi
        echo "====== DONE (${METHOD} | arch=${ARCH} | seed=${SEED} | forget=${FORGET_PCT}%) ======"
        echo "  Output: ${SAVE_DIR}/"
    } 2>&1 | tee "${LOG_FILE}"
}

# ─────────────────────────────────────────────────────────────────
# Launch: methods sequentially, seeds in parallel within a method
# ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "CELEBA UNLEARNING — forget ${FORGET_PCT}%"
echo "  Arch     : ${ARCH}"
echo "  Dataset  : ${DATASET} | Classes: ${NUM_CLASSES}"
echo "  Methods  : ${METHODS[*]}"
echo "  Forget   : ${FORGET_PCT}% of identity classes (forget_fraction=${FORGET_FRACTION})"
echo "  MUKSB/FT : lr=${UNLEARN_LR}  epochs=${UNLEARN_EPOCHS}  bs=${BATCH_SIZE}"
echo "  retrain  : lr=${RETRAIN_LR}  epochs=${RETRAIN_EPOCHS}  bs=${RETRAIN_BATCH_SIZE}  cosine=${RETRAIN_IMAGENET_ARCH:-true}"
echo "  Seeds    : ${SEEDS[*]}"
echo "  GPUs     : ${GPUS[*]}"
echo "  Out root : ${RESULTS_ROOT}/${DATASET}/${ARCH}/"
echo "============================================================"

OVERALL_FAILED=0
for METHOD in "${METHODS[@]}"; do
    echo ""
    echo ">>> Launching ${METHOD} (${ARCH}, forget=${FORGET_PCT}%) across ${#SEEDS[@]} seed(s)"

    # Pick hyperparams based on method
    if [ "${METHOD}" = "retrain" ]; then
        M_LR="${RETRAIN_LR}"
        M_EPOCHS="${RETRAIN_EPOCHS}"
        M_BS="${RETRAIN_BATCH_SIZE}"
        M_IMAGENET_ARCH="${RETRAIN_IMAGENET_ARCH:-true}"
    else
        M_LR="${UNLEARN_LR}"
        M_EPOCHS=""   # use per-seed EPOCHS array
        M_BS="${BATCH_SIZE}"
        M_IMAGENET_ARCH="false"
    fi

    PIDS=()
    for i in "${!SEEDS[@]}"; do
        EPOCHS_I=${M_EPOCHS:-${EPOCHS[$i]}}
        run_one "${METHOD}" "${SEEDS[$i]}" "${GPUS[$i]}" \
                "${EPOCHS_I}" "${M_LR}" "${M_BS}" "${M_IMAGENET_ARCH}" &
        PIDS+=($!)
    done

    FAILED=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}"
        if [ $? -ne 0 ]; then
            echo "FAILED: arch=${ARCH} method=${METHOD} seed=${SEEDS[$i]} gpu=${GPUS[$i]} forget=${FORGET_PCT}%"
            FAILED=$((FAILED + 1))
        fi
    done
    OVERALL_FAILED=$((OVERALL_FAILED + FAILED))
    echo "<<< ${METHOD} done (forget=${FORGET_PCT}%) — ${FAILED} failure(s)"
done

echo ""
echo "============================================================"
if [ "${OVERALL_FAILED}" -eq 0 ]; then
    echo "All (method × seed) runs completed for ${ARCH} forget=${FORGET_PCT}%."
else
    echo "${OVERALL_FAILED} run(s) failed for ${ARCH} forget=${FORGET_PCT}%. See logs."
fi
echo "Results : ${RESULTS_ROOT}/${DATASET}/${ARCH}/"
echo "============================================================"

exit ${OVERALL_FAILED}
