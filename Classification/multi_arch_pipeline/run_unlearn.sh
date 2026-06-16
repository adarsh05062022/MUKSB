#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MUKSB Classification — multi-architecture unlearning pipeline
# Generic worker: runs BOTH `MUKSB` and `retrain` for a given arch.
# Each unlearning method runs in parallel across seeds / GPUs.
#
# This script is sourced/invoked from per-arch wrappers
# (run_resnet18_cifar10.sh, run_vgg16_cifar10.sh, ...).
#
# Required env vars (set by the caller):
#   ARCH            — e.g. resnet18 | vgg16_bn | vgg16_bn_lth
#   MODEL_PATH      — pretrained checkpoint (used by MUKSB; retrain starts fresh)
#   DATASET         — cifar10 (default)
#   NUM_CLASSES     — 10 (default)
#   DATA_DIR        — path to dataset
#   RESULTS_ROOT    — output root
#   SEEDS / GPUS / EPOCHS — bash arrays, same length
#   METHODS         — bash array, subset of {MUKSB, retrain}
# Optional hyperparams (defaults below):
#   NUM_INDEXES, UNLEARN_LR, GAMMA, ALPHA, BATCH_SIZE, ...
#
# Usage (from a wrapper):
#   source run_unlearn.sh
# ─────────────────────────────────────────────────────────────────

set -uo pipefail

# ── Defaults (only set if not provided by caller) ──────────────────
: "${DATASET:=cifar10}"
: "${NUM_CLASSES:=10}"
: "${DATA_DIR:=/storage/s25017/Datasets/CIFAR10}"
: "${RESULTS_ROOT:=results_multi_arch}"

: "${CLASS_TO_REPLACE:=-1}"
: "${NUM_INDEXES:=4500}"

: "${UNLEARN_EPOCHS:=30}"
: "${RETRAIN_EPOCHS:=${UNLEARN_EPOCHS}}"
: "${UNLEARN_LR:=0.03}"
: "${RETRAIN_LR:=${UNLEARN_LR}}"
: "${GAMMA:=0.5}"
: "${WITH_L1:=false}"
: "${ALPHA:=0.2}"

: "${BATCH_SIZE:=512}"
: "${PRINT_FREQ:=50}"
: "${DECREASING_LR:=91,136}"
: "${RETRAIN_DECREASING_LR:=${DECREASING_LR}}"

: "${EXTRA:=lr_0_03}"

FORGET_TAG="random_${NUM_INDEXES}_bs${BATCH_SIZE}_${EXTRA}"

# ── Validate required arrays ──────────────────────────────────────
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
    local EPOCHS_FOR_SEED=${4:-${UNLEARN_EPOCHS}}
    local LR_FOR_RUN=${5:-${UNLEARN_LR}}
    local DECR_LR_FOR_RUN=${6:-${DECREASING_LR}}

    local SAVE_DIR="${RESULTS_ROOT}/${DATASET}/${ARCH}/${METHOD}/${FORGET_TAG}/seed${SEED}"
    local LOG_FILE="${SAVE_DIR}/run.log"

    mkdir -p "${SAVE_DIR}"

    echo "[arch=${ARCH} method=${METHOD} seed=${SEED} gpu=${GPU} lr=${LR_FOR_RUN}] log: ${LOG_FILE}"

    {
        echo "====== ${METHOD} on ${ARCH} (seed=${SEED}, gpu=${GPU}, lr=${LR_FOR_RUN}) ======"
        CMD="python main_random.py \
            --unlearn ${METHOD} \
            --unlearn_epochs ${EPOCHS_FOR_SEED} \
            --unlearn_lr ${LR_FOR_RUN} \
            --mask ${MODEL_PATH} \
            --save_dir ${SAVE_DIR} \
            --dataset ${DATASET} \
            --arch ${ARCH} \
            --num_classes ${NUM_CLASSES} \
            --gpu ${GPU} \
            --class_to_replace ${CLASS_TO_REPLACE} \
            --num_indexes_to_replace ${NUM_INDEXES} \
            --gamma ${GAMMA} \
            --alpha ${ALPHA} \
            --batch_size ${BATCH_SIZE} \
            --seed ${SEED} \
            --print_freq ${PRINT_FREQ} \
            --decreasing_lr ${DECR_LR_FOR_RUN} \
            --data ${DATA_DIR}"

        if [ "${WITH_L1}" = "true" ]; then
            CMD="${CMD} --with_l1"
        fi

        eval ${CMD}
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "ERROR: ${METHOD} failed for arch=${ARCH} seed=${SEED} (rc=${rc})"
            exit $rc
        fi
        echo "====== DONE (${METHOD} | arch=${ARCH} | seed=${SEED}) ======"
        echo "  Output: ${SAVE_DIR}/"
    } 2>&1 | tee "${LOG_FILE}"
}

# ─────────────────────────────────────────────────────────────────
# Launch: methods sequentially, seeds in parallel within a method
# (sequential between methods so GPUs are not over-subscribed)
# ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "MULTI-ARCH UNLEARNING"
echo "  Arch     : ${ARCH}"
echo "  Dataset  : ${DATASET}"
echo "  Methods  : ${METHODS[*]}"
echo "  Forget   : random ${NUM_INDEXES} samples"
echo "  LR=${UNLEARN_LR}  gamma=${GAMMA}"
echo "  Seeds    : ${SEEDS[*]}"
echo "  GPUs     : ${GPUS[*]}"
echo "  Epochs   : ${EPOCHS[*]} (retrain: ${RETRAIN_EPOCHS})"
echo "  Out root : ${RESULTS_ROOT}/${DATASET}/${ARCH}/"
echo "============================================================"

OVERALL_FAILED=0
for METHOD in "${METHODS[@]}"; do
    echo ""
    echo ">>> Launching ${METHOD} (${ARCH}) across ${#SEEDS[@]} seed(s)"
    if [ "${METHOD}" = "retrain" ]; then
        METHOD_EPOCHS=${RETRAIN_EPOCHS}
        METHOD_LR=${RETRAIN_LR}
        METHOD_DECR_LR=${RETRAIN_DECREASING_LR}
    else
        METHOD_EPOCHS=""
        METHOD_LR=${UNLEARN_LR}
        METHOD_DECR_LR=${DECREASING_LR}
    fi

    FAILED=0
    for i in "${!SEEDS[@]}"; do
        EPOCHS_I=${METHOD_EPOCHS:-${EPOCHS[$i]}}
        run_one "${METHOD}" "${SEEDS[$i]}" "${GPUS[$i]}" "${EPOCHS_I}" "${METHOD_LR}" "${METHOD_DECR_LR}"
        if [ $? -ne 0 ]; then
            echo "FAILED: arch=${ARCH} method=${METHOD} seed=${SEEDS[$i]} gpu=${GPUS[$i]}"
            FAILED=$((FAILED + 1))
        fi
    done
    OVERALL_FAILED=$((OVERALL_FAILED + FAILED))
    echo "<<< ${METHOD} done — ${FAILED} failure(s)"
done

echo ""
echo "============================================================"
if [ "${OVERALL_FAILED}" -eq 0 ]; then
    echo "All (method × seed) runs completed successfully for ${ARCH}."
else
    echo "${OVERALL_FAILED} run(s) failed for ${ARCH}. See logs."
fi
echo "Results : ${RESULTS_ROOT}/${DATASET}/${ARCH}/"
echo "============================================================"

exit ${OVERALL_FAILED}
