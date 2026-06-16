#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# CIFAR-100 MUKSB vs MUNBa comparison at lr=0.07
# Runs both methods in parallel on the same GPU with the same setup.
# Only change vs the lr=0.03 runs: UNLEARN_LR=0.07
# ─────────────────────────────────────────────────────────────────

# ── Shared settings ───────────────────────────────────────────────
SEED=1
GPU=7
DATA_DIR="/storage/s25017/Datasets/CIFAR100"
MODEL_PATH="/scratch/s25017/MUKSB/Classification/checkpoints/resnet18_cifar100/0model_SA_best.pth.tar"
RESULTS_ROOT="results"
DATASET="cifar100"
ARCH="resnet18"
NUM_CLASSES=100
CLASS_TO_REPLACE=-1
NUM_INDEXES=4500
UNLEARN_EPOCHS=30
UNLEARN_LR=0.07
GAMMA=0.5
WITH_L1=false
ALPHA=0.2
BATCH_SIZE=512
PRINT_FREQ=50
DECREASING_LR="91,136"
EXTRA="lr_0_07"

cd "$(dirname "$0")/.."  # run from Classification/

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: base checkpoint not found at ${MODEL_PATH}."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────
# Worker: run one unlearn method and save to its own dir
# ─────────────────────────────────────────────────────────────────
run_method() {
    local METHOD=$1
    local TAG=$2   # subfolder name: muksb | munba

    local SAVE_DIR="${RESULTS_ROOT}/${DATASET}/random_${NUM_INDEXES}_bs${BATCH_SIZE}_${EXTRA}/${TAG}/seed${SEED}"
    local LOG_FILE="${SAVE_DIR}/run.log"
    mkdir -p "${SAVE_DIR}"

    echo "[${METHOD}] Starting → ${SAVE_DIR}"

    CMD="python main_random.py \
        --unlearn ${METHOD} \
        --unlearn_epochs ${UNLEARN_EPOCHS} \
        --unlearn_lr ${UNLEARN_LR} \
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
        --decreasing_lr ${DECREASING_LR} \
        --data ${DATA_DIR}"

    if [ "${WITH_L1}" = "true" ]; then
        CMD="${CMD} --with_l1"
    fi

    {
        echo "====== ${METHOD} (seed=${SEED}, gpu=${GPU}, lr=${UNLEARN_LR}) ======"
        eval ${CMD}
        if [ $? -ne 0 ]; then
            echo "ERROR: ${METHOD} failed"
            exit 1
        fi
        echo "====== DONE ${METHOD} ======"
        echo "  Output: ${SAVE_DIR}/"
    } 2>&1 | tee "${LOG_FILE}"
}

# ─────────────────────────────────────────────────────────────────
# Launch both in parallel
# ─────────────────────────────────────────────────────────────────
echo "=========================================="
echo "MUKSB vs MUNBa — CIFAR-100 | lr=${UNLEARN_LR} | epochs=${UNLEARN_EPOCHS}"
echo "  Forget : random ${NUM_INDEXES} samples (seed ${SEED}, class ${CLASS_TO_REPLACE})"
echo "  GPU=${GPU}  batch=${BATCH_SIZE}  gamma=${GAMMA}"
echo "=========================================="

run_method "MUKSB" "muksb" &
PID_MUKSB=$!

run_method "MUNBa" "munba" &
PID_MUNBA=$!

wait ${PID_MUKSB}; STATUS_MUKSB=$?
wait ${PID_MUNBA}; STATUS_MUNBA=$?

echo "=========================================="
echo "Results: ${RESULTS_ROOT}/${DATASET}/random_${NUM_INDEXES}_bs${BATCH_SIZE}_${EXTRA}/"
[ ${STATUS_MUKSB} -eq 0 ] && echo "  MUKSB : DONE" || echo "  MUKSB : FAILED"
[ ${STATUS_MUNBA} -eq 0 ] && echo "  MUNBa : DONE" || echo "  MUNBa : FAILED"
echo "=========================================="
