#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MUKSB Classification — CIFAR-100 MUNBa unlearning (Nash baseline)
#
# Identical settings to run_pipeline_cifar100.sh seed1 MUKSB run:
#   same base checkpoint, same forget split (seed=1, class=-1, 4500 idx),
#   same lr/gamma/alpha/epochs/batch_size.
#
# Usage:
#   bash run_munba_cifar100.sh
# ─────────────────────────────────────────────────────────────────

# ── Single seed / GPU ─────────────────────────────────────────────
SEED=1
GPU=7

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR="/storage/s25017/Datasets/CIFAR100"
MODEL_PATH="/scratch/s25017/MUKSB/Classification/checkpoints/resnet18_cifar100/0model_SA_best.pth.tar"
RESULTS_ROOT="results"

# ── Dataset / Architecture ────────────────────────────────────────
DATASET="cifar100"
ARCH="resnet18"
NUM_CLASSES=100

# ── What to forget (MUST match MUKSB run for valid comparison) ────
CLASS_TO_REPLACE=-1
NUM_INDEXES=4500

# ── Unlearning hyperparams (identical to MUKSB run) ───────────────
UNLEARN_EPOCHS=30
UNLEARN_LR=0.03
GAMMA=0.5
WITH_L1=false
ALPHA=0.2

# ── Misc ──────────────────────────────────────────────────────────
BATCH_SIZE=512
PRINT_FREQ=50
DECREASING_LR="91,136"

EXTRA="lr_0_03"
FORGET_TAG="random_4500_bs512_${EXTRA}"
SAVE_DIR="${RESULTS_ROOT}/${DATASET}/${FORGET_TAG}/munba/seed${SEED}"
LOG_FILE="${SAVE_DIR}/run.log"

cd "$(dirname "$0")/.."  # scripts live in cifar100/ subdir

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: base checkpoint not found at ${MODEL_PATH}."
    echo "       Run Stage 1 first: STAGE=train bash run_pipeline_cifar100.sh"
    exit 1
fi

mkdir -p "${SAVE_DIR}"

echo "=========================================="
echo "MUNBa Unlearning (Nash baseline) — 1 seed"
echo "  Dataset : ${DATASET} | Arch: ${ARCH} | Classes: ${NUM_CLASSES}"
echo "  Forget  : random ${NUM_INDEXES} samples (seed ${SEED}, class ${CLASS_TO_REPLACE})"
echo "  LR=${UNLEARN_LR}  gamma=${GAMMA}  bs=${BATCH_SIZE}  epochs=${UNLEARN_EPOCHS}"
echo "  Seed=${SEED}  GPU=${GPU}"
echo "  Log: ${LOG_FILE}"
echo "=========================================="

{
    echo "====== MUNBa UNLEARNING (seed=${SEED}, gpu=${GPU}) ======"
    CMD="python main_random.py \
        --unlearn MUNBa \
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

    eval ${CMD}

    if [ $? -ne 0 ]; then
        echo "ERROR: MUNBa unlearning failed for seed=${SEED}"
        exit 1
    fi

    echo "====== DONE (seed=${SEED}) ======"
    echo "  Output: ${SAVE_DIR}/"
    echo "  Epoch metrics: ${SAVE_DIR}/epoch_metrics.json"

} 2>&1 | tee "${LOG_FILE}"

echo "=========================================="
echo "Results: ${SAVE_DIR}/"
echo "=========================================="
