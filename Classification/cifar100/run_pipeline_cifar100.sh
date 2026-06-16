#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MUKSB Classification — full CIFAR-100 pipeline (RANDOM forgetting)
#   Stage 1: train a base ResNet-18 on CIFAR-100 (skipped if ckpt exists)
#   Stage 2: run MUKSB random unlearning for a single seed
#
# CIFAR-100 is auto-downloaded by torchvision into ${DATA_DIR} the
# first time (download=True in dataset.py).
#
# Usage:
#   bash run_pipeline_cifar100.sh                # both stages
#   STAGE=train   bash run_pipeline_cifar100.sh  # only train base model
#   STAGE=unlearn bash run_pipeline_cifar100.sh  # only run unlearning
# ─────────────────────────────────────────────────────────────────

STAGE=${STAGE:-all}   # all | train | unlearn

# ── Single seed / GPU / unlearn epochs ────────────────────────────
SEED=1
GPU=7
UNLEARN_EPOCHS=30

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR="/storage/s25017/Datasets/CIFAR100"
CKPT_DIR="/scratch/s25017/MUKSB/Classification/checkpoints/resnet18_cifar100"
MODEL_PATH="${CKPT_DIR}/0model_SA_best.pth.tar"
RESULTS_ROOT="results"

# ── Dataset / Architecture ────────────────────────────────────────
DATASET="cifar100"
ARCH="resnet18"
NUM_CLASSES=100

# ── What to forget ────────────────────────────────────────────────
# Random forgetting: class_to_replace=-1 marks samples across all classes,
# NUM_INDEXES of them (4500 = 10% of the 45k CIFAR-100 train split).
CLASS_TO_REPLACE=-1
NUM_INDEXES=4500

# ── Base-model training hyperparams (Stage 1) ─────────────────────
TRAIN_EPOCHS=182          # standard CIFAR ResNet schedule
TRAIN_LR=0.1
TRAIN_BS=256
TRAIN_DECREASING_LR="91,136"

# ── Unlearning hyperparams (Stage 2) ──────────────────────────────
UNLEARN_LR=0.03
GAMMA=0.5
WITH_L1=false
ALPHA=0.2

# ── Misc ──────────────────────────────────────────────────────────
BATCH_SIZE=512
PRINT_FREQ=50
DECREASING_LR="91,136"

EXTRA="lr_0_03"
FORGET_TAG="random_${NUM_INDEXES}_bs${BATCH_SIZE}_${EXTRA}"   # used in folder names

cd "$(dirname "$0")/.."  # scripts live in cifar100/ subdir
mkdir -p "${DATA_DIR}" "${CKPT_DIR}"

# ─────────────────────────────────────────────────────────────────
# Stage 1: Train base CIFAR-100 model (skipped if checkpoint exists)
# ─────────────────────────────────────────────────────────────────
train_base() {
    if [ -f "${MODEL_PATH}" ]; then
        echo "[stage1] Base checkpoint already exists at ${MODEL_PATH} — skipping training."
        return 0
    fi

    echo "=========================================="
    echo "Stage 1: Training base ${ARCH} on ${DATASET}"
    echo "  GPU=${GPU}  epochs=${TRAIN_EPOCHS}  lr=${TRAIN_LR}  bs=${TRAIN_BS}"
    echo "  Save dir: ${CKPT_DIR}"
    echo "=========================================="

    python main_train.py \
        --dataset ${DATASET} \
        --arch ${ARCH} \
        --num_classes ${NUM_CLASSES} \
        --data ${DATA_DIR} \
        --save_dir ${CKPT_DIR} \
        --gpu ${GPU} \
        --epochs ${TRAIN_EPOCHS} \
        --lr ${TRAIN_LR} \
        --batch_size ${TRAIN_BS} \
        --decreasing_lr ${TRAIN_DECREASING_LR} \
        --seed ${SEED} \
        --print_freq ${PRINT_FREQ} \
        2>&1 | tee "${CKPT_DIR}/train.log"

    if [ ! -f "${MODEL_PATH}" ]; then
        echo "ERROR: training finished but ${MODEL_PATH} was not produced."
        exit 1
    fi
    echo "[stage1] Base checkpoint saved → ${MODEL_PATH}"
}

# ─────────────────────────────────────────────────────────────────
# Stage 2: MUKSB random unlearning (single seed)
# ─────────────────────────────────────────────────────────────────
run_unlearning() {
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "ERROR: base checkpoint not found at ${MODEL_PATH}."
        echo "       Run Stage 1 first:  STAGE=train bash run_pipeline_cifar100.sh"
        exit 1
    fi

    local SAVE_DIR="${RESULTS_ROOT}/${DATASET}/${FORGET_TAG}/output/seed${SEED}"
    local LOG_FILE="${SAVE_DIR}/run.log"
    mkdir -p "${SAVE_DIR}"

    echo "=========================================="
    echo "Stage 2: MUKSB Random Unlearning — 1 seed"
    echo "  Dataset : ${DATASET} | Arch: ${ARCH} | Classes: ${NUM_CLASSES}"
    echo "  Forget  : random ${NUM_INDEXES} samples"
    echo "  LR=${UNLEARN_LR}  gamma=${GAMMA}  bs=${BATCH_SIZE}  epochs=${UNLEARN_EPOCHS}"
    echo "  Seed=${SEED}  GPU=${GPU}"
    echo "  Log: ${LOG_FILE}"
    echo "=========================================="

    {
        echo "====== UNLEARNING (seed=${SEED}, gpu=${GPU}) ======"
        CMD="python main_random.py \
            --unlearn MUKSB \
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
            echo "ERROR: unlearning failed for seed=${SEED}"
            exit 1
        fi

        echo "====== DONE (seed=${SEED}) ======"
        echo "  Output: ${SAVE_DIR}/"
        echo "  Epoch metrics: ${SAVE_DIR}/epoch_metrics.json"

    } 2>&1 | tee "${LOG_FILE}"

    echo "=========================================="
    echo "Results root: ${RESULTS_ROOT}/${DATASET}/${FORGET_TAG}/"
    echo "=========================================="
}

# ─────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────
case "${STAGE}" in
    train)   train_base ;;
    unlearn) run_unlearning ;;
    all)     train_base && run_unlearning ;;
    *)       echo "ERROR: unknown STAGE='${STAGE}' (use train | unlearn | all)"; exit 1 ;;
esac
