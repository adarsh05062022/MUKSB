#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MUKSB Classification — CIFAR-100 RETRAIN baseline (gold standard)
#
# Trains a ResNet-18 FROM SCRATCH on the RETAIN set only (the 4500
# randomly-forgotten samples are excluded). This is the reference
# "retrained-without-forget-set" model used to judge how well MUKSB
# (and other methods) approximate exact unlearning.
#
# IMPORTANT — same forget split as the MUKSB run:
#   The forget/retain split is determined by --seed, --class_to_replace
#   and --num_indexes_to_replace. We reuse SEED=1, class=-1, 4500 so the
#   excluded samples are identical to run_pipeline_cifar100.sh's seed1 run.
#
#   With --unlearn retrain, main_random.py does NOT load the base
#   checkpoint into the model (it trains from random init); --mask is
#   still required because the file is torch.load-ed regardless.
#
# Usage:
#   bash run_retrain_cifar100.sh
# ─────────────────────────────────────────────────────────────────

# ── Single seed / GPU ─────────────────────────────────────────────
SEED=1
GPU=7

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR="/storage/s25017/Datasets/CIFAR100"
CKPT_DIR="/scratch/s25017/MUKSB/Classification/checkpoints/resnet18_cifar100"
MODEL_PATH="${CKPT_DIR}/0model_SA_best.pth.tar"   # only used as --mask (not loaded)
RESULTS_ROOT="results"

# ── Dataset / Architecture ────────────────────────────────────────
DATASET="cifar100"
ARCH="resnet18"
NUM_CLASSES=100

# ── What to forget (MUST match the MUKSB run for a valid comparison) ──
CLASS_TO_REPLACE=-1
NUM_INDEXES=4500

# ── Retrain recipe (mirror the base-model training schedule) ──────
RETRAIN_EPOCHS=182
RETRAIN_LR=0.1
DECREASING_LR="91,136"     # MultiStepLR milestones (gamma=0.1)
BATCH_SIZE=256
PRINT_FREQ=50

# Place the baseline right next to the MUKSB output for easy comparison:
#   results/cifar100/random_4500_bs512_lr_0_03/output/seed1   (MUKSB)
#   results/cifar100/random_4500_bs512_lr_0_03/retrain/seed1  (this)
FORGET_TAG="random_4500_bs512_lr_0_03"
SAVE_DIR="${RESULTS_ROOT}/${DATASET}/${FORGET_TAG}/retrain/seed${SEED}"
LOG_FILE="${SAVE_DIR}/run.log"

cd "$(dirname "$0")/.."  # scripts live in cifar100/ subdir

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: base checkpoint not found at ${MODEL_PATH}."
    echo "       (--mask is required even for retrain). Run Stage 1 first:"
    echo "       STAGE=train bash run_pipeline_cifar100.sh"
    exit 1
fi

mkdir -p "${SAVE_DIR}"

echo "=========================================="
echo "RETRAIN baseline (retain-only) — 1 seed"
echo "  Dataset : ${DATASET} | Arch: ${ARCH} | Classes: ${NUM_CLASSES}"
echo "  Exclude : random ${NUM_INDEXES} samples (seed ${SEED}, class ${CLASS_TO_REPLACE})"
echo "  Recipe  : ${RETRAIN_EPOCHS} ep, lr=${RETRAIN_LR}, steps=${DECREASING_LR}, bs=${BATCH_SIZE}"
echo "  Seed=${SEED}  GPU=${GPU}"
echo "  Log: ${LOG_FILE}"
echo "=========================================="

{
    echo "====== RETRAIN (seed=${SEED}, gpu=${GPU}) ======"
    python main_random.py \
        --unlearn retrain \
        --unlearn_epochs ${RETRAIN_EPOCHS} \
        --unlearn_lr ${RETRAIN_LR} \
        --mask ${MODEL_PATH} \
        --save_dir ${SAVE_DIR} \
        --dataset ${DATASET} \
        --arch ${ARCH} \
        --num_classes ${NUM_CLASSES} \
        --gpu ${GPU} \
        --class_to_replace ${CLASS_TO_REPLACE} \
        --num_indexes_to_replace ${NUM_INDEXES} \
        --batch_size ${BATCH_SIZE} \
        --seed ${SEED} \
        --print_freq ${PRINT_FREQ} \
        --decreasing_lr ${DECREASING_LR} \
        --data ${DATA_DIR}

    if [ $? -ne 0 ]; then
        echo "ERROR: retrain failed for seed=${SEED}"
        exit 1
    fi

    echo "====== DONE (seed=${SEED}) ======"
    echo "  Output: ${SAVE_DIR}/"
    echo "  Epoch metrics: ${SAVE_DIR}/epoch_metrics.json"

} 2>&1 | tee "${LOG_FILE}"

echo "=========================================="
echo "Retrain baseline saved under: ${SAVE_DIR}/"
echo "=========================================="
