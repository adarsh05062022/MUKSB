#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Run MUKSB + retrain on VGG-16 (with BN) / CIFAR-10
# Pre-trains the model on full CIFAR-10 if no checkpoint exists.
#
# Usage:
#   bash run_vgg16_cifar10.sh
# Override the pretrain GPU/epochs:
#   PRETRAIN_GPU=0 PRETRAIN_EPOCHS=160 bash run_vgg16_cifar10.sh
# ─────────────────────────────────────────────────────────────────

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export ARCH="vgg16_bn"
export DATASET="cifar10"
export NUM_CLASSES=10
export DATA_DIR="/storage/s25017/Datasets/CIFAR10"

CKPT_DIR="${CLS_DIR}/checkpoints/vgg16_bn_cifar10"
export MODEL_PATH="${CKPT_DIR}/0model_SA_best.pth.tar"
export RESULTS_ROOT="${CLS_DIR}/results_multi_arch"

# Forget / hyperparams — same as ResNet-18 for direct comparison
export NUM_INDEXES=4500
export UNLEARN_LR=0.03
export GAMMA=0.5
export ALPHA=0.2
export BATCH_SIZE=512
export DECREASING_LR="91,136"
export EXTRA="lr_0_03"

# Seeds / GPUs / epochs (one per parallel worker)
SEEDS=(1 )
GPUS=(6 )
EPOCHS=(30 )

METHODS=(MUKSB retrain FT)

# ── Step 1: pre-train if necessary ────────────────────────────────
if [ ! -f "${MODEL_PATH}" ]; then
    echo ">>> No VGG-16 checkpoint found, pre-training first."
    ARCH="${ARCH}" \
    SAVE_DIR="${CKPT_DIR}" \
    GPU="${PRETRAIN_GPU:-0}" \
    EPOCHS="${PRETRAIN_EPOCHS:-160}" \
    LR="${PRETRAIN_LR:-0.1}" \
    BATCH_SIZE="${PRETRAIN_BATCH:-256}" \
    SEED="${PRETRAIN_SEED:-2}" \
    DATASET="${DATASET}" \
    NUM_CLASSES="${NUM_CLASSES}" \
    DATA_DIR="${DATA_DIR}" \
    bash "${SCRIPT_DIR}/pretrain.sh"
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "${MODEL_PATH}" ]; then
        echo "ERROR: VGG-16 pre-training did not produce ${MODEL_PATH}"
        exit 1
    fi
fi

# ── Step 2: run MUKSB + retrain ───────────────────────────────────
source "${SCRIPT_DIR}/run_unlearn.sh"
