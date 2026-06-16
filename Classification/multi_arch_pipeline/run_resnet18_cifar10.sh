#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Run MUKSB + retrain on ResNet-18 / CIFAR-10
# Uses the existing pretrained checkpoint under checkpoints/resnet18_cifar10/.
# ─────────────────────────────────────────────────────────────────

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export ARCH="resnet18"
export DATASET="cifar10"
export NUM_CLASSES=10
export DATA_DIR="/storage/s25017/Datasets/CIFAR10"
export MODEL_PATH="${CLS_DIR}/checkpoints/resnet18_cifar10/0model_SA_best.pth.tar"
export RESULTS_ROOT="${CLS_DIR}/results_multi_arch"

# Forget / hyperparams
export NUM_INDEXES=22500
export UNLEARN_LR=0.03
export GAMMA=0.5
export ALPHA=0.2
export BATCH_SIZE=512
export DECREASING_LR="91,136"
export EXTRA="lr_0_03"

# Seeds / GPUs / epochs (one per parallel worker)
SEEDS=(1 )
GPUS=(4)
EPOCHS=(30)
export RETRAIN_EPOCHS=160

METHODS=(MUKSB retrain FT)

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: ResNet-18 checkpoint not found: ${MODEL_PATH}"
    exit 1
fi

source "${SCRIPT_DIR}/run_unlearn.sh"
