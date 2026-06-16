#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Run MUKSB + retrain on Swin-T / CIFAR-10
#
# Swin-T is a 28 M-parameter Vision Transformer designed for 224×224
# inputs.  The model wrapper (models/SwinT.py) upsamples the 32×32
# CIFAR inputs to 224×224 internally, so the dataloaders are
# untouched.  Pre-training fine-tunes ImageNet weights on CIFAR-10
# (set MUKSB_SWIN_PRETRAINED=0 to disable).
#
# Smaller batch size / lower LR than the convnet wrappers to suit
# the higher memory footprint and ViT fine-tuning dynamics.
#
# Usage:
#   bash run_swin_t_cifar10.sh
# ─────────────────────────────────────────────────────────────────

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export ARCH="swin_t"
export DATASET="cifar10"
export NUM_CLASSES=10
export DATA_DIR="/storage/s25017/Datasets/CIFAR10"

CKPT_DIR="${CLS_DIR}/checkpoints/swin_t_cifar10"
export MODEL_PATH="${CKPT_DIR}/0model_SA_best.pth.tar"
export RESULTS_ROOT="${CLS_DIR}/results_multi_arch"

# Forget set kept identical to convnets for direct comparison
export NUM_INDEXES=4500

# ── Unlearn hyperparams (ViT-tuned) ───────────────────────────────
# Lower LR is standard for ViT fine-tuning.
export UNLEARN_LR="${UNLEARN_LR:-0.001}"
export GAMMA="${GAMMA:-0.5}"
export ALPHA="${ALPHA:-0.2}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export DECREASING_LR="${DECREASING_LR:-20,25}"
export EXTRA="${EXTRA:-lr_0_001}"

# Retrain-specific: more epochs and a standard ViT training LR.
# 100 epochs with milestones at 60 and 80 follows the same
# decay cadence as pretraining but gives the model time to converge
# from scratch on the retain set.
export RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-100}"
export RETRAIN_LR="${RETRAIN_LR:-0.001}"
export RETRAIN_DECREASING_LR="${RETRAIN_DECREASING_LR:-60,80}"

# Seeds / GPUs / epochs (one per parallel worker)
# Drop to 3 seeds by default because Swin-T runs are slower than the
# convnets — override SEEDS/GPUS/EPOCHS to scale up.
SEEDS=(${SEEDS:-1 2 3})
GPUS=(${GPUS:-0 1 2})
EPOCHS=(${EPOCHS:-30 30 30})

METHODS=(${METHODS:-MUKSB FT retrain})

# ── Step 1: pre-train (ImageNet → CIFAR-10 fine-tune) if needed ──
if [ ! -f "${MODEL_PATH}" ]; then
    echo ">>> No Swin-T CIFAR-10 checkpoint, pre-training first (ImageNet init)."
    ARCH="${ARCH}" \
    SAVE_DIR="${CKPT_DIR}" \
    GPU="${PRETRAIN_GPU:-0}" \
    EPOCHS="${PRETRAIN_EPOCHS:-30}" \
    LR="${PRETRAIN_LR:-0.001}" \
    BATCH_SIZE="${PRETRAIN_BATCH:-128}" \
    SEED="${PRETRAIN_SEED:-2}" \
    DATASET="${DATASET}" \
    NUM_CLASSES="${NUM_CLASSES}" \
    DATA_DIR="${DATA_DIR}" \
    DECREASING_LR="${PRETRAIN_DECREASING_LR:-20,25}" \
    MUKSB_SWIN_PRETRAINED="${MUKSB_SWIN_PRETRAINED:-1}" \
    bash "${SCRIPT_DIR}/pretrain.sh"
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "${MODEL_PATH}" ]; then
        echo "ERROR: Swin-T pre-training did not produce ${MODEL_PATH}"
        exit 1
    fi
fi

# Make sure the unlearning runs also load random-init compatible state
# (only matters if MUKSB_SWIN_PRETRAINED=0 was used for pre-training).
export MUKSB_SWIN_PRETRAINED="${MUKSB_SWIN_PRETRAINED:-1}"

# ── Step 2: run MUKSB + retrain ───────────────────────────────────
source "${SCRIPT_DIR}/run_unlearn.sh"
