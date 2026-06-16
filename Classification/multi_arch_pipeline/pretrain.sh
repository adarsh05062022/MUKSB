#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# pretrain.sh — pre-train a model on full CIFAR-10
# Used to produce the checkpoint needed by MUKSB unlearning.
#
# Required env vars:
#   ARCH        — e.g. vgg16_bn | resnet18 | vgg16_bn_lth
#   SAVE_DIR    — where to save the checkpoint (creates 0model_SA_best.pth.tar)
# Optional:
#   GPU=0  EPOCHS=160  LR=0.1  BATCH_SIZE=256  SEED=2
#   DATASET=cifar10  NUM_CLASSES=10
#   DATA_DIR=/storage/s25017/Datasets/CIFAR10
#   DECREASING_LR="91,136"
#
# Usage:
#   ARCH=vgg16_bn SAVE_DIR=checkpoints/vgg16_bn_cifar10 bash pretrain.sh
# ─────────────────────────────────────────────────────────────────

set -uo pipefail

: "${GPU:=0}"
: "${EPOCHS:=160}"
: "${LR:=0.1}"
: "${BATCH_SIZE:=256}"
: "${SEED:=2}"
: "${DATASET:=cifar10}"
: "${NUM_CLASSES:=10}"
: "${DATA_DIR:=/storage/s25017/Datasets/CIFAR10}"
: "${DECREASING_LR:=91,136}"
: "${PRINT_FREQ:=50}"

if [ -z "${ARCH:-}" ] || [ -z "${SAVE_DIR:-}" ]; then
    echo "ERROR: ARCH and SAVE_DIR must be set."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve to absolute path if relative
case "${SAVE_DIR}" in
    /*) ;;
    *) SAVE_DIR="${CLS_DIR}/${SAVE_DIR}" ;;
esac

mkdir -p "${SAVE_DIR}"
CKPT="${SAVE_DIR}/0model_SA_best.pth.tar"

if [ -f "${CKPT}" ]; then
    echo "Checkpoint already exists, skipping pre-training: ${CKPT}"
    exit 0
fi

cd "${CLS_DIR}"

LOG="${SAVE_DIR}/pretrain.log"
echo "[pretrain] arch=${ARCH} gpu=${GPU} epochs=${EPOCHS} lr=${LR} log=${LOG}"

{
    echo "====== PRE-TRAIN ${ARCH} on ${DATASET} ======"
    python main_train.py \
        --arch          "${ARCH}" \
        --dataset       "${DATASET}" \
        --num_classes   "${NUM_CLASSES}" \
        --data          "${DATA_DIR}" \
        --save_dir      "${SAVE_DIR}" \
        --epochs        "${EPOCHS}" \
        --lr            "${LR}" \
        --batch_size    "${BATCH_SIZE}" \
        --seed          "${SEED}" \
        --gpu           "${GPU}" \
        --decreasing_lr "${DECREASING_LR}" \
        --print_freq    "${PRINT_FREQ}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "ERROR: pre-training failed (rc=${rc})"
        exit $rc
    fi
    echo "====== PRE-TRAIN DONE — checkpoint: ${CKPT} ======"
} 2>&1 | tee "${LOG}"
