#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Sweep forget-set sizes (10%–50%) for VGG-16 (BN) / CIFAR-10
# Runs: 10% (4500), 20% (9000), 30% (13500), 40% (18000), 50% (22500)
# Seed: 1 only
# ─────────────────────────────────────────────────────────────────

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CKPT_DIR="${CLS_DIR}/checkpoints/vgg16_bn_cifar10"
MODEL_PATH="${CKPT_DIR}/0model_SA_best.pth.tar"

# ── Pre-train VGG-16 if checkpoint is missing ─────────────────────
if [ ! -f "${MODEL_PATH}" ]; then
    echo ">>> No VGG-16 checkpoint found, pre-training first."
    ARCH="vgg16_bn" \
    SAVE_DIR="${CKPT_DIR}" \
    GPU="${PRETRAIN_GPU:-0}" \
    EPOCHS="${PRETRAIN_EPOCHS:-160}" \
    LR="${PRETRAIN_LR:-0.1}" \
    BATCH_SIZE="${PRETRAIN_BATCH:-256}" \
    SEED="${PRETRAIN_SEED:-2}" \
    DATASET="cifar10" \
    NUM_CLASSES="10" \
    DATA_DIR="/storage/s25017/Datasets/CIFAR10" \
    bash "${SCRIPT_DIR}/pretrain.sh"
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "${MODEL_PATH}" ]; then
        echo "ERROR: VGG-16 pre-training did not produce ${MODEL_PATH}"
        exit 1
    fi
fi

# ── Sweep over forget-set sizes ───────────────────────────────────
# CIFAR-10 train = 45000 samples
# 10%→4500  20%→9000  30%→13500  40%→18000  50%→22500
FORGET_SIZES=(4500 9000)
FORGET_PCTS=(10    20)

OVERALL_FAILED=0

PIDS=()
PCT_FOR_PID=()

for idx in "${!FORGET_SIZES[@]}"; do
    NUM_IDX="${FORGET_SIZES[$idx]}"
    PCT="${FORGET_PCTS[$idx]}"

    echo ""
    echo "############################################################"
    echo "  VGG-16 / CIFAR-10 — forget ${PCT}% (${NUM_IDX} samples)"
    echo "############################################################"

    (
        export ARCH="vgg16_bn"
        export DATASET="cifar10"
        export NUM_CLASSES=10
        export DATA_DIR="/storage/s25017/Datasets/CIFAR10"
        export MODEL_PATH="${MODEL_PATH}"
        export RESULTS_ROOT="${CLS_DIR}/results_multi_arch"

        export NUM_INDEXES="${NUM_IDX}"
        export UNLEARN_LR=0.017
        export GAMMA=0.5
        export ALPHA=0.2
        export BATCH_SIZE=256
        export DECREASING_LR="91,136"
        export EXTRA="lr_0_017_forget_${PCT}pct"

        SEEDS=(273)
        GPUS=(3)
        EPOCHS=(30)
        RETRAIN_EPOCHS=160

        METHODS=(MUKSB FT RL Salun MUNBa retrain)

        source "${SCRIPT_DIR}/run_unlearn.sh"
    ) &
    PIDS+=($!)
    PCT_FOR_PID+=("${PCT}")
done

for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "ERROR: forget ${PCT_FOR_PID[$i]}% run failed (rc=${rc})"
        OVERALL_FAILED=$((OVERALL_FAILED + 1))
    fi
done

echo ""
echo "############################################################"
if [ "${OVERALL_FAILED}" -eq 0 ]; then
    echo "All forget-size sweeps completed successfully."
else
    echo "${OVERALL_FAILED} sweep(s) failed. See per-run logs above."
fi
echo "############################################################"

exit ${OVERALL_FAILED}
