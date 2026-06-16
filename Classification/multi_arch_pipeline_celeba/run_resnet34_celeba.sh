#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Run MUKSB + retrain on ResNet-34 / CelebA-HQ-307
# across multiple forget fractions (10 → 50%) sequentially.
#
# Each forget fraction runs its seeds in parallel on separate GPUs.
#
# Usage:
#   bash run_resnet34_celeba.sh
#   nohup bash run_resnet34_celeba.sh > celeba_sweep.out 2>&1 &
# Run only specific fractions:
#   FORGET_FRACTIONS=(0.1 0.3) nohup bash run_resnet34_celeba.sh > celeba_sweep.out 2>&1 &
# ─────────────────────────────────────────────────────────────────

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export ARCH="resnet34"
export DATASET="celeba"
export NUM_CLASSES=307
export DATA_DIR="/storage/s25017/Datasets/CELEB_HQ_307"
export MODEL_PATH="${CLS_DIR}/checkpoints/resnet34_celeba/0model_SA_best.pth.tar"
export RESULTS_ROOT="${CLS_DIR}/results_multi_arch"

# ── Unlearning hyperparams (MUKSB / FT) ───────────────────────────
# Per paper: lr=0.02, 10 epochs, bs=8, SGD
export UNLEARN_LR=0.02
export UNLEARN_EPOCHS=10
export GAMMA=0.5
export ALPHA=0.2
export BATCH_SIZE=32              # MUKSB / FT batch size
export DECREASING_LR="91,136"
export EXTRA="lr_0_02"
export CLASS_TO_REPLACE=-1
export NUM_INDEXES=4500           # dummy, ignored by CelebA loader

# ── Retrain hyperparams (match base-model training) ────────────────
# Per paper: SGD, lr=1e-3, 10 epochs, bs=8, cosine schedule.
# Cosine schedule is triggered by --imagenet_arch flag in impl.py.
export RETRAIN_LR=1e-3
export RETRAIN_EPOCHS=10
export RETRAIN_BATCH_SIZE=8       # retrain uses smaller bs per paper
export RETRAIN_IMAGENET_ARCH=true # enables cosine LR schedule for retrain

# ── Seeds / GPUs / Epochs per seed ───────────────────────────────
SEEDS=(272)
GPUS=(7)
EPOCHS=(10)

export METHODS=(MUKSB retrain FT)

# ── Forget fractions to sweep ─────────────────────────────────────
# Override from outside: FORGET_FRACTIONS=(0.1 0.3) bash run_resnet34_celeba.sh
: "${FORGET_FRACTIONS:=}"
if [ -z "${FORGET_FRACTIONS}" ]; then
    FORGET_FRACTIONS=(0.1 0.2 0.3 0.4 0.5)
fi

# ── Validate checkpoint ───────────────────────────────────────────
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: ResNet-34 CelebA checkpoint not found: ${MODEL_PATH}"
    exit 1
fi

echo "############################################################"
echo "#  ResNet-34 / CelebA-HQ-307 — forget fraction sweep"
echo "#  Fractions : ${FORGET_FRACTIONS[*]}"
echo "#  Methods   : ${METHODS[*]}"
echo "#  Seeds     : ${SEEDS[*]}"
echo "#  GPUs      : ${GPUS[*]}"
echo "############################################################"

SWEEP_FAILED=0
for FRAC in "${FORGET_FRACTIONS[@]}"; do
    export FORGET_FRACTION="${FRAC}"
    PCT=$(echo "${FRAC} * 100" | bc | cut -d. -f1)

    echo ""
    echo "############################################################"
    echo "#  Forget fraction: ${FRAC} (${PCT}%)"
    echo "############################################################"

    source "${SCRIPT_DIR}/run_unlearn_celeba.sh"
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "WARNING: forget=${PCT}% pipeline returned non-zero (rc=${rc})"
        SWEEP_FAILED=$((SWEEP_FAILED + 1))
    fi
done

echo ""
echo "############################################################"
echo "#  Sweep complete"
echo "############################################################"
if [ "${SWEEP_FAILED}" -eq 0 ]; then
    echo "All forget-fraction runs completed successfully."
else
    echo "${SWEEP_FAILED} forget-fraction run(s) had failures."
fi
echo "Results root: ${RESULTS_ROOT}/${DATASET}/${ARCH}/"
exit ${SWEEP_FAILED}
