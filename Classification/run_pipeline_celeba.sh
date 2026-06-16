#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MUKSB Classification — full pipeline for CelebA
# (unlearning on the full parameter space, one seed per GPU in parallel)
#
# Forget set: 10% of identity classes (~30 of 307), handled
# internally by the CelebA dataloader — no NUM_INDEXES needed.
#
# Usage:
#   bash run_pipeline_celeba.sh
# ─────────────────────────────────────────────────────────────────

# ── Seeds, GPUs, and epochs (all arrays must be same length) ──────
# EPOCHS[i] sets unlearn_epochs for SEEDS[i]
SEEDS=(272 474 838 492 385)
GPUS=(7  6  3  5  0 )
EPOCHS=(20 20 20 20 20)   # set to empty string "" to use default UNLEARN_EPOCHS

# ── Paths ─────────────────────────────────────────────────────────
MODEL_PATH="/scratch/s25017/MUKSB/Classification/checkpoints/resnet34_celeba/0model_SA_best.pth.tar"
DATA_DIR="/storage/s25017/Datasets/CELEB_HQ_307"
RESULTS_ROOT="results"

# ── Dataset / Architecture ────────────────────────────────────────
DATASET="celeba"
ARCH="resnet34"
NUM_CLASSES=307

# ── Unlearning hyperparams ────────────────────────────────────────
UNLEARN_EPOCHS=20
# UNLEARN_LR=0.017
UNLEARN_LR=0.03
GAMMA=0.5
WITH_L1=false
ALPHA=0.2

EXTRA="lr_0_03"


# ── Misc ──────────────────────────────────────────────────────────
BATCH_SIZE=32
PRINT_FREQ=50
DECREASING_LR="91,136"

# ── What to forget ────────────────────────────────────────────────
# FORGET_FRACTION: fraction of identity classes to forget (0.1 = 10%)
# CLASS_TO_REPLACE / NUM_INDEXES are required by arg_parser but ignored by CelebA loader.
FORGET_FRACTION=0.5
CLASS_TO_REPLACE=-1
NUM_INDEXES=4500
FORGET_PCT=$(echo "${FORGET_FRACTION} * 100" | bc | cut -d. -f1)
FORGET_TAG="random_${FORGET_PCT}pct_bs${BATCH_SIZE}_epochs${UNLEARN_EPOCHS}_${EXTRA}"   # used in folder names




# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────
if [ ${#SEEDS[@]} -ne ${#GPUS[@]} ] || [ ${#SEEDS[@]} -ne ${#EPOCHS[@]} ]; then
    echo "ERROR: SEEDS, GPUS, and EPOCHS arrays must all be the same length."
    exit 1
fi

cd "$(dirname "$0")"

# ─────────────────────────────────────────────────────────────────
# Per-seed worker function (runs in background)
# ─────────────────────────────────────────────────────────────────
run_seed() {
    local SEED=$1
    local GPU=$2
    local EPOCHS_FOR_SEED=${3:-${UNLEARN_EPOCHS}}

    local SAVE_DIR="${RESULTS_ROOT}/${DATASET}/${FORGET_TAG}/muksb_output/seed${SEED}"
    local LOG_FILE="${SAVE_DIR}/run.log"

    mkdir -p "${SAVE_DIR}"

    echo "[seed=${SEED} gpu=${GPU}] Starting unlearning → log: ${LOG_FILE}"

    {
        echo "====== UNLEARNING (seed=${SEED}, gpu=${GPU}) ======"
        CMD="python main_random.py \
            --unlearn MUKSB \
            --unlearn_epochs ${EPOCHS_FOR_SEED} \
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
            --data ${DATA_DIR} \
            --forget_fraction ${FORGET_FRACTION}"

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
}

# ─────────────────────────────────────────────────────────────────
# Launch all seeds in parallel
# ─────────────────────────────────────────────────────────────────
echo "=========================================="
echo "MUKSB CelebA pipeline — ${#SEEDS[@]} seed(s)"
echo "  Dataset : ${DATASET} | Arch: ${ARCH} | Classes: ${NUM_CLASSES}"
echo "  Forget  : ${FORGET_PCT}% of identity classes (forget_fraction=${FORGET_FRACTION})"
echo "  LR=${UNLEARN_LR}  gamma=${GAMMA}"
echo "  Seeds   : ${SEEDS[*]}"
echo "  GPUs    : ${GPUS[*]}"
echo "  Epochs  : ${EPOCHS[*]}"
echo "=========================================="

PIDS=()
for i in "${!SEEDS[@]}"; do
    run_seed "${SEEDS[$i]}" "${GPUS[$i]}" "${EPOCHS[$i]}" &
    PIDS+=($!)
done

# Wait for all and report failures
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    if [ $? -ne 0 ]; then
        echo "FAILED: seed=${SEEDS[$i]} gpu=${GPUS[$i]}"
        FAILED=$((FAILED + 1))
    fi
done

echo "=========================================="
if [ "${FAILED}" -eq 0 ]; then
    echo "All seeds completed successfully."
else
    echo "${FAILED} seed(s) failed. Check the log files above."
fi
echo "Results root: ${RESULTS_ROOT}/${DATASET}/${FORGET_TAG}/"
echo "=========================================="
