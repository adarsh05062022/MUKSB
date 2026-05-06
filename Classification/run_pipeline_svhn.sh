#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MUKSB Classification — full SVHN pipeline (CLASS-WISE forgetting)
#   Stage 1: train a base ResNet-18 on SVHN (skipped if checkpoint exists)
#   Stage 2: run MUKSB class-wise unlearning, one seed per GPU in parallel
#
# Matches the SVHN setup in MUNBa Table 8:
#   SGD, batch size 256, lr 0.1, 50 epochs for the base/retrain model.
#   (We approximate the paper's cosine schedule with a MultiStepLR
#   because main_train.py only enables cosine under --imagenet_arch,
#   which would also switch ResNet-18 to its ImageNet variant.)
#
# SVHN is auto-downloaded by torchvision into ${DATA_DIR} the first time.
#
# Usage:
#   bash run_pipeline_svhn.sh                # both stages
#   STAGE=train  bash run_pipeline_svhn.sh   # only train base model
#   STAGE=unlearn bash run_pipeline_svhn.sh  # only run unlearning
# ─────────────────────────────────────────────────────────────────

STAGE=${STAGE:-all}   # all | train | unlearn

# ── Seeds, GPUs, and epochs (all arrays must be same length) ──────
# EPOCHS[i] sets unlearn_epochs for SEEDS[i]
SEEDS=(1  2)
GPUS=(1  1)
EPOCHS=(10 10)

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR="/storage/s25017/Datasets/SVHN"
CKPT_DIR="/storage/s25017/MUKSB/Classification/checkpoints/resnet18_svhn"
MODEL_PATH="${CKPT_DIR}/0model_SA_best.pth.tar"
RESULTS_ROOT="results"

# ── Dataset / Architecture ────────────────────────────────────────
DATASET="svhn"
ARCH="resnet18"
NUM_CLASSES=10

# ── What to forget ────────────────────────────────────────────────
# Class-wise forgetting: pick one digit class (0–9) to forget.
# num_indexes_to_replace is left unset → main_forget.py forgets ALL samples of that class.
CLASS_TO_FORGET=1     # paper-style: forget a single class

# ── Base-model training hyperparams (Stage 1, paper-aligned) ──────
TRAIN_GPU=${GPUS[0]}
TRAIN_EPOCHS=50          # paper: 50 epochs on SVHN
TRAIN_LR=0.1             # paper: cosine init at 0.1
TRAIN_BS=256             # paper: batch size 256
TRAIN_DECREASING_LR="25,40"   # MultiStepLR approx of cosine over 50 epochs
TRAIN_SEED=1

# ── Unlearning hyperparams (Stage 2) ──────────────────────────────
UNLEARN_EPOCHS=10
UNLEARN_LR=0.013
GAMMA=0.5
WITH_L1=false
ALPHA=0.2

# ── Misc ──────────────────────────────────────────────────────────
BATCH_SIZE=256
PRINT_FREQ=50
DECREASING_LR="91,136"

EXTRA="lr_${UNLEARN_LR}"
FORGET_TAG="class_${CLASS_TO_FORGET}_bs${BATCH_SIZE}_${EXTRA}"

# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────
if [ ${#SEEDS[@]} -ne ${#GPUS[@]} ] || [ ${#SEEDS[@]} -ne ${#EPOCHS[@]} ]; then
    echo "ERROR: SEEDS, GPUS, and EPOCHS arrays must all be the same length."
    exit 1
fi

cd "$(dirname "$0")"
mkdir -p "${DATA_DIR}" "${CKPT_DIR}"

# ─────────────────────────────────────────────────────────────────
# Stage 1: Train base SVHN model (skipped if checkpoint already exists)
# ─────────────────────────────────────────────────────────────────
train_base() {
    if [ -f "${MODEL_PATH}" ]; then
        echo "[stage1] Base checkpoint already exists at ${MODEL_PATH} — skipping training."
        return 0
    fi

    echo "=========================================="
    echo "Stage 1: Training base ${ARCH} on ${DATASET}"
    echo "  GPU=${TRAIN_GPU}  epochs=${TRAIN_EPOCHS}  lr=${TRAIN_LR}  bs=${TRAIN_BS}"
    echo "  Save dir: ${CKPT_DIR}"
    echo "=========================================="

    python main_train.py \
        --dataset ${DATASET} \
        --arch ${ARCH} \
        --num_classes ${NUM_CLASSES} \
        --data ${DATA_DIR} \
        --save_dir ${CKPT_DIR} \
        --gpu ${TRAIN_GPU} \
        --epochs ${TRAIN_EPOCHS} \
        --lr ${TRAIN_LR} \
        --batch_size ${TRAIN_BS} \
        --decreasing_lr ${TRAIN_DECREASING_LR} \
        --seed ${TRAIN_SEED} \
        --print_freq ${PRINT_FREQ} \
        2>&1 | tee "${CKPT_DIR}/train.log"

    if [ ! -f "${MODEL_PATH}" ]; then
        echo "ERROR: training finished but ${MODEL_PATH} was not produced."
        exit 1
    fi
    echo "[stage1] Base checkpoint saved → ${MODEL_PATH}"
}

# ─────────────────────────────────────────────────────────────────
# Stage 2: Per-seed class-forget worker (runs in background)
# ─────────────────────────────────────────────────────────────────
run_seed() {
    local SEED=$1
    local GPU=$2
    local EPOCHS_FOR_SEED=${3:-${UNLEARN_EPOCHS}}

    local SAVE_DIR="${RESULTS_ROOT}/${DATASET}/${FORGET_TAG}/output/seed${SEED}"
    local LOG_FILE="${SAVE_DIR}/run.log"

    mkdir -p "${SAVE_DIR}"

    echo "[seed=${SEED} gpu=${GPU}] Starting class-${CLASS_TO_FORGET} unlearning → log: ${LOG_FILE}"

    {
        echo "====== UNLEARNING (seed=${SEED}, gpu=${GPU}, class=${CLASS_TO_FORGET}) ======"
        CMD="python main_forget.py \
            --unlearn MUKSB \
            --unlearn_epochs ${EPOCHS_FOR_SEED} \
            --unlearn_lr ${UNLEARN_LR} \
            --mask ${MODEL_PATH} \
            --save_dir ${SAVE_DIR} \
            --dataset ${DATASET} \
            --arch ${ARCH} \
            --num_classes ${NUM_CLASSES} \
            --gpu ${GPU} \
            --class_to_replace ${CLASS_TO_FORGET} \
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
}

run_unlearning() {
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "ERROR: base checkpoint not found at ${MODEL_PATH}."
        echo "       Run Stage 1 first:  STAGE=train bash run_pipeline_svhn.sh"
        exit 1
    fi

    echo "=========================================="
    echo "Stage 2: MUKSB Class-wise Unlearning — ${#SEEDS[@]} seed(s)"
    echo "  Dataset : ${DATASET} | Arch: ${ARCH}"
    echo "  Forget  : class ${CLASS_TO_FORGET} (all samples of digit ${CLASS_TO_FORGET})"
    echo "  LR=${UNLEARN_LR}  gamma=${GAMMA}  bs=${BATCH_SIZE}"
    echo "  Seeds   : ${SEEDS[*]}"
    echo "  GPUs    : ${GPUS[*]}"
    echo "  Epochs  : ${EPOCHS[*]}"
    echo "=========================================="

    PIDS=()
    for i in "${!SEEDS[@]}"; do
        run_seed "${SEEDS[$i]}" "${GPUS[$i]}" "${EPOCHS[$i]}" &
        PIDS+=($!)
    done

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
