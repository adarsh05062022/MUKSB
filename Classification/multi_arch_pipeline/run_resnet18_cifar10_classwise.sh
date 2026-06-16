#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Class-wise forgetting on ResNet-18 / CIFAR-10
# Forgets each of the 10 classes individually.
# One seed, two methods: MUKSB and FT.
#
# Classes run in parallel — one job per GPU (round-robin).
#
# Usage:
#   bash run_resnet18_cifar10_classwise.sh
#   nohup bash run_resnet18_cifar10_classwise.sh > classwise.out 2>&1 &
#
# Run only specific classes (space-separated string):
#   CLASSES="0 1 5" nohup bash run_resnet18_cifar10_classwise.sh > classwise.out 2>&1 &
#
# Use specific GPUs (space-separated string):
#   GPUS="1 4" nohup bash run_resnet18_cifar10_classwise.sh > classwise.out 2>&1 &
# ─────────────────────────────────────────────────────────────────

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CLS_DIR}"

# ── Dataset / model ───────────────────────────────────────────────
ARCH="resnet18"
DATASET="cifar10"
NUM_CLASSES=10
DATA_DIR="/storage/s25017/Datasets/CIFAR10"
MODEL_PATH="${CLS_DIR}/checkpoints/resnet18_cifar10/0model_SA_best.pth.tar"
RESULTS_ROOT="${CLS_DIR}/results_classwise"

# ── Unlearning hyperparams (MUKSB / FT) ───────────────────────────
UNLEARN_LR=0.03
UNLEARN_EPOCHS=30
GAMMA=0.5
ALPHA=0.2
BATCH_SIZE=256
DECREASING_LR="91,136"
EXTRA="lr_0_03"
WITH_L1=false
PRINT_FREQ=50
NUM_WORKERS=2

# ── Retrain hyperparams (match base-model training) ────────────────
RETRAIN_LR=0.03
RETRAIN_EPOCHS=160
RETRAIN_BATCH_SIZE=512

# ── Seed ─────────────────────────────────────────────────────────
SEED=1

# ── GPUs to use (round-robin across classes) ─────────────────────
# Override: GPUS="1 4" bash run_resnet18_cifar10_classwise.sh
if [ -n "${GPUS:-}" ]; then
    read -ra GPU_LIST <<< "${GPUS}"
else
    GPU_LIST=(4)
fi

# ── Methods ───────────────────────────────────────────────────────
METHODS=(RL MUNBa SalUn)

# ── Classes to forget (default: all 10) ──────────────────────────
# Override: CLASSES="0 1 2" bash run_resnet18_cifar10_classwise.sh
if [ -n "${CLASSES:-}" ]; then
    read -ra CLASSES <<< "${CLASSES}"
else
    CLASSES=($(seq 0 $((NUM_CLASSES - 1))))
fi

# ── Validate checkpoint ───────────────────────────────────────────
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: ResNet-18 CIFAR-10 checkpoint not found: ${MODEL_PATH}"
    exit 1
fi

NUM_GPUS=${#GPU_LIST[@]}

echo "############################################################"
echo "#  ResNet-18 / CIFAR-10 — class-wise forgetting (parallel)"
echo "#  Classes  : ${#CLASSES[@]} total (${CLASSES[0]} … ${CLASSES[-1]})"
echo "#  Methods  : ${METHODS[*]}"
echo "#  Seed     : ${SEED}"
echo "#  GPUs     : ${GPU_LIST[*]} (${NUM_GPUS} parallel workers)"
echo "#  MUKSB/FT : lr=${UNLEARN_LR}  epochs=${UNLEARN_EPOCHS}  bs=${BATCH_SIZE}"
echo "#  retrain  : lr=${RETRAIN_LR}  epochs=${RETRAIN_EPOCHS}  bs=${RETRAIN_BATCH_SIZE}"
echo "############################################################"

# ─────────────────────────────────────────────────────────────────
# Per-(method, class) worker
# ─────────────────────────────────────────────────────────────────
run_one() {
    local METHOD=$1
    local CLASS=$2
    local GPU=$3

    # Pick hyperparams based on method
    local M_LR M_EPOCHS M_BS
    if [ "${METHOD}" = "retrain" ]; then
        M_LR=${RETRAIN_LR}
        M_EPOCHS=${RETRAIN_EPOCHS}
        M_BS=${RETRAIN_BATCH_SIZE}
    else
        M_LR=${UNLEARN_LR}
        M_EPOCHS=${UNLEARN_EPOCHS}
        M_BS=${BATCH_SIZE}
    fi

    local FORGET_TAG="class${CLASS}_bs${M_BS}_${EXTRA}"
    local SAVE_DIR="${RESULTS_ROOT}/${DATASET}/${ARCH}/${METHOD}/${FORGET_TAG}/seed${SEED}"
    local LOG_FILE="${SAVE_DIR}/run.log"

    mkdir -p "${SAVE_DIR}"

    echo "[method=${METHOD} class=${CLASS} seed=${SEED} gpu=${GPU} lr=${M_LR} epochs=${M_EPOCHS}] log: ${LOG_FILE}"

    {
        echo "====== ${METHOD} | class=${CLASS} | seed=${SEED} | gpu=${GPU} | lr=${M_LR} | epochs=${M_EPOCHS} ======"
        CMD="python main_forget.py \
            --unlearn ${METHOD} \
            --unlearn_epochs ${M_EPOCHS} \
            --unlearn_lr ${M_LR} \
            --mask ${MODEL_PATH} \
            --save_dir ${SAVE_DIR} \
            --dataset ${DATASET} \
            --arch ${ARCH} \
            --num_classes ${NUM_CLASSES} \
            --gpu ${GPU} \
            --class_to_replace ${CLASS} \
            --gamma ${GAMMA} \
            --alpha ${ALPHA} \
            --batch_size ${M_BS} \
            --seed ${SEED} \
            --print_freq ${PRINT_FREQ} \
            --decreasing_lr ${DECREASING_LR} \
            --data ${DATA_DIR} \
            --workers ${NUM_WORKERS}"

        if [ "${WITH_L1}" = "true" ]; then
            CMD="${CMD} --with_l1"
        fi

        eval ${CMD}
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "ERROR: ${METHOD} failed for class=${CLASS} (rc=${rc})"
            exit $rc
        fi
        echo "====== DONE (${METHOD} | class=${CLASS} | seed=${SEED}) ======"
        echo "  Output: ${SAVE_DIR}/"
    } 2>&1 | tee "${LOG_FILE}"
}

# ─────────────────────────────────────────────────────────────────
# Launch: classes sequentially, methods in parallel per class
# ─────────────────────────────────────────────────────────────────
OVERALL_FAILED=0
TOTAL=${#CLASSES[@]}
CLASS_IDX=0

for CLASS in "${CLASSES[@]}"; do
    CLASS_IDX=$((CLASS_IDX + 1))
    echo ""
    echo "############################################################"
    echo "#  Class ${CLASS}  (${CLASS_IDX}/${TOTAL}) — launching ${#METHODS[@]} methods in parallel"
    echo "############################################################"

    PIDS=()
    LABELS=()
    JOB_IDX=0

    for METHOD in "${METHODS[@]}"; do
        GPU=${GPU_LIST[$((JOB_IDX % NUM_GPUS))]}
        run_one "${METHOD}" "${CLASS}" "${GPU}" &
        PIDS+=($!)
        LABELS+=("method=${METHOD} class=${CLASS} gpu=${GPU}")
        JOB_IDX=$((JOB_IDX + 1))
    done

    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "FAILED: ${LABELS[$i]} (rc=${rc})"
            OVERALL_FAILED=$((OVERALL_FAILED + 1))
        else
            echo "OK: ${LABELS[$i]}"
        fi
    done
    echo "<<< Class ${CLASS} done"
done

echo ""
echo "############################################################"
echo "#  Class-wise sweep complete"
echo "############################################################"
if [ "${OVERALL_FAILED}" -eq 0 ]; then
    echo "All (method × class) runs completed successfully."
else
    echo "${OVERALL_FAILED} run(s) had failures. Check logs under:"
    echo "  ${RESULTS_ROOT}/${DATASET}/${ARCH}/"
fi
echo "Results root: ${RESULTS_ROOT}/${DATASET}/${ARCH}/"
exit ${OVERALL_FAILED}
