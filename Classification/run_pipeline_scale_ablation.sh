#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Ablation 2: Step-size magnitude (fix KS direction, vary scale) — CLASS-WISE
#
# All four arms use the identical KS bisector direction (g_star).
# Only the effective_scale multiplier changes:
#
#   MUKSB (full)  — harmonic mean: 2||gr||||gf||/(||gr||+||gf||)
#   Variant C     — arithmetic mean: (||gr||+||gf||)/2
#   Variant D     — minimum norm: min(||gr||, ||gf||)
#   Variant E     — fixed scalar: 1.0  (direction-only, no adaptive scaling)
#
# Forgetting mode: class-wise (one full class per run, 10 classes total).
# Classes run sequentially; all 4 methods launch in parallel per class.
#
# Usage:
#   bash run_pipeline_scale_ablation.sh
#
# Run specific classes only:
#   CLASSES="0 1 2" bash run_pipeline_scale_ablation.sh
#
# Use specific GPUs:
#   GPUS="1 4 5 6" bash run_pipeline_scale_ablation.sh
#
# Skip re-running full MUKSB (if results already exist for a class):
#   SKIP_FULL=true bash run_pipeline_scale_ablation.sh
# ─────────────────────────────────────────────────────────────────────────────

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SKIP_FULL=${SKIP_FULL:-false}

# ── Seed / Epochs ─────────────────────────────────────────────────────────────
SEED=1
EPOCHS=30

# ── GPU assignment (one GPU per variant, all run in parallel per class) ────────
# Override with: GPUS="0 1 2 3" bash run_pipeline_scale_ablation.sh
if [ -n "${GPUS:-}" ]; then
    read -ra GPU_LIST <<< "${GPUS}"
else
    GPU_LIST=(1 5 6 4)   # Full  ArithMean  MinNorm  Fixed
fi
GPU_FULL=${GPU_LIST[0]}
GPU_VARC=${GPU_LIST[1]}
GPU_VARD=${GPU_LIST[2]}
GPU_VARE=${GPU_LIST[3]}

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH="${SCRIPT_DIR}/checkpoints/resnet18_cifar10/0model_SA_best.pth.tar"
DATA_DIR="/storage/s25017/Datasets/CIFAR10"
RESULTS_ROOT="/scratch/s25017/MUKSB/Classification/results_ablation_direction"

# ── Dataset / Architecture ────────────────────────────────────────────────────
DATASET="cifar10"
ARCH="resnet18"
NUM_CLASSES=10

# ── Classes to forget (default: all 10) ──────────────────────────────────────
if [ -n "${CLASSES:-}" ]; then
    read -ra CLASS_LIST <<< "${CLASSES}"
else
    CLASS_LIST=($(seq 0 $((NUM_CLASSES - 1))))
fi

# ── Unlearning hyperparams ────────────────────────────────────────────────────
UNLEARN_LR=0.03
GAMMA=0.5
WITH_L1=false
ALPHA=0.2
EXTRA="lr_0_03"

# ── Misc ──────────────────────────────────────────────────────────────────────
BATCH_SIZE=256
PRINT_FREQ=50
DECREASING_LR="91,136"
NUM_WORKERS=2

# ── Validate checkpoint ───────────────────────────────────────────────────────
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: checkpoint not found: ${MODEL_PATH}"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Per-(method, class) worker
# ─────────────────────────────────────────────────────────────────────────────
launch() {
    local METHOD=$1
    local CLASS=$2
    local GPU=$3

    local FORGET_TAG="class${CLASS}_bs${BATCH_SIZE}_${EXTRA}"
    local SAVE_DIR="${RESULTS_ROOT}/${DATASET}/${ARCH}/${METHOD}/${FORGET_TAG}/seed${SEED}"
    local LOG_FILE="${SAVE_DIR}/run.log"

    mkdir -p "${SAVE_DIR}"
    echo "[${METHOD} | class=${CLASS} | gpu=${GPU} | seed=${SEED}] → ${LOG_FILE}"

    {
        echo "====== ${METHOD} | class=${CLASS} | gpu=${GPU} | seed=${SEED} ======"
        CMD="python main_forget.py \
            --unlearn ${METHOD} \
            --unlearn_epochs ${EPOCHS} \
            --unlearn_lr ${UNLEARN_LR} \
            --mask ${MODEL_PATH} \
            --save_dir ${SAVE_DIR} \
            --dataset ${DATASET} \
            --arch ${ARCH} \
            --num_classes ${NUM_CLASSES} \
            --gpu ${GPU} \
            --class_to_replace ${CLASS} \
            --gamma ${GAMMA} \
            --alpha ${ALPHA} \
            --batch_size ${BATCH_SIZE} \
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
            echo "ERROR: ${METHOD} class=${CLASS} failed (rc=${rc})"
            exit $rc
        fi

        echo "====== DONE: ${METHOD} | class=${CLASS} ======"
        echo "  Metrics: ${SAVE_DIR}/epoch_metrics.json"

    } 2>&1 | tee "${LOG_FILE}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Scale Ablation — CIFAR-10 class-wise forgetting"
echo "  Arch    : ${ARCH}  |  LR=${UNLEARN_LR}  gamma=${GAMMA}"
echo "  Seed    : ${SEED}  |  Epochs: ${EPOCHS}  |  BS: ${BATCH_SIZE}"
echo "  Classes : ${CLASS_LIST[*]}"
echo "  GPUs    : Full=${GPU_FULL}  C=${GPU_VARC}  D=${GPU_VARD}  E=${GPU_VARE}"
echo "  Results : ${RESULTS_ROOT}/${DATASET}/${ARCH}/"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Classes sequentially — all 4 methods in parallel per class
# ─────────────────────────────────────────────────────────────────────────────
OVERALL_FAILED=0
TOTAL=${#CLASS_LIST[@]}
CLASS_IDX=0

for CLASS in "${CLASS_LIST[@]}"; do
    CLASS_IDX=$((CLASS_IDX + 1))
    echo ""
    echo "============================================================"
    echo "  Class ${CLASS}  (${CLASS_IDX}/${TOTAL}) — launching 4 variants in parallel"
    echo "============================================================"

    PIDS=()
    LABELS=()

    if [ "${SKIP_FULL}" != "true" ]; then
        launch "MUKSB"           "${CLASS}" "${GPU_FULL}" &
        PIDS+=($!)  LABELS+=("MUKSB/class${CLASS}")
    else
        echo "[SKIP] MUKSB (full) — SKIP_FULL=true"
    fi

    launch "MUKSB_ArithMean" "${CLASS}" "${GPU_VARC}" &
    PIDS+=($!)  LABELS+=("MUKSB_ArithMean/class${CLASS}")

    launch "MUKSB_MinNorm"   "${CLASS}" "${GPU_VARD}" &
    PIDS+=($!)  LABELS+=("MUKSB_MinNorm/class${CLASS}")

    launch "MUKSB_Fixed"     "${CLASS}" "${GPU_VARE}" &
    PIDS+=($!)  LABELS+=("MUKSB_Fixed/class${CLASS}")

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
echo "============================================================"
if [ "${OVERALL_FAILED}" -eq 0 ]; then
    echo "All variants completed successfully."
else
    echo "${OVERALL_FAILED} run(s) failed. Check logs under:"
    echo "  ${RESULTS_ROOT}/${DATASET}/${ARCH}/"
fi
echo "Results: ${RESULTS_ROOT}/${DATASET}/${ARCH}/"
echo "============================================================"
exit ${OVERALL_FAILED}
