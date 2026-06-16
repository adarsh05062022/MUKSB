#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Ablation 1: update-direction strategy — CLASS-WISE forgetting
#
# Compares three gradient-merge strategies on CIFAR-10 class-wise
# forgetting:
#
#   MUKSB        — KS bisector of unit grads + harmonic-mean scaling (full)
#   MUKSB_RawSum — raw sum  gr + gf  (no unit normalisation)          [Var A]
#   MUKSB_MeanUnit — arithmetic mean of unit gradients                [Var B]
#
# Classes run SEQUENTIALLY; all 3 methods run in PARALLEL per class.
#
# Results land in:
#   results_ablation_direction/<method>/classwise/<dataset>/class<N>/seed<S>/
#
# Usage:
#   bash run_ablation_direction.sh
# ─────────────────────────────────────────────────────────────────

# ── Methods to compare ────────────────────────────────────────────
METHODS=("MUKSB"  "MUKSB_RawSum"  "MUKSB_MeanUnit")
# One dedicated GPU per method (index-aligned with METHODS)
METHOD_GPUS=(1     6               7)

# ── Classes to forget (run sequentially) ─────────────────────────
CLASS_LIST=(1 4)

# ── Single seed ───────────────────────────────────────────────────
SEED=1

# ── Epochs (same for all methods and classes) ─────────────────────
UNLEARN_EPOCHS=30

# ── Paths ─────────────────────────────────────────────────────────
MODEL_PATH="/scratch/s25017/MUKSB/Classification/checkpoints/resnet18_cifar10/0model_SA_best.pth.tar"
DATA_DIR="/storage/s25017/Datasets/CIFAR10"
RESULTS_ROOT="results_ablation_direction"

# ── Dataset / Architecture ────────────────────────────────────────
DATASET="cifar10"
ARCH="resnet18"
NUM_CLASSES=10

# ── Shared unlearning hyperparams ────────────────────────────────
UNLEARN_LR=0.017
GAMMA=0.5
WITH_L1=false
ALPHA=0.2

# ── Misc ──────────────────────────────────────────────────────────
BATCH_SIZE=512
PRINT_FREQ=50
DECREASING_LR="91,136"

# ─────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

# ─────────────────────────────────────────────────────────────────
# Per-(method, class) worker — runs in background
# ─────────────────────────────────────────────────────────────────
run_one() {
    local METHOD=$1
    local CLASS=$2
    local GPU=$3

    local SAVE_DIR="${RESULTS_ROOT}/${METHOD}/classwise/${DATASET}/class${CLASS}/seed${SEED}"
    local LOG_FILE="${SAVE_DIR}/run.log"

    mkdir -p "${SAVE_DIR}"

    echo "[${METHOD} | class=${CLASS} | gpu=${GPU} | seed=${SEED}] Starting → ${LOG_FILE}"

    # Build args as an array — avoids an extra bash subshell from { } | tee
    local ARGS=(
        --unlearn "${METHOD}"
        --unlearn_epochs "${UNLEARN_EPOCHS}"
        --unlearn_lr "${UNLEARN_LR}"
        --mask "${MODEL_PATH}"
        --save_dir "${SAVE_DIR}"
        --dataset "${DATASET}"
        --arch "${ARCH}"
        --num_classes "${NUM_CLASSES}"
        --gpu "${GPU}"
        --class_to_replace "${CLASS}"
        --gamma "${GAMMA}"
        --alpha "${ALPHA}"
        --batch_size "${BATCH_SIZE}"
        --seed "${SEED}"
        --print_freq "${PRINT_FREQ}"
        --decreasing_lr "${DECREASING_LR}"
        --data "${DATA_DIR}"
    )

    if [ "${WITH_L1}" = "true" ]; then
        ARGS+=(--with_l1)
    fi

    # python is the sole left-side command — bash execs it directly, no subshell
    python main_forget.py "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
}

# ─────────────────────────────────────────────────────────────────
# Print header
# ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Ablation 1: Update-Direction Strategy — Class-wise forgetting"
echo "  Dataset  : ${DATASET} | Arch: ${ARCH}"
echo "  Classes  : ${CLASS_LIST[*]} (sequential)"
echo "  Methods  : ${METHODS[*]} (parallel per class)"
echo "  Seed     : ${SEED}"
echo "  GPUs     : ${METHOD_GPUS[*]} (one per method)"
echo "  Epochs   : ${UNLEARN_EPOCHS} | LR: ${UNLEARN_LR} | γ: ${GAMMA}"
echo "  Results  : ${RESULTS_ROOT}/<method>/classwise/${DATASET}/class<N>/seed${SEED}/"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────
# Classes sequential — 3 methods parallel per class
# ─────────────────────────────────────────────────────────────────
OVERALL_FAILED=0
TOTAL=${#CLASS_LIST[@]}
CLASS_IDX=0

for CLASS in "${CLASS_LIST[@]}"; do
    CLASS_IDX=$((CLASS_IDX + 1))
    echo ""
    echo "============================================================"
    echo "  Class ${CLASS}  (${CLASS_IDX}/${TOTAL}) — launching ${#METHODS[@]} methods in parallel"
    echo "============================================================"

    PIDS=()
    LABELS=()

    for m_idx in "${!METHODS[@]}"; do
        METHOD="${METHODS[$m_idx]}"
        GPU="${METHOD_GPUS[$m_idx]}"

        run_one "${METHOD}" "${CLASS}" "${GPU}" &
        PIDS+=($!)
        LABELS+=("${METHOD}/class${CLASS}/gpu${GPU}")
    done

    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}"
        if [ $? -ne 0 ]; then
            echo "FAILED: ${LABELS[$i]}"
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
    echo "All jobs completed successfully."
else
    echo "${OVERALL_FAILED} job(s) failed — check log files in ${RESULTS_ROOT}/."
fi
echo ""
echo "Results layout:"
for CLASS in "${CLASS_LIST[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        echo "  ${RESULTS_ROOT}/${METHOD}/classwise/${DATASET}/class${CLASS}/seed${SEED}/"
    done
done
echo "============================================================"
exit ${OVERALL_FAILED}
