#!/bin/bash
# run_table1.sh — Replicate Table 1 (CIFAR-10) from the MUNBa/MUKSB paper.
#
# Setup:
#   - Dataset  : CIFAR-10 (50 000 training samples)
#   - Forget   : 10% randomly selected samples = 5 000 samples (--num_indexes_to_replace 5000)
#   - Model    : ResNet-18
#   - Pretrained checkpoint: checkpoints/resnet18_cifar10/0checkpoint.pth.tar
#   - Epochs   : 10  |  LR : 0.01  |  Seed : 2
#
# Usage:
#   chmod +x run_table1.sh
#   ./run_table1.sh              # all methods, GPU 0
#   GPU=1 ./run_table1.sh        # different GPU
#   EPOCHS=5 ./run_table1.sh     # different epoch count
#   METHODS="GA FT MUNBa MUKSB" ./run_table1.sh   # subset only

set -euo pipefail

# ── Configurable via environment variables ────────────────────────────────────
GPU="${GPU:-0}"
SEED="${SEED:-2}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-0.01}"
BATCH="${BATCH:-256}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASK="${SCRIPT_DIR}/checkpoints/resnet18_cifar10/0checkpoint.pth.tar"
DATA="/storage/s25017/Datasets/CIFAR10"
OUT_CSV="${SCRIPT_DIR}/results/table1_cifar10_e${EPOCHS}.csv"
SAVE_ROOT="${SCRIPT_DIR}/sweep_checkpoints/table1_e${EPOCHS}"

# ── Validate checkpoint ───────────────────────────────────────────────────────
if [[ ! -f "${MASK}" ]]; then
    echo "[ERROR] Checkpoint not found: ${MASK}"
    exit 1
fi

echo "============================================================"
echo "  Table 1 replication — CIFAR-10"
echo "  Checkpoint : ${MASK}"
echo "  Epochs     : ${EPOCHS}  |  LR: ${LR}  |  Seed: ${SEED}"
echo "  GPU        : ${GPU}"
echo "  Output CSV : ${OUT_CSV}"
echo "============================================================"

mkdir -p "$(dirname "${OUT_CSV}")"
mkdir -p "${SAVE_ROOT}"

cd "${SCRIPT_DIR}"

# Build common python args
COMMON_ARGS=(
    python run_sweep.py
    --mask         "${MASK}"
    --dataset      cifar10
    --arch         resnet18
    --num_classes  10
    --data         "${DATA}"
    --class_to_replace       -1
    --num_indexes_to_replace 5000
    --gpu          "${GPU}"
    --seed         "${SEED}"
    --batch_size   "${BATCH}"
    --num_workers  4
    --unlearn_epochs  "${EPOCHS}"
    --unlearn_lr      "${LR}"
    --momentum        0.9
    --weight_decay    5e-4
    --decreasing_lr   "91,136"
    --alpha           0.2
    --save_root    "${SAVE_ROOT}"
    --out_csv      "${OUT_CSV}"
    --skip_done
)

# Append --methods if a subset was requested
if [[ -n "${METHODS:-}" ]]; then
    COMMON_ARGS+=(--methods ${METHODS})
fi

echo ""
echo "Running: ${COMMON_ARGS[*]}"
echo ""

"${COMMON_ARGS[@]}"

echo ""
echo "Done. Results saved to: ${OUT_CSV}"
