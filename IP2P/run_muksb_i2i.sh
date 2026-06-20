#!/usr/bin/env bash
# ============================================================================
# IP2P/run_muksb_i2i.sh
# MUKSB NSFW unlearning for InstructPix2Pix — all mask variants
#
# Runs one variant per GPU sequentially by default.  Override MASK_VARIANT
# to run a single variant, or launch multiple copies with different variants
# on separate GPUs.
#
# Usage
# -----
#   # All variants (default):
#   bash run_muksb_i2i.sh
#
#   # Single variant:
#   MASK_VARIANT=dual_fisher DEVICE=1 bash run_muksb_i2i.sh
#
#   # Smoke test (1 epoch, tiny):
#   SMOKE=1 bash run_muksb_i2i.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default to the active env's python (IP2P needs `diffusers`, which lives in
# the `munba3` env, not `salun`). Override with PY=/path/to/python if needed.
PY=${PY:-python}

FORGET_PATH=${FORGET_PATH:-/storage/s25017/Datasets/NSFW_removal/nude}
REMAIN_PATH=${REMAIN_PATH:-/storage/s25017/Datasets/NSFW_removal/with_dress}
CKPT=${CKPT:-timbrooks/instruct-pix2pix}

DEVICE=${DEVICE:-0}
EPOCHS=${EPOCHS:-5}
LR=${LR:-1e-5}
BATCH=${BATCH:-4}
MASK_DENSITY=${MASK_DENSITY:-0.5}
TRAIN_METHOD=${TRAIN_METHOD:-xattn}
BETA=${BETA:-5.0}

if [[ "${SMOKE:-0}" == "1" ]]; then
    EPOCHS=1
    echo "[smoke] Running 1 epoch smoke test"
fi

# ── Single-variant mode ───────────────────────────────────────────────────────
if [[ -n "${MASK_VARIANT:-}" ]]; then
    echo "============================================================"
    echo " MUKSB I2I — mask_variant=${MASK_VARIANT}  device=${DEVICE}"
    echo "============================================================"
    ${PY} MUKSB_nsfw_i2i.py \
        --mask_variant "${MASK_VARIANT}" \
        --mask_density "${MASK_DENSITY}" \
        --train_method "${TRAIN_METHOD}" \
        --beta         "${BETA}" \
        --epochs       "${EPOCHS}" \
        --lr           "${LR}" \
        --batch_size   "${BATCH}" \
        --ckpt_path    "${CKPT}" \
        --forget_path  "${FORGET_PATH}" \
        --remain_path  "${REMAIN_PATH}" \
        --device       "${DEVICE}"
    exit 0
fi

# ── All variants (sequential on DEVICE) ──────────────────────────────────────
for VARIANT in none dual_fisher salun forget_fisher random; do
    echo "============================================================"
    echo " MUKSB I2I — mask_variant=${VARIANT}  device=${DEVICE}"
    echo "============================================================"
    ${PY} MUKSB_nsfw_i2i.py \
        --mask_variant "${VARIANT}" \
        --mask_density "${MASK_DENSITY}" \
        --train_method "${TRAIN_METHOD}" \
        --beta         "${BETA}" \
        --epochs       "${EPOCHS}" \
        --lr           "${LR}" \
        --batch_size   "${BATCH}" \
        --ckpt_path    "${CKPT}" \
        --forget_path  "${FORGET_PATH}" \
        --remain_path  "${REMAIN_PATH}" \
        --device       "${DEVICE}"
done

echo "All MUKSB I2I variants complete."
