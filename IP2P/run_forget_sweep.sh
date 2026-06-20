#!/usr/bin/env bash
# ============================================================================
# IP2P/run_forget_sweep.sh
# Strengthen NSFW forgetting: train_method=full (whole UNet), no mask,
# at 5 and 10 epochs. Auto-generates a viz contact sheet for the FINAL
# epoch of each run so you can eyeball forget + retain together.
#
# Each EPOCHS value is an INDEPENDENT run trained from the base model
# (the 10-epoch run is NOT a continuation of the 5-epoch run).
#
# Usage
# -----
#   # default: full @ 5 and 10 epochs on GPU 0
#   bash run_forget_sweep.sh
#
#   # pick GPU / batch:
#   DEVICE=3 BATCH=2 bash run_forget_sweep.sh
#
#   # only one of them:
#   EPOCHS_LIST=10 DEVICE=3 bash run_forget_sweep.sh
#
#   # run the two in parallel on two GPUs (two terminals):
#   EPOCHS_LIST=5  DEVICE=2 bash run_forget_sweep.sh
#   EPOCHS_LIST=10 DEVICE=3 bash run_forget_sweep.sh
#
#   # push forget harder:
#   BETA=10 DEVICE=3 bash run_forget_sweep.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default to the active env's python (needs `diffusers` → munba3 env).
PY=${PY:-python}

DEVICE=${DEVICE:-0}
BATCH=${BATCH:-4}
LR=${LR:-1e-5}
BETA=${BETA:-5.0}
TRAIN_METHOD=${TRAIN_METHOD:-full}
EPOCHS_LIST=${EPOCHS_LIST:-"5 10"}

FORGET_PATH=${FORGET_PATH:-/storage/s25017/Datasets/NSFW_removal/nude}
REMAIN_PATH=${REMAIN_PATH:-/storage/s25017/Datasets/NSFW_removal/with_dress}
CKPT=${CKPT:-timbrooks/instruct-pix2pix}
VIZ_SRC=${VIZ_SRC:-/storage/s25017/Datasets/NSFW_removal/with_dress}
N_SOURCES=${N_SOURCES:-4}

run_one () {
    local EP=$1
    echo "============================================================"
    echo " TRAIN  method=${TRAIN_METHOD}  epochs=${EP}  beta=${BETA}  device=${DEVICE}"
    echo "============================================================"
    ${PY} MUKSB_nsfw_i2i.py \
        --mask_variant none \
        --train_method "${TRAIN_METHOD}" \
        --beta         "${BETA}" \
        --epochs       "${EP}" \
        --lr           "${LR}" \
        --batch_size   "${BATCH}" \
        --ckpt_path    "${CKPT}" \
        --forget_path  "${FORGET_PATH}" \
        --remain_path  "${REMAIN_PATH}" \
        --device       "${DEVICE}"

    # Locate the final-epoch checkpoint (tag includes lr / U<num_forget>).
    local DIR
    DIR=$(ls -d models/i2p-nsfw-MUKSB-i2i-method_${TRAIN_METHOD}-*E${EP}_U*/epoch_${EP} 2>/dev/null | head -1 || true)
    if [[ -n "${DIR}" && -f "${DIR}/model_index.json" ]]; then
        echo "============================================================"
        echo " VIZ    ${DIR}"
        echo "============================================================"
        ${PY} viz_check.py \
            --model_path "${DIR}" \
            --src_dir    "${VIZ_SRC}" \
            --device     "${DEVICE}" \
            --n_sources  "${N_SOURCES}" \
            --out_dir    "viz_${TRAIN_METHOD}_E${EP}"
        echo " contact sheet → viz_${TRAIN_METHOD}_E${EP}/contact_sheet.png"
    else
        echo "WARN: final checkpoint for E${EP} not found (looked for epoch_${EP}); skipping viz"
    fi
}

for EP in ${EPOCHS_LIST}; do
    run_one "${EP}"
done

echo "Sweep complete."
