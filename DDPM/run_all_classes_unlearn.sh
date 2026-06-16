#!/usr/bin/env bash
# Unlearn each CIFAR-10 class (0-9) one at a time from the base checkpoint,
# then sample ALL 10 classes from each unlearned model in separate folders.
#
# Usage (from DDPM/):
#   bash run_all_classes_unlearn.sh
#
# Output layout:
#   results/cifar10/forget/<METHOD>/<ALPHA>_no_mask/class<N>/<timestamp>/
#     ckpts/ckpt.pth
#     forget_info.txt
#     class<N>_forget/
#       class0_images/   <- N_SAMPLES individual images generated for class 0
#       canvas_class0.png
#       class1_images/
#       canvas_class1.png
#       ...
#       class9_images/
#       canvas_class9.png
set -euo pipefail

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GPU="6"
BASE_CKPT_FOLDER="results/cifar10/2026_05_13_232008"   # trained to 180k steps

UNLEARN_CFG="cifar10_saliency_unlearn.yml"
SAMPLE_CFG="cifar10_sample.yml"

MODE="muksb_unlearn"    # muksb_unlearn | saliency_unlearn
METHOD="rl"             # rl | ga
GAMMA=0.5
ALPHA=0.001             # use decimal (not 1e-3) — Python formats float as 0.001 in folder names
USE_MASK=0              # set to 1 and fill MASK_DIR if you have saliency masks
MASK_DIR=""             # e.g. "results/cifar10/masks"

SEED=1234
N_SAMPLES=500           # images to sample from the forgotten class per run
COND_SCALE=2.0
# ───────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${GPU}"

# The unlearn runner saves to: results/cifar10/forget/<METHOD>/<ALPHA>_no_mask/class<N>/<timestamp>/
# ALPHA must be decimal (e.g. 0.001) to match Python's float formatting of the path.
UNLEARN_PARENT_BASE="results/cifar10/forget/${METHOD}/${ALPHA}_no_mask"

CIFAR10_CLASSES=(airplane automobile bird cat deer dog frog horse ship truck)

for LABEL in {0..9}; do
    CLASS_NAME="${CIFAR10_CLASSES[$LABEL]}"

    echo ""
    echo "================================================================"
    echo " Forgetting class ${LABEL} (${CLASS_NAME})"
    echo "================================================================"

    # ── 1. Unlearn ──────────────────────────────────────────────────────────
    UNLEARN_ARGS=(
        --config   "${UNLEARN_CFG}"
        --ckpt_folder "${BASE_CKPT_FOLDER}"
        --label_to_forget "${LABEL}"
        --mode     "${MODE}"
        --method   "${METHOD}"
        --alpha    "${ALPHA}"
        --gamma    "${GAMMA}"
        --seed     "${SEED}"
    )

    if [[ "${USE_MASK}" == "1" ]]; then
        MASK_PATH="${MASK_DIR}/mask_label_${LABEL}.pt"
        if [[ ! -f "${MASK_PATH}" ]]; then
            echo "  ERROR: USE_MASK=1 but mask not found at ${MASK_PATH}" >&2
            exit 1
        fi
        UNLEARN_ARGS+=(--mask_path "${MASK_PATH}")
    fi

    python train.py "${UNLEARN_ARGS[@]}"

    # Locate the freshest timestamp folder under class-specific parent dir
    UNLEARN_CLASS_PARENT="${UNLEARN_PARENT_BASE}/class${LABEL}"
    UNLEARN_CKPT_FOLDER=$(ls -td "${UNLEARN_CLASS_PARENT}"/*/  2>/dev/null | head -1 | sed 's|/$||')

    if [[ -z "${UNLEARN_CKPT_FOLDER}" || ! -f "${UNLEARN_CKPT_FOLDER}/ckpts/ckpt.pth" ]]; then
        echo "  ERROR: could not find unlearned checkpoint for class ${LABEL} under ${UNLEARN_CLASS_PARENT}" >&2
        exit 1
    fi

    # Write a metadata file so the folder is always identifiable
    cat > "${UNLEARN_CKPT_FOLDER}/forget_info.txt" <<EOF
label_to_forget: ${LABEL}
class_name:      ${CLASS_NAME}
mode:            ${MODE}
method:          ${METHOD}
gamma:           ${GAMMA}
alpha:           ${ALPHA}
base_ckpt:       ${BASE_CKPT_FOLDER}
seed:            ${SEED}
EOF
    echo "  Unlearned checkpoint: ${UNLEARN_CKPT_FOLDER}"

    # ── 2. Sample ALL 10 classes (each gets its own subfolder + canvas) ────────
    echo "  Sampling ${N_SAMPLES} images for ALL classes from forget-${LABEL} (${CLASS_NAME}) model..."
    python sample.py \
        --config "${SAMPLE_CFG}" \
        --ckpt_folder "${UNLEARN_CKPT_FOLDER}" \
        --mode sample_classes \
        --classes_to_generate "0,1,2,3,4,5,6,7,8,9" \
        --n_samples_per_class "${N_SAMPLES}" \
        --cond_scale "${COND_SCALE}" \
        --forget_label "${LABEL}" \
        --seed "${SEED}"

    echo "  Samples saved under: ${UNLEARN_CKPT_FOLDER}/class${LABEL}_forget/"
    for C in {0..9}; do
        echo "    class${C}_images/   canvas_class${C}.png"
    done
done

echo ""
echo "All 10 classes processed."
echo "Results under: ${UNLEARN_PARENT_BASE}/"
