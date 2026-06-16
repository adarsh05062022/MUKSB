#!/usr/bin/env bash
# Run paper_eval.py for every forget-class experiment found under FORGET_ROOT.
# Assumes one sub-folder per forgotten class, named with timestamps as produced
# by run_all_classes_unlearn.sh.
#
# Usage (from DDPM/):
#   bash run_paper_eval.sh
set -euo pipefail

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GPU="0"

# Root that contains one timestamped folder per forgotten class
FORGET_ROOT="results/cifar10/forget/rl/0.001_no_mask"

# Base model class_samples (for before/after comparison in grid + charts)
# Set to "" if you haven't sampled the base model yet
BASE_SAMPLES="results/cifar10/2026_05_13_232008/class_samples"

CLF_CKPT="cifar10_resnet34.pth"
N_CLF_SAMPLES=500
# ───────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${GPU}"

CIFAR10_CLASSES=(airplane automobile bird cat deer dog frog horse ship truck)

# Collect all timestamped folders sorted oldest→newest (matches unlearn order 0→9)
mapfile -t FOLDERS < <(ls -td "${FORGET_ROOT}"/*/  2>/dev/null | tac | sed 's|/$||')

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
    echo "No folders found under ${FORGET_ROOT}" >&2
    exit 1
fi

echo "Found ${#FOLDERS[@]} experiment folder(s) under ${FORGET_ROOT}"

for IDX in "${!FOLDERS[@]}"; do
    FOLDER="${FOLDERS[$IDX]}"
    LABEL="${IDX}"
    CLASS_NAME="${CIFAR10_CLASSES[$LABEL]}"

    SAMPLE_DIR="${FOLDER}/class_samples"
    OUT_DIR="${FOLDER}/paper_figs"

    if [[ ! -d "${SAMPLE_DIR}/${LABEL}" ]]; then
        echo "  SKIP class ${LABEL} (${CLASS_NAME}): no samples at ${SAMPLE_DIR}/${LABEL}"
        continue
    fi

    echo ""
    echo "================================================================"
    echo " paper_eval: forget class ${LABEL} (${CLASS_NAME})"
    echo " folder: ${FOLDER}"
    echo "================================================================"

    EVAL_ARGS=(
        --forget_samples  "${SAMPLE_DIR}"
        --label_to_forget "${LABEL}"
        --clf_ckpt        "${CLF_CKPT}"
        --output_dir      "${OUT_DIR}"
        --n_clf_samples   "${N_CLF_SAMPLES}"
    )

    if [[ -n "${BASE_SAMPLES}" && -d "${BASE_SAMPLES}/${LABEL}" ]]; then
        EVAL_ARGS+=(--base_samples "${BASE_SAMPLES}")
    fi

    python paper_eval.py "${EVAL_ARGS[@]}"

    echo "  Figures saved: ${OUT_DIR}/"
done

echo ""
echo "All evaluations complete."
echo "Figures are under: ${FORGET_ROOT}/<timestamp>/paper_figs/"
