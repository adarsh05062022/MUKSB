#!/usr/bin/env bash
# For each class-specific unlearned checkpoint (class0..class9),
# generate images for ALL 10 classes one by one (one sample.py call per class).
#
# Usage (from DDPM/):
#   bash run_generate_classwise.sh
set -euo pipefail

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GPU="5"

METHOD="rl"
ALPHA="0.001"
SAMPLE_CFG="cifar10_sample.yml"

N_SAMPLES=10
COND_SCALE=2.0
SEED=1234
# ───────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${GPU}"

UNLEARN_PARENT_BASE="results/cifar10/forget/${METHOD}/${ALPHA}_no_mask"
CIFAR10_CLASSES=(airplane automobile bird cat deer dog frog horse ship truck)

for FORGET_LABEL in {0..9}; do
    FORGET_CLASS="${CIFAR10_CLASSES[$FORGET_LABEL]}"
    UNLEARN_CLASS_PARENT="${UNLEARN_PARENT_BASE}/class${FORGET_LABEL}"

    # Pick the most recently modified timestamped folder
    CKPT_FOLDER=$(ls -td "${UNLEARN_CLASS_PARENT}"/*/  2>/dev/null | head -1 | sed 's|/$||')

    if [[ -z "${CKPT_FOLDER}" || ! -f "${CKPT_FOLDER}/ckpts/ckpt.pth" ]]; then
        echo "SKIP forget_class=${FORGET_LABEL}: no checkpoint under ${UNLEARN_CLASS_PARENT}" >&2
        continue
    fi

    echo ""
    echo "================================================================"
    echo " Checkpoint: forget class ${FORGET_LABEL} (${FORGET_CLASS})"
    echo " Folder: ${CKPT_FOLDER}"
    echo "================================================================"

    for GEN_LABEL in {0..9}; do
        GEN_CLASS="${CIFAR10_CLASSES[$GEN_LABEL]}"
        echo "  -> Generating class ${GEN_LABEL} (${GEN_CLASS}) ..."

        python sample.py \
            --config          "${SAMPLE_CFG}" \
            --ckpt_folder     "${CKPT_FOLDER}" \
            --mode            sample_classes \
            --classes_to_generate "${GEN_LABEL}" \
            --n_samples_per_class "${N_SAMPLES}" \
            --cond_scale      "${COND_SCALE}" \
            --forget_label    "${FORGET_LABEL}" \
            --seed            "${SEED}"

        echo "     Done: class ${GEN_LABEL} (${GEN_CLASS})"
    done

    echo "  All 10 classes generated for forget_class=${FORGET_LABEL} (${FORGET_CLASS})"
done

echo ""
echo "All checkpoints processed."
echo "Results under: ${UNLEARN_PARENT_BASE}/"
