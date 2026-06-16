#!/usr/bin/env bash
# Driver for the DDPM SalUn / MUKSB pipeline.
#
# Edit the variables in the CONFIG block below, then run:
#     bash run_muksb.sh
#
# STAGE selects which step to execute. Run them in order the first time:
#   train         -> train base conditional DDPM
#   generate_mask -> compute SalUn saliency mask (only needed if USE_MASK=1)
#   unlearn       -> run muksb_unlearn (or saliency_unlearn) on the base ckpt
#   sample_fid    -> generate samples for FID/IS on remaining classes
#   save_ref      -> dump reference dataset minus the forgotten class
#   eval_fid      -> run evaluator.py (FID/IS/Precision/Recall)
#   train_clf     -> fine-tune ResNet34 classifier
#   sample_class  -> generate samples for the forgotten class
#   eval_clf      -> classifier-based forget evaluation
#   all           -> run the full unlearn+eval chain (assumes base ckpt exists)
set -euo pipefail

# ─── CONFIG ────────────────────────────────────────────────────────────────────
STAGE="unlearn"                 # see options above
GPUS="0"                        # e.g. "0" or "0,1"

# Dataset / configs
DATASET="cifar10"               # cifar10 | stl10
TRAIN_CFG="cifar10_train.yml"
UNLEARN_CFG="cifar10_saliency_unlearn.yml"
SAMPLE_CFG="cifar10_sample.yml"

# Checkpoints
BASE_CKPT_FOLDER="results/cifar10/yyyy_mm_dd_hhmmss"        # base DDPM ckpt
UNLEARN_CKPT_FOLDER="results/cifar10/unlearn/rl/yyyy_mm_dd_hhmmss"  # produced by unlearn stage

# Forgetting target
LABEL_TO_FORGET=0               # 0-9

# Unlearning hyperparameters
MODE="muksb_unlearn"            # muksb_unlearn | saliency_unlearn
METHOD="rl"                     # rl | ga
GAMMA=0.5                       # MUKSB retain priority (0..1); 0.5 = symmetric
ALPHA=1e-3                      # remain-loss weight (SalUn) / KS remain weight
USE_MASK=0                      # 0 = no mask, 1 = use SalUn mask
MASK_PATH=""                    # required if USE_MASK=1
MASK_RATIO=0.5                  # only used by generate_mask stage
SEED=1234

# Sampling
N_SAMPLES_PER_CLASS=5000        # for FID; use 500 for classifier eval
CLASSES_TO_GENERATE="x0"        # 'x0' = all except 0; or e.g. "1,2,3" or "0"
COND_SCALE=2.0

# Evaluation
REF_DIR="cifar10_without_label_0"
FID_SAMPLES_DIR="${UNLEARN_CKPT_FOLDER}/fid_samples_without_label_${LABEL_TO_FORGET}_guidance_${COND_SCALE}"
CLASS_SAMPLES_DIR="${UNLEARN_CKPT_FOLDER}/class_samples/${LABEL_TO_FORGET}"
CLF_CKPT="${DATASET}_resnet34.pth"
# ───────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${GPUS}"

run_train() {
    python train.py --config "${TRAIN_CFG}" --mode train --seed "${SEED}"
}

run_generate_mask() {
    python train.py \
        --config "${UNLEARN_CFG}" \
        --ckpt_folder "${BASE_CKPT_FOLDER}" \
        --label_to_forget "${LABEL_TO_FORGET}" \
        --mode generate_mask \
        --mask_ratio "${MASK_RATIO}" \
        --seed "${SEED}"
}

run_unlearn() {
    local args=(
        --config "${UNLEARN_CFG}"
        --ckpt_folder "${BASE_CKPT_FOLDER}"
        --label_to_forget "${LABEL_TO_FORGET}"
        --mode "${MODE}"
        --method "${METHOD}"
        --alpha "${ALPHA}"
        --seed "${SEED}"
    )
    if [[ "${MODE}" == "muksb_unlearn" ]]; then
        args+=(--gamma "${GAMMA}")
    fi
    if [[ "${USE_MASK}" == "1" ]]; then
        if [[ -z "${MASK_PATH}" ]]; then
            echo "USE_MASK=1 but MASK_PATH is empty" >&2
            exit 1
        fi
        args+=(--mask_path "${MASK_PATH}")
    fi
    python train.py "${args[@]}"
}

run_sample_fid() {
    python sample.py \
        --config "${SAMPLE_CFG}" \
        --ckpt_folder "${UNLEARN_CKPT_FOLDER}" \
        --mode sample_fid \
        --n_samples_per_class "${N_SAMPLES_PER_CLASS}" \
        --classes_to_generate "${CLASSES_TO_GENERATE}" \
        --cond_scale "${COND_SCALE}"
}

run_save_ref() {
    python save_base_dataset.py \
        --dataset "${DATASET}" \
        --label_to_forget "${LABEL_TO_FORGET}"
}

run_eval_fid() {
    python evaluator.py "${FID_SAMPLES_DIR}" "${REF_DIR}"
}

run_train_clf() {
    python train_classifier.py --dataset "${DATASET}"
}

run_sample_class() {
    python sample.py \
        --config "${SAMPLE_CFG}" \
        --ckpt_folder "${UNLEARN_CKPT_FOLDER}" \
        --mode sample_classes \
        --classes_to_generate "${LABEL_TO_FORGET}" \
        --n_samples_per_class "${N_SAMPLES_PER_CLASS}" \
        --cond_scale "${COND_SCALE}"
}

run_eval_clf() {
    python classifier_evaluation.py \
        --sample_path "${CLASS_SAMPLES_DIR}" \
        --dataset "${DATASET}" \
        --label_of_forgotten_class "${LABEL_TO_FORGET}"
}

case "${STAGE}" in
    train)         run_train ;;
    generate_mask) run_generate_mask ;;
    unlearn)       run_unlearn ;;
    sample_fid)    run_sample_fid ;;
    save_ref)      run_save_ref ;;
    eval_fid)      run_eval_fid ;;
    train_clf)     run_train_clf ;;
    sample_class)  run_sample_class ;;
    eval_clf)      run_eval_clf ;;
    all)
        run_unlearn
        run_sample_fid
        run_save_ref
        run_eval_fid
        run_sample_class
        run_eval_clf
        ;;
    *)
        echo "Unknown STAGE: ${STAGE}" >&2
        exit 1
        ;;
esac
