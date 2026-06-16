#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  CIFAR-10 / ResNet-18 — 10% random forgetting
#  Runs 9 unlearning methods × N seeds with PER-METHOD tuned hyperparameters.
#
#  Methods : Retrain, FT, GA, IU, BE, ℓ1-sparse, SalUn, MUNBa, MUKSB (ours)
#
#  Shared mask:
#    SalUn, MUNBa and MUKSB all update the SAME subset of weights — one saliency
#    mask at density MASK_DENSITY (default 0.5), generated per seed and fed to all
#    three via --path. This makes their comparison like-for-like.
#
#  Parallelism:
#    Methods run in PARALLEL (one job per GPU, round-robin over GPUS); seeds run
#    SEQUENTIALLY (one full method-sweep finishes before the next seed starts).
#    Spread methods over GPUs with e.g. GPUS="1 2 3 4 5 6 7 8".
#
#  Everything (checkpoints / logs / results) is written under THIS folder:
#    cifar10_random_forget/
#      ├── checkpoints/<method>/seed<N>/   (model + <key>eval_result.pth.tar)
#      ├── logs/<method>_seed<N>.log
#      └── results/                         (filled by aggregate_results.py)
#
#  Usage:
#    bash run_all_methods.sh
#    nohup bash run_all_methods.sh > run_all.out 2>&1 &
#
#  Overrides (all optional, space-separated strings):
#    SEEDS="1 2 3"      bash run_all_methods.sh
#    GPUS="6 7"         bash run_all_methods.sh        # round-robin over seeds
#    METHODS="FT MUKSB" bash run_all_methods.sh        # subset of methods
#    SKIP_DONE=false  GPUS="6 7"   bash run_all_methods.sh        # re-run finished jobs
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_DIR="$(cd "${HERE}/.." && pwd)"
cd "${CLS_DIR}"

# ── Fixed task setup ─────────────────────────────────────────────────────────
DATASET="cifar10"
ARCH="resnet18"
NUM_CLASSES=10
DATA_DIR="/storage/s25017/Datasets/CIFAR10"
MODEL_PATH="${CLS_DIR}/checkpoints/resnet18_cifar10/0checkpoint.pth.tar"
CLASS_TO_REPLACE=-1
NUM_INDEXES=5000           # 10% of 50,000 CIFAR-10 training samples
PRINT_FREQ=50
NUM_WORKERS=4

# ── Output dirs (self-contained under this folder) ───────────────────────────
CKPT_ROOT="${HERE}/checkpoints"
LOG_ROOT="${HERE}/logs"
RESULTS_DIR="${HERE}/results"
MASK_ROOT="${HERE}/masks"
mkdir -p "${CKPT_ROOT}" "${LOG_ROOT}" "${RESULTS_DIR}" "${MASK_ROOT}"

# ── Shared saliency mask (fair comparison) ───────────────────────────────────
# SalUn, MUNBa and MUKSB all update the SAME subset of weights, defined by one
# gradient-saliency mask at this density (top-MASK_DENSITY by |∇L_forget|).
# One mask is generated per seed and fed to all three via --path.
MASK_DENSITY="${MASK_DENSITY:-0.5}"
MASKED_METHODS=(SalUn MUNBa MUKSB)

# ── Seeds & GPUs ─────────────────────────────────────────────────────────────
# Seeds run sequentially; within a seed the methods run in parallel, each on a
# DIFFERENT GPU (round-robin over GPUS). Default spreads over all 8 GPUs.
# Restrict to specific GPUs, e.g. the idle ones: GPUS="4 5 1 7".
read -ra SEEDS <<< "${SEEDS:-1}"
read -ra GPUS  <<< "${GPUS:-0 1 2 3 4 5 6 7}"

# ── Methods to run ───────────────────────────────────────────────────────────
# Selection priority:  positional args  >  METHODS env var  >  all methods.
#   bash run_all_methods.sh FT MUKSB            # only FT and MUKSB
#   bash run_all_methods.sh MUKSB               # only MUKSB
#   METHODS="FT MUKSB" bash run_all_methods.sh  # same, via env var
# ALL_METHODS=(retrain FT GA IU BE l1sparse SalUn MUNBa MUKSB)
ALL_METHODS=(SalUn MUNBa MUKSB)

case "${1:-}" in
    -h|--help|help)
        echo "Usage: bash run_all_methods.sh [METHOD ...]"
        echo "  Methods : ${ALL_METHODS[*]}"
        echo "  Env vars: SEEDS, GPUS, METHODS, SKIP_DONE"
        echo "  Examples:"
        echo "    bash run_all_methods.sh                    # all methods"
        echo "    bash run_all_methods.sh MUKSB              # only MUKSB"
        echo "    bash run_all_methods.sh FT GA MUKSB        # a subset"
        echo "    SEEDS=\"1 2 3\" bash run_all_methods.sh MUKSB"
        exit 0
        ;;
esac

if [ "$#" -gt 0 ]; then
    METHODS=("$@")                          # positional args take priority
elif [ -n "${METHODS:-}" ]; then
    read -ra METHODS <<< "${METHODS}"       # then the METHODS env var
else
    METHODS=("${ALL_METHODS[@]}")           # default: everything
fi

# Normalize names case-insensitively to canonical keys (e.g. "salun" → "SalUn").
_NORM=()
for M in "${METHODS[@]}"; do
    canon=""
    for K in "${ALL_METHODS[@]}"; do
        if [ "${M,,}" = "${K,,}" ]; then canon="${K}"; break; fi
    done
    _NORM+=("${canon:-$M}")                  # keep original if no match → validation errors
done
METHODS=("${_NORM[@]}")

SKIP_DONE="${SKIP_DONE:-true}"

# ─────────────────────────────────────────────────────────────────────────────
#  Per-method config.  Edit freely — this is the "tuned" table.
#  Fields:  <unlearn_key>|<lr>|<epochs>|<batch>|<decreasing_lr>|<extra flags>
#
#  Provenance of the defaults:
#    retrain / MUKSB / MUNBa  → repo scripts (run_pipeline.sh, classwise pipeline)
#    FT / GA / IU / BE / ℓ1   → standard unlearning-benchmark values; review & tune.
# ─────────────────────────────────────────────────────────────────────────────
declare -A CFG
#                  key                 lr      ep   bs   dec_lr   extra
CFG[retrain]="retrain|0.1|160|256|80,120|"                        # oracle: train from scratch on retain set
CFG[FT]="FT|0.01|10|256|91,136|"                                  # fine-tune on retain
CFG[GA]="GA|0.0001|5|256|91,136|"                                 # gradient ascent on forget (small lr)
CFG[IU]="IU|0.01|1|256|91,136|--iu_damping 1e-3 --iu_scale 1.0"   # one-shot influence (epochs unused)
CFG[BE]="boundary_expanding|0.0001|10|256|91,136|"               # boundary expanding
CFG[l1sparse]="FT_l1|0.03|10|256|91,136|--with_l1 --alpha 5e-4"   # ℓ1-sparse fine-tuning
CFG[SalUn]="SalUn|0.03|10|256|91,136|--salun_density 0.5"         # saliency unlearning
CFG[MUNBa]="MUNBa|0.03|10|256|91,136|--beta 1.0"                  # Nash bargaining baseline
CFG[MUKSB]="MUKSB|0.03|10|256|91,136|--gamma 0.5 --alpha 0.2"     # ours (KS bargaining)

# ── Validate requested methods ───────────────────────────────────────────────
for M in "${METHODS[@]}"; do
    if [ -z "${CFG[$M]+x}" ]; then
        echo "ERROR: unknown method '${M}'. Valid methods: ${ALL_METHODS[*]}"
        exit 1
    fi
done

# ── Validate checkpoint ──────────────────────────────────────────────────────
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: pretrained checkpoint not found: ${MODEL_PATH}"
    exit 1
fi

echo "############################################################"
echo "#  CIFAR-10 / ResNet-18 — 10% random forgetting (${NUM_INDEXES} samples)"
echo "#  Methods : ${METHODS[*]}"
echo "#  Seeds   : ${SEEDS[*]}"
echo "#  GPUs    : ${GPUS[*]}"
echo "#  Ckpt    : ${MODEL_PATH}"
echo "#  Output  : ${HERE}"
echo "############################################################"

# ─────────────────────────────────────────────────────────────────────────────
#  Single (method, seed) run
# ─────────────────────────────────────────────────────────────────────────────
run_method() {
    local METHOD=$1 SEED=$2 GPU=$3
    IFS='|' read -r KEY LR EP BS DLR EXTRA <<< "${CFG[$METHOD]}"

    local SAVE_DIR="${CKPT_ROOT}/${METHOD}/seed${SEED}"
    local LOG_FILE="${LOG_ROOT}/${METHOD}_seed${SEED}.log"
    local RES_RUN_DIR="${RESULTS_DIR}/${METHOD}/seed${SEED}"
    mkdir -p "${SAVE_DIR}" "${RES_RUN_DIR}"
    local rc=0

    # Masked methods (SalUn/MUNBa/MUKSB) share the per-seed saliency mask.
    local MASK_FLAG=""
    for MM in "${MASKED_METHODS[@]}"; do
        if [ "${METHOD}" = "${MM}" ]; then
            MASK_FLAG="--path ${MASK_ROOT}/seed${SEED}/with_${MASK_DENSITY}.pt"
        fi
    done

    if [ "${SKIP_DONE}" = "true" ] && ls "${SAVE_DIR}/"*eval_result.pth.tar >/dev/null 2>&1; then
        echo "[skip] method=${METHOD} seed=${SEED} (eval_result already exists)"
    else
        echo "[run ] method=${METHOD} key=${KEY} seed=${SEED} gpu=${GPU} lr=${LR} ep=${EP} bs=${BS} ${MASK_FLAG:+mask=${MASK_DENSITY}} → ${LOG_FILE}"
        {
            echo "====== ${METHOD} (key=${KEY}) | seed=${SEED} | gpu=${GPU} | lr=${LR} | epochs=${EP} | bs=${BS} | ${MASK_FLAG:-no-mask} ======"
            python main_random.py \
                --unlearn ${KEY} \
                --unlearn_lr ${LR} \
                --unlearn_epochs ${EP} \
                --batch_size ${BS} \
                --decreasing_lr ${DLR} \
                --mask ${MODEL_PATH} \
                --save_dir ${SAVE_DIR} \
                --dataset ${DATASET} \
                --arch ${ARCH} \
                --num_classes ${NUM_CLASSES} \
                --gpu ${GPU} \
                --class_to_replace ${CLASS_TO_REPLACE} \
                --num_indexes_to_replace ${NUM_INDEXES} \
                --seed ${SEED} \
                --print_freq ${PRINT_FREQ} \
                --workers ${NUM_WORKERS} \
                --data ${DATA_DIR} \
                ${MASK_FLAG} \
                ${EXTRA}
            rc=$?
            echo "====== rc=${rc} (${METHOD} | seed=${SEED}) ======"
            exit ${rc}
        } 2>&1 | tee "${LOG_FILE}"
        rc=${PIPESTATUS[0]}
    fi

    # Mirror the per-epoch status JSON (MUKSB-style epoch_metrics) into results/.
    if [ -f "${SAVE_DIR}/epoch_metrics.json" ]; then
        cp -f "${SAVE_DIR}/epoch_metrics.json" "${RES_RUN_DIR}/epoch_metrics.json"
    fi
    return ${rc}
}

# ─────────────────────────────────────────────────────────────────────────────
#  Ensure the shared saliency mask exists for a seed (generate once, on demand).
#  Only runs if a masked method (SalUn/MUNBa/MUKSB) is in the requested set.
# ─────────────────────────────────────────────────────────────────────────────
ensure_mask() {
    local SEED=$1 GPU=$2
    local MDIR="${MASK_ROOT}/seed${SEED}"
    local MFILE="${MDIR}/with_${MASK_DENSITY}.pt"

    local need=false
    for M in "${METHODS[@]}"; do
        for MM in "${MASKED_METHODS[@]}"; do
            [ "${M}" = "${MM}" ] && need=true
        done
    done
    [ "${need}" = false ] && return 0

    if [ -f "${MFILE}" ]; then
        echo "[mask] seed=${SEED} reuse ${MFILE}"
        return 0
    fi

    mkdir -p "${MDIR}"
    echo "[mask] seed=${SEED} gpu=${GPU} — generating density-${MASK_DENSITY} saliency mask → ${LOG_ROOT}/mask_seed${SEED}.log"
    python generate_mask.py \
        --unlearn MUKSB \
        --mask ${MODEL_PATH} \
        --save_dir ${MDIR} \
        --dataset ${DATASET} \
        --arch ${ARCH} \
        --num_classes ${NUM_CLASSES} \
        --gpu ${GPU} \
        --class_to_replace ${CLASS_TO_REPLACE} \
        --num_indexes_to_replace ${NUM_INDEXES} \
        --seed ${SEED} \
        --batch_size 256 \
        --unlearn_lr 0.01 \
        --workers ${NUM_WORKERS} \
        --data ${DATA_DIR} \
        > "${LOG_ROOT}/mask_seed${SEED}.log" 2>&1

    if [ ! -f "${MFILE}" ]; then
        echo "[mask] ERROR: mask generation failed for seed=${SEED} (see ${LOG_ROOT}/mask_seed${SEED}.log)"
        return 1
    fi
    echo "[mask] seed=${SEED} done → ${MFILE}"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Per-seed worker: launch every requested method in PARALLEL, one job per GPU
#  (round-robin over GPUS), then wait for them all.
# ─────────────────────────────────────────────────────────────────────────────
run_seed_methods_parallel() {
    local SEED=$1
    local PIDS=() LABELS=() j=0 failed=0

    # Build the shared mask first (sequential) so masked methods can read it.
    ensure_mask "${SEED}" "${GPUS[0]}" || { echo "[mask] aborting seed=${SEED}"; return 1; }

    for M in "${METHODS[@]}"; do
        local GPU=${GPUS[$(( j % NUM_GPUS ))]}
        run_method "${M}" "${SEED}" "${GPU}" &
        PIDS+=($!)
        LABELS+=("method=${M} seed=${SEED} gpu=${GPU}")
        j=$((j + 1))
    done
    for k in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$k]}"; then
            echo "FAILED: ${LABELS[$k]}"
            failed=$((failed + 1))
        else
            echo "OK: ${LABELS[$k]}"
        fi
    done
    return ${failed}
}

# ─────────────────────────────────────────────────────────────────────────────
#  Launch: seeds SEQUENTIAL, methods PARALLEL within each seed
# ─────────────────────────────────────────────────────────────────────────────
NUM_GPUS=${#GPUS[@]}
FAILED=0
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "#### seed=${SEED} — launching ${#METHODS[@]} methods in parallel over GPUs: ${GPUS[*]} ####"
    run_seed_methods_parallel "${SEED}"
    FAILED=$((FAILED + $?))
    echo "<<< seed=${SEED} done"
done

echo "############################################################"
if [ "${FAILED}" -eq 0 ]; then
    echo "#  All (method × seed) runs finished without error."
else
    echo "#  ${FAILED} run(s) reported failures — check logs in ${LOG_ROOT}/"
fi
echo "#  Aggregate results with:"
echo "#    python ${HERE}/aggregate_results.py --root ${CKPT_ROOT} --out ${RESULTS_DIR}"
echo "############################################################"
