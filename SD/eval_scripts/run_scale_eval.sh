#!/usr/bin/env bash
# ============================================================================
# run_scale_eval.sh — wrapper around run_scale_eval.py with MULTI-GPU sharding
#
# Evaluates scale-ablation checkpoints end-to-end:
#   generate (50 imgs/class) → FID → classify → UA + RA  → consolidated JSON
#
# The checkpoints are sharded round-robin across GPUS and one orchestrator
# process runs PER GPU in parallel. Each worker handles its shard sequentially
# (and is itself fault-tolerant — a bad checkpoint never stops the others).
#
# The forget class + scale variant are auto-parsed from each filename
# (cls_<C>, MUKSB_scale_<variant>). Append ':CLASS' to a path to override.
#
# Usage:
#   # edit CKPTS below, then run across GPUs 2,5,6:
#   bash run_scale_eval.sh
#
#   # custom GPUs / pass checkpoints on the command line:
#   GPUS="2 5 6" bash run_scale_eval.sh /path/a.pt /path/b.pt ...
#
#   # single GPU (old behaviour):
#   GPUS="2" bash run_scale_eval.sh
# ============================================================================
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SD_DIR}"

PY=${PY:-/storage/s25017/miniconda3/envs/munba3/bin/python}
GPUS=${GPUS:-2 5 6}                 # GPUs to shard across (parallel workers)
IMAGE_SIZE=${IMAGE_SIZE:-512}
NUM_SAMPLES=${NUM_SAMPLES:-5}      # 10*NUM_SAMPLES imgs/class -> 50
EVAL_ROOT=${EVAL_ROOT:-${SCRIPT_DIR}/scale_ablation}
EXTRA_ARGS=${EXTRA_ARGS:-}         # e.g. EXTRA_ARGS="--skip_existing"
read -ra GPU_LIST <<< "${GPUS}"
NUM_GPUS=${#GPU_LIST[@]}

# ── checkpoints (override by passing paths as args) ──────────────────────────
if [ "$#" -gt 0 ]; then
    CKPTS=("$@")
else
    CKPTS=(
        # class 1
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_1-MUKSB_scale_arithmetic-method_full-lr_5e-06_E5_U955_/diffusers-cls_1-MUKSB_scale_arithmetic-method_full-lr_5e-06_E5_U955_-epoch_4.pt"
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_1-MUKSB_scale_fixed-method_full-lr_5e-06_E5_U955_/diffusers-cls_1-MUKSB_scale_fixed-method_full-lr_5e-06_E5_U955_-epoch_4.pt"
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_1-MUKSB_scale_min-method_full-lr_5e-06_E5_U955_/diffusers-cls_1-MUKSB_scale_min-method_full-lr_5e-06_E5_U955_-epoch_4.pt"
        # class 3
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_3-MUKSB_scale_arithmetic-method_full-lr_5e-06_E5_U858_/diffusers-cls_3-MUKSB_scale_arithmetic-method_full-lr_5e-06_E5_U858_-epoch_4.pt"
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_3-MUKSB_scale_fixed-method_full-lr_5e-06_E5_U858_/diffusers-cls_3-MUKSB_scale_fixed-method_full-lr_5e-06_E5_U858_-epoch_4.pt"
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_3-MUKSB_scale_min-method_full-lr_5e-06_E5_U858_/diffusers-cls_3-MUKSB_scale_min-method_full-lr_5e-06_E5_U858_-epoch_4.pt"
        # class 6
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_6-MUKSB_scale_arithmetic-method_full-lr_5e-06_E5_U961_/diffusers-cls_6-MUKSB_scale_arithmetic-method_full-lr_5e-06_E5_U961_-epoch_4.pt"
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_6-MUKSB_scale_fixed-method_full-lr_5e-06_E5_U961_/diffusers-cls_6-MUKSB_scale_fixed-method_full-lr_5e-06_E5_U961_-epoch_4.pt"
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_6-MUKSB_scale_min-method_full-lr_5e-06_E5_U961_/diffusers-cls_6-MUKSB_scale_min-method_full-lr_5e-06_E5_U961_-epoch_4.pt"
        # class 7
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_7-MUKSB_scale_arithmetic-method_full-lr_5e-06_E5_U931_/diffusers-cls_7-MUKSB_scale_arithmetic-method_full-lr_5e-06_E5_U931_-epoch_4.pt"
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_7-MUKSB_scale_fixed-method_full-lr_5e-06_E5_U931_/diffusers-cls_7-MUKSB_scale_fixed-method_full-lr_5e-06_E5_U931_-epoch_4.pt"
        "/scratch/s25017/MUKSB/SD/models/compvis-cls_7-MUKSB_scale_min-method_full-lr_5e-06_E5_U931_/diffusers-cls_7-MUKSB_scale_min-method_full-lr_5e-06_E5_U931_-epoch_4.pt"
    )
fi

LOG_DIR="${EVAL_ROOT}/logs"
RESULTS_DIR="${EVAL_ROOT}/results"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

echo "============================================================"
echo " Scale-ablation eval — multi-GPU"
echo "  GPUs        : ${GPU_LIST[*]}  (${NUM_GPUS} parallel worker(s))"
echo "  Checkpoints : ${#CKPTS[@]}"
echo "  Eval root   : ${EVAL_ROOT}"
echo "  Image size  : ${IMAGE_SIZE}   Num samples: ${NUM_SAMPLES}"
echo "============================================================"

# ── shard checkpoints round-robin across GPUs ────────────────────────────────
PIDS=()
WGPU=()
for gi in "${!GPU_LIST[@]}"; do
    GPU=${GPU_LIST[$gi]}
    SHARD=()
    for ci in "${!CKPTS[@]}"; do
        if [ $(( ci % NUM_GPUS )) -eq "${gi}" ]; then
            SHARD+=("${CKPTS[$ci]}")
        fi
    done

    if [ "${#SHARD[@]}" -eq 0 ]; then
        echo "[gpu ${GPU}] no checkpoints — skipping"
        continue
    fi

    WLOG="${LOG_DIR}/worker_gpu${GPU}.log"
    WSUMMARY="${RESULTS_DIR}/scale_eval_summary.gpu${GPU}.json"
    echo "[gpu ${GPU}] ${#SHARD[@]} checkpoint(s) → ${WLOG}"

    "${PY}" eval_scripts/run_scale_eval.py \
        --device "cuda:${GPU}" \
        --image_size "${IMAGE_SIZE}" \
        --num_samples "${NUM_SAMPLES}" \
        --eval_root "${EVAL_ROOT}" \
        --summary_json "${WSUMMARY}" \
        ${EXTRA_ARGS} \
        --checkpoints "${SHARD[@]}" \
        > "${WLOG}" 2>&1 &
    PIDS+=($!)
    WGPU+=("${GPU}")
done

# ── wait for all workers ─────────────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[gpu ${WGPU[$i]}] worker OK"
    else
        echo "[gpu ${WGPU[$i]}] worker exited non-zero — see ${LOG_DIR}/worker_gpu${WGPU[$i]}.log"
        FAILED=$((FAILED + 1))
    fi
done

# ── merge per-GPU summaries into one combined summary ────────────────────────
COMBINED="${RESULTS_DIR}/scale_eval_summary.json"
"${PY}" - "${RESULTS_DIR}" "${COMBINED}" <<'PYEOF'
import glob, json, os, sys
results_dir, combined = sys.argv[1], sys.argv[2]
merged = []
for f in sorted(glob.glob(os.path.join(results_dir, "scale_eval_summary.gpu*.json"))):
    try:
        with open(f) as fh:
            merged.extend(json.load(fh))
    except Exception as e:
        print(f"  [merge warn] {f}: {e}")
with open(combined, "w") as fh:
    json.dump(merged, fh, indent=2)

def g(r, k):
    v = r.get(k)
    return "—" if v is None else v

print(f"\nCombined summary ({len(merged)} checkpoints): {combined}")
print(f"{'variant':>12} {'cls':>4} {'FID':>8} {'UA1':>7} {'UA5':>7} {'RA':>7}  status")
for r in merged:
    fid = g(r, "fid")
    fid = f"{fid:>8.3f}" if isinstance(fid, (int, float)) else f"{str(fid):>8}"
    print(f"{str(g(r,'scale_variant')):>12} {str(g(r,'forget_class_idx')):>4} "
          f"{fid} {str(g(r,'ua_top1')):>7} {str(g(r,'ua_top5')):>7} "
          f"{str(g(r,'ra')):>7}  {r.get('status','?')}")
n_fail = sum(1 for r in merged if r.get("status") == "failed")
n_part = sum(1 for r in merged if r.get("status") == "partial")
print(f"\nok={sum(1 for r in merged if r.get('status')=='ok')}  "
      f"partial={n_part}  failed={n_fail}")
PYEOF

echo "============================================================"
if [ "${FAILED}" -eq 0 ]; then
    echo "All ${NUM_GPUS} worker(s) finished."
else
    echo "${FAILED} worker(s) exited non-zero (individual checkpoints may still"
    echo "have succeeded — check ${COMBINED} and per-worker logs)."
fi
echo "Combined summary: ${COMBINED}"
echo "============================================================"
exit ${FAILED}
