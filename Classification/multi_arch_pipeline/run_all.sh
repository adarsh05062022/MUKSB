#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Run the full multi-architecture ablation: ResNet-18 + VGG-16,
# each with both `MUKSB` and `retrain` baselines.
#
# Runs architectures sequentially (each one already parallelises
# seeds across 5 GPUs).  After all runs finish, aggregates the
# results into a single CSV for the paper supplementary.
# ─────────────────────────────────────────────────────────────────

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCHES=("${@}")
if [ ${#ARCHES[@]} -eq 0 ]; then
    ARCHES=(resnet18 vgg16_bn swin_t)
fi

OVERALL_FAILED=0
for A in "${ARCHES[@]}"; do
    case "${A}" in
        resnet18)
            echo "############################################################"
            echo "#  ResNet-18 / CIFAR-10"
            echo "############################################################"
            bash "${SCRIPT_DIR}/run_resnet18_cifar10.sh"
            rc=$?
            ;;
        vgg16_bn|vgg16)
            echo "############################################################"
            echo "#  VGG-16 (BN) / CIFAR-10"
            echo "############################################################"
            bash "${SCRIPT_DIR}/run_vgg16_cifar10.sh"
            rc=$?
            ;;
        swin_t|swin)
            echo "############################################################"
            echo "#  Swin-T / CIFAR-10"
            echo "############################################################"
            bash "${SCRIPT_DIR}/run_swin_t_cifar10.sh"
            rc=$?
            ;;
        *)
            echo "Unknown architecture: ${A} (supported: resnet18, vgg16_bn, swin_t)"
            rc=1
            ;;
    esac
    if [ "${rc}" -ne 0 ]; then
        echo "WARNING: arch ${A} pipeline returned non-zero (rc=${rc})"
        OVERALL_FAILED=$((OVERALL_FAILED + 1))
    fi
done

echo ""
echo "############################################################"
echo "#  Aggregating results"
echo "############################################################"
python "${SCRIPT_DIR}/aggregate_results.py" \
    --results_root "$(cd "${SCRIPT_DIR}/.." && pwd)/results_multi_arch" \
    --out_csv      "${SCRIPT_DIR}/multi_arch_summary.csv"

echo ""
if [ "${OVERALL_FAILED}" -eq 0 ]; then
    echo "All architectures completed successfully."
else
    echo "${OVERALL_FAILED} architecture pipeline(s) had failures."
fi
exit ${OVERALL_FAILED}
