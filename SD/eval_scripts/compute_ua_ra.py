"""
Compute Unlearning Accuracy (UA) and Retention Accuracy (RA) from classification CSVs.

Each CSV file is from a model that forgot a DIFFERENT class.
The filename tells which class was forgotten: cls_0, cls_1, etc.

For each CSV:
  UA   = % of forget-class images where top-1 prediction is NOT the
         forget class's ImageNet index  (higher = better forgetting)
  RA   = per-class accuracy on retain classes, averaged
         (top-1 == correct ImageNet index)  (higher = better retention)

Usage
-----
    python SD/eval_scripts/compute_ua_ra.py \\
        --csv_dir SD/eval_scripts/CLASS/UA/Imagenatte
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

# Imagenette metadata
IMAGENETTE_CLASSES = [
    "tench", "english springer", "cassette player", "chain saw",
    "church", "french horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]

# Imagenette class index → ImageNet-1k index
IMAGENETTE_TO_IMAGENET_IDX = {
    0: 0,    # tench
    1: 217,  # english springer
    2: 482,  # cassette player
    3: 491,  # chain saw
    4: 497,  # church
    5: 566,  # french horn
    6: 569,  # garbage truck
    7: 571,  # gas pump
    8: 574,  # golf ball
    9: 701,  # parachute
}


def extract_class_to_forget(filename: str) -> int:
    """Extract class index from filename like 'diffusers-cls_0-...'"""
    match = re.search(r'cls_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def compute_ua_ra_single_csv(csv_path: str, class_to_forget: int, topk: int = 5):
    """
    Compute UA and RA for a single CSV file.

    Args:
        csv_path: Path to classification CSV
        class_to_forget: Imagenette class index that was forgotten
        topk: Top-k for UA calculation
    """
    df = pd.read_csv(csv_path)

    forget_imagenet = IMAGENETTE_TO_IMAGENET_IDX[class_to_forget]
    forget_name = IMAGENETTE_CLASSES[class_to_forget]
    per_class_results = {}

    # Process each class
    for cls_idx in range(10):
        class_data = df[df["classidx"] == cls_idx]
        if len(class_data) == 0:
            continue

        class_name = IMAGENETTE_CLASSES[cls_idx]
        target_imagenet = IMAGENETTE_TO_IMAGENET_IDX[cls_idx]
        total = len(class_data)

        if cls_idx == class_to_forget:
            # UA = % where top-1 is NOT the forget class
            ua_top1 = (class_data["index_top1"] != forget_imagenet).sum() / total * 100

            # UA_topk = % where forget class does NOT appear in top-k
            topk_cols = [f"index_top{k}" for k in range(1, topk + 1)]
            in_topk = class_data[topk_cols].apply(
                lambda row: forget_imagenet in row.values, axis=1
            )
            ua_topk = (~in_topk).sum() / total * 100

            per_class_results[class_name] = {
                "role": "forget",
                "total": total,
                "ua_top1": round(ua_top1, 2),
                f"ua_top{topk}": round(ua_topk, 2),
                "imagenet_idx": forget_imagenet,
            }
        else:
            # RA = % where top-1 == correct class
            correct = (class_data["index_top1"] == target_imagenet).sum()
            acc = correct / total * 100

            per_class_results[class_name] = {
                "role": "retain",
                "total": total,
                "acc_top1": round(acc, 2),
                "correct": int(correct),
                "imagenet_idx": target_imagenet,
            }

    # Calculate aggregates
    ua_top1 = per_class_results[forget_name]["ua_top1"]
    ua_topk = per_class_results[forget_name][f"ua_top{topk}"]

    retain_accs = [
        v["acc_top1"]
        for k, v in per_class_results.items()
        if v["role"] == "retain"
    ]
    ra = round(sum(retain_accs) / len(retain_accs), 2) if retain_accs else None

    return {
        "class_to_forget": class_to_forget,
        "forget_class": forget_name,
        "ua_top1": ua_top1,
        f"ua_top{topk}": ua_topk,
        "ra": ra,
        "per_class": per_class_results,
    }


def compute_ua_ra(csv_dir: str, topk: int = 5):
    """
    Process all CSV files and compute UA/RA for each.

    Args:
        csv_dir: Directory containing classification CSV files
        topk: Top-k for UA calculation
    """
    csv_files = sorted(Path(csv_dir).glob("*_classification.csv"))

    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return None

    print(f"Processing {len(csv_files)} CSV files\n")

    all_results = {}

    # Process each CSV file
    for csv_path in csv_files:
        filename = csv_path.name
        class_to_forget = extract_class_to_forget(filename)

        if class_to_forget is None:
            print(f"  [SKIP] Could not extract class from {filename}")
            continue

        print(f"  [{class_to_forget}] {filename}")

        try:
            results = compute_ua_ra_single_csv(str(csv_path), class_to_forget, topk)
            all_results[class_to_forget] = results

        except Exception as e:
            print(f"       Error: {e}")
            continue

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Class':<20} {'Forget':<12} {'UA (top-1)':<15} {'UA (top-5)':<15} {'RA (avg)':<15}")
    print(f"{'='*80}")

    for cls_idx in sorted(all_results.keys()):
        results = all_results[cls_idx]
        forget_name = results["forget_class"]
        ua_top1 = results["ua_top1"]
        ua_topk = results[f"ua_top{topk}"]
        ra = results["ra"]

        print(f"{forget_name:<20} cls_{cls_idx:<10} {ua_top1:>6.2f}%{'':<7} {ua_topk:>6.2f}%{'':<7} {ra:>6.2f}%")

    print(f"{'='*80}\n")

    # Save detailed results
    output_json = os.path.join(csv_dir, "ua_ra_summary.json")
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Result] Detailed results saved to {output_json}\n")

    # Print per-class details
    for cls_idx in sorted(all_results.keys()):
        results = all_results[cls_idx]
        forget_name = results["forget_class"]
        ua_top1 = results["ua_top1"]
        ra = results["ra"]

        print(f"\nClass {cls_idx} ({forget_name}) - Forget Class")
        print(f"  UA (top-1): {ua_top1}% (higher = better forgetting)")
        print(f"  RA (avg):   {ra}% (higher = better retention)")
        print(f"  Per-retain-class accuracy:")

        for cls_name_key in sorted(results["per_class"].keys()):
            info = results["per_class"][cls_name_key]
            if info["role"] == "retain":
                print(f"    {cls_name_key:<22}  {info['acc_top1']:6.2f}%"
                      f"  ({info['correct']}/{info['total']})")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute UA + RA from Imagenette classification CSVs"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="SD/eval_scripts/CLASS/UA/Imagenatte",
        help="Directory containing classification CSV files"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k for UA calculation"
    )
    args = parser.parse_args()

    compute_ua_ra(
        csv_dir=args.csv_dir,
        topk=args.topk,
    )
