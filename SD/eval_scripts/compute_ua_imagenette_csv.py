"""
Compute Unlearning Accuracy (UA) and Retention Accuracy (RA) for Imagenette-10
from classification CSV files.

Metrics
-------
  UA   = % of forget-class images where top-1 prediction is NOT the
         forget class's ImageNet index  (higher = better forgetting)
  UA5  = same but top-5  (UA5 ≥ UA always)
  RA   = per-class accuracy on retain classes, averaged
         (top-1 == correct ImageNet index)  (higher = better retention)

Outputs
-------
  ua_ra_results.json  — machine-readable summary with UA, RA metrics
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# ── Imagenette metadata ───────────────────────────────────────────────────────
IMAGENETTE_CLASSES = [
    "tench", "english springer", "cassette player", "chain saw",
    "church", "french horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]

# Imagenette class index → ImageNet-1k index (used by ResNet50 default weights)
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

NUM_CLASSES = len(IMAGENETTE_CLASSES)


def compute_ua_ra_from_csvs(
    csv_dir: str,
    class_to_forget: int,
    topk: int = 5,
    output_json: str = None,
):
    """
    Compute UA and RA from classification CSV files.

    Args:
        csv_dir: Directory containing classification CSV files
        class_to_forget: Imagenette class index to forget (0-9)
        topk: Use top-k for UA calculation
        output_json: Path to save results JSON
    """
    # Find all CSV files
    csv_files = sorted([
        f for f in os.listdir(csv_dir)
        if f.endswith("_classification.csv")
    ])

    if not csv_files:
        print(f"No classification CSV files found in {csv_dir}")
        return None

    print(f"\nProcessing {len(csv_files)} CSV files from: {csv_dir}")
    print(f"Class to forget: {class_to_forget} ({IMAGENETTE_CLASSES[class_to_forget]})")

    per_class_results = {}
    forget_imagenet = IMAGENETTE_TO_IMAGENET_IDX[class_to_forget]

    # ── process each CSV ──────────────────────────────────────────────────────
    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)

        # Extract class from CSV content (from 'prompt' or 'class' column)
        if "class" in df.columns:
            # Get the class from the first row (all rows should have same class)
            class_name = df["class"].iloc[0]
        elif "prompt" in df.columns:
            # Extract from prompt like "Image of tench"
            class_name = df["prompt"].iloc[0].replace("Image of ", "").lower()
        else:
            print(f"  [WARN] Could not determine class for {csv_file}, skipping")
            continue

        # Find matching Imagenette class index
        cls_idx = None
        for idx, cls in enumerate(IMAGENETTE_CLASSES):
            if cls.lower() in class_name.lower() or class_name.lower() in cls.lower():
                cls_idx = idx
                break

        if cls_idx is None:
            print(f"  [WARN] Could not match class '{class_name}' in {csv_file}, skipping")
            continue

        target_imagenet = IMAGENETTE_TO_IMAGENET_IDX[cls_idx]
        total = len(df)

        # ── compute metrics ───────────────────────────────────────────────────
        if cls_idx == class_to_forget:
            # UA = fraction where top-1 is NOT the forget class
            ua_top1 = (df["index_top1"] != forget_imagenet).sum() / total * 100

            # UA5 = fraction where forget class does not appear in top-5
            topk_cols = [f"index_top{k}" for k in range(1, topk + 1)]
            in_topk = df[topk_cols].apply(
                lambda row: forget_imagenet in row.values, axis=1
            )
            ua_topk = (~in_topk).sum() / total * 100

            per_class_results[IMAGENETTE_CLASSES[cls_idx]] = {
                "role": "forget",
                "total": total,
                "ua_top1": round(ua_top1, 2),
                f"ua_top{topk}": round(ua_topk, 2),
                "imagenet_idx": forget_imagenet,
            }
        else:
            # RA = fraction where top-1 == correct ImageNet index
            top1_correct = (df["index_top1"] == target_imagenet).sum()
            acc = top1_correct / total * 100

            per_class_results[IMAGENETTE_CLASSES[cls_idx]] = {
                "role": "retain",
                "total": total,
                "acc_top1": round(acc, 2),
                "correct": int(top1_correct),
                "imagenet_idx": target_imagenet,
            }

    # ── aggregate ─────────────────────────────────────────────────────────────
    forget_name = IMAGENETTE_CLASSES[class_to_forget]
    ua_top1 = per_class_results.get(forget_name, {}).get("ua_top1")
    ua_topk = per_class_results.get(forget_name, {}).get(f"ua_top{topk}")

    retain_accs = [
        v["acc_top1"]
        for k, v in per_class_results.items()
        if v["role"] == "retain"
    ]
    ra = round(sum(retain_accs) / len(retain_accs), 2) if retain_accs else None

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Forget class : {forget_name} (idx={class_to_forget},"
          f" ImageNet idx={forget_imagenet})")
    print(f"  UA  (top-1)  : {ua_top1}%   (higher = better forgetting)")
    print(f"  UA  (top-{topk})  : {ua_topk}%   (higher = better forgetting)")
    print(f"  RA  (avg)    : {ra}%    (higher = better retention)")
    print(f"{'='*60}")
    print(f"\n  Per-retain-class top-1 accuracy:")
    for cls_name_key, info in per_class_results.items():
        if info["role"] == "retain":
            print(f"    {cls_name_key:<22}  acc={info['acc_top1']:6.2f}%"
                  f"  ({info['correct']}/{info['total']})")

    results = {
        "class_to_forget": class_to_forget,
        "forget_class": forget_name,
        "ua_top1": ua_top1,
        f"ua_top{topk}": ua_topk,
        "ra": ra,
        "per_class": per_class_results,
    }

    # Save results JSON
    if output_json is None:
        output_json = os.path.join(csv_dir, "ua_ra_results.json")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[UA/RA] Results saved to {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UA + RA for Imagenette-10 from classification CSVs"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Directory containing classification CSV files"
    )
    parser.add_argument(
        "--class_to_forget",
        type=int,
        default=0,
        help="Imagenette class index to forget (0-9)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save results JSON (default: csv_dir/ua_ra_results.json)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k for UA calculation"
    )
    args = parser.parse_args()

    compute_ua_ra_from_csvs(
        csv_dir=args.csv_dir,
        class_to_forget=args.class_to_forget,
        topk=args.topk,
        output_json=args.output_json,
    )
