"""
Evaluation/imagenette/compute_ua_imagenette.py
===============================================
Compute Unlearning Accuracy (UA) and Retention Accuracy (RA) for
Imagenette-10 using ResNet50 (ImageNet weights) — matching the MUNBa
approach in eval_scripts/imageclassify.py.

Classification approach
-----------------------
  • ResNet50 with ResNet50_Weights.DEFAULT (1000-class ImageNet)
  • Top-k predictions per image (default k=5), batched (default 250)
  • Imagenette classes are a subset of ImageNet — each maps to a known
    ImageNet index (see IMAGENETTE_TO_IMAGENET_IDX below)

Metrics
-------
  UA   = % of forget-class images where top-1 prediction is NOT the
         forget class's ImageNet index  (higher = better forgetting)
  UA5  = same but top-5  (UA5 ≥ UA always)
  RA   = per-class accuracy on retain classes, averaged
         (top-1 == correct ImageNet index)  (higher = better retention)

Outputs (saved inside gen_root)
--------------------------------
  ua_ra_results.json             — machine-readable summary
  <cls>_classification.csv       — per-image top-k CSV for every class

Usage
-----
    python Evaluation/imagenette/compute_ua_imagenette.py \\
        --gen_root        Evaluation/imagenette/generated/my-model \\
        --class_to_forget 0 \\
        --device          0

    # No --classifier_path needed — uses ResNet50 from torchvision.
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

# ── Imagenette metadata ───────────────────────────────────────────────────────
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]

# Imagenette class index → ImageNet-1k index (used by ResNet50 default weights)
IMAGENETTE_TO_IMAGENET_IDX = {
    0: 0,    # tench
    1: 217,  # English springer
    2: 482,  # cassette player
    3: 491,  # chain saw
    4: 497,  # church
    5: 566,  # French horn
    6: 569,  # garbage truck
    7: 571,  # gas pump
    8: 574,  # golf ball
    9: 701,  # parachute
}

NUM_CLASSES = len(IMAGENETTE_CLASSES)


def _load_model(device: str):
    weights = ResNet50_Weights.DEFAULT
    model   = resnet50(weights=weights)
    model.to(device).eval()
    return model, weights


def _classify_folder(
    folder: str,
    model,
    preprocess,
    weights,
    device: str,
    topk: int,
    batch_size: int,
) -> pd.DataFrame:
    """
    Classify all images in folder.
    Returns DataFrame with columns:
        case_number, category_top1..topk, index_top1..topk, scores_top1..topk
    """
    names = sorted([
        n for n in os.listdir(folder)
        if n.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if not names:
        return pd.DataFrame()

    images = []
    for name in tqdm(names, desc=f"  Loading {os.path.basename(folder)}", leave=False):
        img = Image.open(os.path.join(folder, name)).convert("RGB")
        images.append(preprocess(img))

    images     = torch.stack(images)
    n          = len(names)
    bs         = min(batch_size, n)

    scores_dict     = defaultdict(list)
    indexes_dict    = defaultdict(list)
    categories_dict = defaultdict(list)

    for i in range(((n - 1) // bs) + 1):
        batch = images[i * bs: min(n, (i + 1) * bs)].to(device)
        with torch.no_grad():
            prediction = model(batch).softmax(1)
        probs, class_ids = torch.topk(prediction, topk, dim=1)

        for k in range(1, topk + 1):
            scores_dict[f"top{k}"].extend(
                probs[:, k - 1].detach().cpu().tolist()
            )
            indexes_dict[f"top{k}"].extend(
                class_ids[:, k - 1].detach().cpu().tolist()
            )
            categories_dict[f"top{k}"].extend([
                weights.meta["categories"][idx]
                for idx in class_ids[:, k - 1].detach().cpu().tolist()
            ])

    # parse case numbers from filenames: "<case_number>_*.png" or "00000.png"
    case_numbers = []
    for name in names:
        stem = name.split("_")[0].replace(".png", "").replace(".jpg", "")
        try:
            case_numbers.append(int(stem))
        except ValueError:
            case_numbers.append(-1)

    row = {"case_number": case_numbers}
    for k in range(1, topk + 1):
        row[f"category_top{k}"] = categories_dict[f"top{k}"]
        row[f"index_top{k}"]    = indexes_dict[f"top{k}"]
        row[f"scores_top{k}"]   = scores_dict[f"top{k}"]

    return pd.DataFrame(row)


def compute_ua_ra(
    gen_root: str,
    class_to_forget: int,
    device: str,
    topk: int = 5,
    batch_size: int = 250,
):
    model, weights = _load_model(device)
    preprocess     = weights.transforms()

    retain_indices  = [i for i in range(NUM_CLASSES) if i != class_to_forget]
    forget_imagenet = IMAGENETTE_TO_IMAGENET_IDX[class_to_forget]

    per_class_results = {}

    # ── process each class folder ─────────────────────────────────────────────
    for cls_idx in range(NUM_CLASSES):
        cls_name  = IMAGENETTE_CLASSES[cls_idx].replace(" ", "_")
        cls_dir   = os.path.join(gen_root, f"{cls_idx:02d}_{cls_name}")

        if not os.path.isdir(cls_dir):
            # fallback: scan for dir starting with index prefix
            for entry in sorted(os.listdir(gen_root)):
                if entry.startswith(f"{cls_idx:02d}_"):
                    cls_dir = os.path.join(gen_root, entry)
                    break

        if not os.path.isdir(cls_dir):
            print(f"  [UA] class {cls_idx} ({cls_name}): dir not found, skipping.")
            continue

        print(f"  [UA] Classifying class {cls_idx:2d}  {cls_name:<22} → {cls_dir}")
        df = _classify_folder(
            cls_dir, model, preprocess, weights,
            device, topk, batch_size,
        )
        if df.empty:
            print(f"       No images found.")
            continue

        # save per-class CSV
        csv_path = os.path.join(gen_root, f"{cls_idx:02d}_{cls_name}_classification.csv")
        df.to_csv(csv_path, index=False)

        # ── compute UA (forget class only) ────────────────────────────────────
        target_imagenet = IMAGENETTE_TO_IMAGENET_IDX[cls_idx]
        top1_correct    = (df["index_top1"] == target_imagenet).sum()
        total           = len(df)

        if cls_idx == class_to_forget:
            # UA = fraction where top-1 is NOT the forget class
            ua_top1 = (df["index_top1"] != forget_imagenet).sum() / total * 100
            # UA5 = fraction where forget class does not appear in top-5
            topk_cols   = [f"index_top{k}" for k in range(1, topk + 1)]
            in_topk     = df[topk_cols].apply(
                lambda row: forget_imagenet in row.values, axis=1
            )
            ua_topk     = (~in_topk).sum() / total * 100
            per_class_results[IMAGENETTE_CLASSES[cls_idx]] = {
                "role":          "forget",
                "total":         total,
                "ua_top1":       round(ua_top1, 2),
                f"ua_top{topk}": round(ua_topk, 2),
                "imagenet_idx":  forget_imagenet,
            }
        else:
            # RA = fraction where top-1 == correct ImageNet index
            acc = top1_correct / total * 100
            per_class_results[IMAGENETTE_CLASSES[cls_idx]] = {
                "role":         "retain",
                "total":        total,
                "acc_top1":     round(acc, 2),
                "correct":      int(top1_correct),
                "imagenet_idx": target_imagenet,
            }

    # ── aggregate ─────────────────────────────────────────────────────────────
    forget_name = IMAGENETTE_CLASSES[class_to_forget]
    ua_top1  = per_class_results.get(forget_name, {}).get("ua_top1")
    ua_topk  = per_class_results.get(forget_name, {}).get(f"ua_top{topk}")

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
        "class_to_forget":  class_to_forget,
        "forget_class":     forget_name,
        "ua_top1":          ua_top1,
        f"ua_top{topk}":    ua_topk,
        "ra":               ra,
        "per_class":        per_class_results,
    }

    out_json = os.path.join(gen_root, "ua_ra_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[UA/RA] Results saved to {out_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UA + RA for Imagenette-10 using ResNet50 (ImageNet weights)"
    )
    parser.add_argument("--gen_root",        type=str, required=True,
                        help="Root folder containing per-class generated image sub-dirs")
    parser.add_argument("--class_to_forget", type=int, default=0,
                        help="Imagenette class index to forget (0-9)")
    parser.add_argument("--device",          type=str, default="0")
    parser.add_argument("--topk",            type=int, default=5)
    parser.add_argument("--batch_size",      type=int, default=250)
    args = parser.parse_args()

    compute_ua_ra(
        gen_root        = args.gen_root,
        class_to_forget = args.class_to_forget,
        device          = f"cuda:{args.device}",
        topk            = args.topk,
        batch_size      = args.batch_size,
    )
