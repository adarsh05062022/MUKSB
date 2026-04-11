"""
Evaluation/imagenette/compute_fid_imagenette.py
================================================
Compute FID for the retain classes only (all classes except class_to_forget).

FID is computed between:
    generated : <gen_root>/<cls>/*.png  (retain classes only)
    reference : /storage/s25017/Datasets/imagenette2/val/<wnid>/

Uses torchmetrics FrechetInceptionDistance(feature=64) to match the paper.

Usage
-----
    python Evaluation/imagenette/compute_fid_imagenette.py \\
        --gen_root       generated/my-model \\
        --class_to_forget 0 \\
        --device          0
"""

import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# ── Imagenette metadata ───────────────────────────────────────────────────────
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]
IMAGENETTE_WNIDS = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
]
IMAGENETTE_VAL = "/storage/s25017/Datasets/imagenette2/val"


# ── Image loading helper ──────────────────────────────────────────────────────

class ImagePathDataset(Dataset):
    """Reads images from a list of paths and returns uint8 tensors (3, H, W).
    Corrupt / truncated images are skipped at construction time."""

    def __init__(self, img_paths: list, image_size: int = 299):
        Image.MAX_IMAGE_PIXELS = None          # lift decompression-bomb limit
        # verify every path up-front; drop anything that can't be decoded
        valid = []
        for p in img_paths:
            try:
                with Image.open(p) as im:
                    im.verify()                # catches truncated / corrupt files
                valid.append(p)
            except (UnidentifiedImageError, Exception):
                print(f"[FID] Skipping corrupt/truncated image: {p}")
        self.paths = valid
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.PILToTensor(),                   # → uint8 (3, H, W) in [0, 255]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def _collect_img_paths(dirs: list) -> list:
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(d, fname))
    return paths


# ── Public API ────────────────────────────────────────────────────────────────

def compute_fid(gen_root: str, class_to_forget: int, device: str, batch_size: int = 64):
    retain_indices = [i for i in range(10) if i != class_to_forget]

    # ── collect generated retain images ──────────────────────────────────────
    gen_dirs = []
    for i in retain_indices:
        cls_name = IMAGENETTE_CLASSES[i].replace(" ", "_")
        d = os.path.join(gen_root, f"{i:02d}_{cls_name}")
        if not os.path.isdir(d):
            for entry in os.listdir(gen_root):
                if entry.startswith(f"{i:02d}_"):
                    d = os.path.join(gen_root, entry)
                    break
        gen_dirs.append(d)

    # ── collect reference retain images ──────────────────────────────────────
    ref_dirs = [os.path.join(IMAGENETTE_VAL, IMAGENETTE_WNIDS[i]) for i in retain_indices]

    gen_paths = _collect_img_paths(gen_dirs)
    ref_paths = _collect_img_paths(ref_dirs)

    # match reference count to generated count for a fair comparison
    if len(ref_paths) > len(gen_paths):
        rng = np.random.default_rng(seed=42)
        ref_paths = rng.choice(ref_paths, size=len(gen_paths), replace=False).tolist()

    print(f"\n[FID] Forget class : {IMAGENETTE_CLASSES[class_to_forget]} (idx={class_to_forget})")
    print(f"[FID] Retain gen   : {len(gen_paths)} images")
    print(f"[FID] Retain ref   : {len(ref_paths)} images  (sampled to match gen count)")

    results = {
        "class_to_forget":   class_to_forget,
        "forget_class_name": IMAGENETTE_CLASSES[class_to_forget],
        "gen_root":          gen_root,
        "retain_fid":        None,
    }

    if len(gen_paths) == 0:
        print("[FID] ERROR: No generated retain images found. Run generate_imagenette.py first.")
        return results

    # ── build dataloaders ─────────────────────────────────────────────────────
    real_loader = DataLoader(
        ImagePathDataset(ref_paths),
        batch_size=batch_size, num_workers=4, pin_memory=True,
    )
    fake_loader = DataLoader(
        ImagePathDataset(gen_paths),
        batch_size=batch_size, num_workers=4, pin_memory=True,
    )

    # ── compute FID with feature=64 (paper setting) on GPU in batches ─────────
    fid_metric = FrechetInceptionDistance(feature=64).to(device)

    print("[FID] Updating with reference images …")
    for batch in tqdm(real_loader, desc="  Real", leave=False):
        fid_metric.update(batch.to(device), real=True)

    print("[FID] Updating with generated images …")
    for batch in tqdm(fake_loader, desc="  Fake", leave=False):
        fid_metric.update(batch.to(device), real=False)

    fid_val = fid_metric.compute().item()
    fid_metric.reset()
    del fid_metric

    results["retain_fid"] = round(fid_val, 4)

    print(f"\n{'='*50}")
    print(f"  Retain FID : {fid_val:.4f}  (lower = better retention)")
    print(f"{'='*50}\n")

    out_json = os.path.join(gen_root, "fid_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[FID] Results saved to {out_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID on Imagenette retain classes")
    parser.add_argument("--gen_root",        type=str, required=True)
    parser.add_argument("--class_to_forget", type=int, default=0)
    parser.add_argument("--device",          type=str, default="0")
    parser.add_argument("--batch_size",      type=int, default=64)
    args = parser.parse_args()

    compute_fid(
        gen_root        = args.gen_root,
        class_to_forget = args.class_to_forget,
        device          = f"cuda:{args.device}",
        batch_size      = args.batch_size,
    )