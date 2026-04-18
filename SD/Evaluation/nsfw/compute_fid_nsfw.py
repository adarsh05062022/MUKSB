"""
Evaluation/nsfw/compute_fid_nsfw.py
=====================================
Compute FID between generated NSFW/I2P images and real COCO images.

Mirrors compute_fid_i2p() from SD/eval_scripts/compute_fid.py exactly:
    - FrechetInceptionDistance(feature=2048)
    - Bicubic resize to image_size
    - Images normalised to [-1, 1] then de-normalised to uint8 [0, 255]
    - GPU, batched DataLoader

Usage
-----
    python Evaluation/nsfw/compute_fid_nsfw.py \\
        --gen_dir   Evaluation/nsfw/coco_30k/my-model \\
        --real_path /storage/s25017/Datasets/COCO/coco_30_val_2014_images \\
        --device    0
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

# ── import shared dataset helpers from eval_scripts ───────────────────────────
_EVAL_SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "eval_scripts")
)
if _EVAL_SCRIPTS not in sys.path:
    sys.path.insert(0, _EVAL_SCRIPTS)

from dataset import setup_fid_data_i2p   # noqa: E402  (after sys.path patch)


def compute_fid_nsfw(
    gen_dir: str,
    real_path: str,
    image_size: int = 512,
    batch_size: int = 256,
    device: str = "cuda:0",
) -> dict:
    """
    Compute FID between generated images in *gen_dir* and real images in
    *real_path*, using the same code path as compute_fid_i2p() in
    eval_scripts/compute_fid.py (feature=2048).

    Returns a dict with key ``fid`` (float).
    """
    results = {
        "gen_dir":    gen_dir,
        "real_path":  real_path,
        "image_size": image_size,
        "fid":        None,
    }

    if not os.path.isdir(gen_dir):
        print(f"[FID-NSFW] ERROR: gen_dir not found: {gen_dir}")
        return results

    if not os.path.isdir(real_path):
        print(f"[FID-NSFW] ERROR: real_path not found: {real_path}")
        return results

    # ── build datasets (same as compute_fid_i2p) ─────────────────────────────
    real_ds, fake_ds = setup_fid_data_i2p(
        real_path, gen_dir, image_size, interpolation="bicubic"
    )

    real_loader = DataLoader(real_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    fake_loader = DataLoader(fake_ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    print(f"[FID-NSFW] Real images : {len(real_ds)}")
    print(f"[FID-NSFW] Fake images : {len(fake_ds)}")

    # ── FID metric — feature=2048 matches compute_fid_i2p ────────────────────
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    print("[FID-NSFW] Updating with real images …")
    for batch in real_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        # de-normalise [-1, 1] → uint8 [0, 255]  (same as compute_fid_i2p)
        batch = ((batch + 1) * 127.5).clamp(0, 255).to(torch.uint8).to(device)
        fid_metric.update(batch, real=True)

    print("[FID-NSFW] Updating with generated images …")
    for batch in fake_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = ((batch + 1) * 127.5).clamp(0, 255).to(torch.uint8).to(device)
        fid_metric.update(batch, real=False)

    fid_val = fid_metric.compute().item()
    fid_metric.reset()
    del fid_metric

    results["fid"] = round(fid_val, 4)

    print(f"\n{'='*50}")
    print(f"  FID (NSFW / COCO) : {fid_val:.4f}  (lower = better retention)")
    print(f"{'='*50}\n")

    out_json = os.path.join(gen_dir, "fid_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[FID-NSFW] Results saved to {out_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FID between generated NSFW images and real COCO images"
    )
    parser.add_argument("--gen_dir",    type=str, required=True,
                        help="Folder containing generated PNG images")
    parser.add_argument("--real_path",  type=str,
                        default="/storage/s25017/Datasets/COCO/coco_30_val_2014_images",
                        help="Folder containing real COCO PNG images")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device",     type=str, default="0")
    args = parser.parse_args()

    compute_fid_nsfw(
        gen_dir    = args.gen_dir,
        real_path  = args.real_path,
        image_size = args.image_size,
        batch_size = args.batch_size,
        device     = f"cuda:{args.device}",
    )
