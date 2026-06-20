"""
SD3/Evaluation/nsfw/compute_fid_nsfw.py
=========================================
Compute FID between generated NSFW/I2P images and real COCO images.

Mirrors SD/Evaluation/nsfw/compute_fid_nsfw.py exactly:
    - FrechetInceptionDistance(feature=2048)
    - Bicubic resize to image_size
    - Images normalised to [-1, 1] then de-normalised to uint8 [0, 255]
    - GPU, batched DataLoader

Self-contained: the tiny image-folder dataset is inlined here so this module
does not depend on the SD/eval_scripts tree (which does not exist under SD3).

Usage
-----
    python Evaluation/nsfw/compute_fid_nsfw.py \\
        --gen_dir   Evaluation/nsfw/coco_5k/my-model \\
        --real_path /storage/s25017/Datasets/COCO/coco_5k_val_2014_images \\
        --device    0
"""

import argparse
import json
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import InterpolationMode

# ── Interpolation lookup ──────────────────────────────────────────────────────
INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic":  InterpolationMode.BICUBIC,
    "lanczos":  InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def get_transform(interpolation: InterpolationMode = InterpolationMode.BICUBIC,
                  size: int = 512) -> T.Compose:
    """Output tensor in [-1, 1]; caller de-normalises to uint8 [0, 255]."""
    return T.Compose([
        T.Resize((size, size), interpolation=interpolation),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])


class Fake_I2P(Dataset):
    """Generic image folder dataset (no class labels). Used for I2P FID."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir    = data_dir
        self.transform   = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image      = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def setup_fid_data_i2p(real_path: str, path: str, image_size: int,
                       interpolation: str = "bicubic"):
    """Returns (real_set, fake_set) for I2P FID (no class filtering)."""
    interp    = INTERPOLATIONS[interpolation]
    transform = get_transform(interp, image_size)
    real_set  = Fake_I2P(real_path, transform=transform)
    fake_set  = Fake_I2P(path, transform=transform)
    return real_set, fake_set


def compute_fid_nsfw(
    gen_dir: str,
    real_path: str,
    image_size: int = 512,
    batch_size: int = 256,
    device: str = "cuda:0",
) -> dict:
    """
    Compute FID between generated images in *gen_dir* and real images in
    *real_path* (feature=2048).  Returns a dict with key ``fid`` (float).
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

    real_ds, fake_ds = setup_fid_data_i2p(
        real_path, gen_dir, image_size, interpolation="bicubic"
    )

    real_loader = DataLoader(real_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    fake_loader = DataLoader(fake_ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    print(f"[FID-NSFW] Real images : {len(real_ds)}")
    print(f"[FID-NSFW] Fake images : {len(fake_ds)}")

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    print("[FID-NSFW] Updating with real images …")
    for batch in real_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
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
                        default="/storage/s25017/Datasets/COCO/coco_5k_val_2014_images",
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
