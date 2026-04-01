"""
dataset.py — FID data utilities supporting single or multi-class forgetting.
"""

import os
from typing import List, Union

import numpy as np
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

# ── Interpolation lookup ────────────────────────────────────────────────────
INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic":  InterpolationMode.BICUBIC,
    "lanczos":  InterpolationMode.LANCZOS,
}

ClassSpec = Union[int, List[int]]  # single class or list of classes


def _to_set(class_to_forget: ClassSpec) -> set:
    """Normalise class_to_forget to a Python set for consistent membership tests."""
    if isinstance(class_to_forget, (list, tuple)):
        return set(class_to_forget)
    return {int(class_to_forget)}


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def get_transform(interpolation: InterpolationMode = InterpolationMode.BICUBIC,
                  size: int = 512) -> T.Compose:
    """
    Standard image transform used for FID evaluation.

    Output tensor is in [-1, 1] (Normalize([0.5], [0.5])).
    Caller must denormalise to [0, 255] uint8 before feeding to FID metric.
    """
    return T.Compose([
        T.Resize((size, size), interpolation=interpolation),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])


# ── Real Imagenette dataset ─────────────────────────────────────────────────

class Imagenette(Dataset):
    """
    Wraps the HuggingFace frgfm/imagenette dataset.

    When class_to_forget is set, any sample from a forgotten class receives a
    random label instead (for unlearning training).  For FID evaluation, use
    setup_fid_data which filters by label directly.
    """

    def __init__(self,
                 split: str,
                 class_to_forget: ClassSpec = None,
                 transform=None):
        self.dataset = load_dataset("frgfm/imagenette", "160px")[split]
        self.class_to_idx = {
            cls: i
            for i, cls in enumerate(self.dataset.features["label"].names)
        }
        self.forget_set = _to_set(class_to_forget) if class_to_forget is not None else set()
        self.num_classes = max(self.class_to_idx.values()) + 1
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image   = example["image"]
        label   = example["label"]

        if label in self.forget_set:
            label = np.random.randint(0, self.num_classes)

        if self.transform:
            image = self.transform(image)

        return image, label


# ── Fake (generated) Imagenette dataset ────────────────────────────────────

class Fake_Imagenette(Dataset):
    """
    Loads generated .png images from a directory.

    Filename convention expected: ``{class_idx}_{anything}.png``

    Files belonging to any class in class_to_forget are excluded so that the
    generated set only contains retain-class images for FID comparison.
    """

    def __init__(self,
                 data_dir: str,
                 class_to_forget: ClassSpec,
                 transform=None):
        self.data_dir  = data_dir
        self.transform = transform
        forget_set     = _to_set(class_to_forget)

        all_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

        # Filter: exclude any file whose leading class index is in forget_set.
        # Parse the class index robustly (no string-prefix tricks).
        self.image_files = []
        for f in all_files:
            try:
                cls_idx = int(f.split("_")[0])
            except ValueError:
                continue  # skip files with unexpected naming
            if cls_idx not in forget_set:
                self.image_files.append(f)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx):
        filename   = self.image_files[idx]
        class_idx  = int(filename.split("_")[0])
        image_path = os.path.join(self.data_dir, filename)
        image      = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, class_idx


# ── I2P (no-label) dataset ──────────────────────────────────────────────────

class Fake_I2P(Dataset):
    """Generic image folder dataset (no class labels). Used for I2P FID evaluation."""

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


# ── FID data builders ───────────────────────────────────────────────────────

def setup_fid_data(class_to_forget: ClassSpec,
                   path: str,
                   image_size: int,
                   interpolation: str = "bicubic"):
    
    forget_set = _to_set(class_to_forget)
    interp     = INTERPOLATIONS[interpolation]
    transform  = get_transform(interp, image_size)

    # ── Generated images FIRST to know the count ──
    fake_ds  = Fake_Imagenette(path, class_to_forget, transform=transform)
    fake_set = [img for img, _ in fake_ds]
    max_real = len(fake_set)   # match exactly

    # ── Real images capped to fake count ──
    train_ds = Imagenette("train", transform=transform)
    real_set = []
    for image, label in train_ds:
        if label not in forget_set:
            real_set.append(image)
        if len(real_set) >= max_real:
            break

    print(f"Real: {len(real_set)} | Fake: {len(fake_set)}")  # sanity check
    return real_set, fake_set

def setup_fid_data_i2p(real_path: str,
                        path: str,
                        image_size: int,
                        interpolation: str = "bicubic"):
    """
    Returns (real_set, fake_set) for I2P FID (no class filtering).
    """
    interp    = INTERPOLATIONS[interpolation]
    transform = get_transform(interp, image_size)

    real_set = Fake_I2P(real_path, transform=transform)
    fake_set = Fake_I2P(path, transform=transform)

    return real_set, fake_set