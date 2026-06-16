"""
SD/train_scripts/dataset.py — MUKSB
Stable Diffusion dataset utilities: Imagenette forget/retain splits,
model loading from CompVis checkpoint + config, and NSFW dataset stubs.

All SD infrastructure (ldm, omegaconf, etc.) is local to MUKSB/SD/.
"""
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as torch_transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import Imagenette
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SD_DIR   = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SD_DIR not in sys.path:
    sys.path.insert(0, _SD_DIR)

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from datasets import load_dataset


INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic":  InterpolationMode.BICUBIC,
    "lanczos":  InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    return torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def setup_model(config, ckpt, device):
    """Load a CompVis Stable Diffusion model from config path + checkpoint."""
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    global_step = pl_sd.get("global_step", None)
    if global_step is None:
        print("global_step key not found in model checkpoint")
    sd    = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(config.model)
    m, u  = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Imagenette helpers
# ─────────────────────────────────────────────────────────────────────────────

IMAGENETTE_WNID_TO_NAME = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def imagenette_class_names(dataset):
    """Return human-readable class names, robust to torchvision version differences."""
    names = []
    for cls in dataset.classes:
        if isinstance(cls, (tuple, list)):
            names.append(cls[0])
        else:
            names.append(cls)
    return names


def _load_imagenette(image_size, interpolation="bicubic",
                     root="/storage/s25017/Datasets"):
    """Load Imagenette train split once and return (dataset, descriptions)."""
    transform = get_transform(INTERPOLATIONS[interpolation], image_size)
    dataset   = Imagenette(root=root, split="train", transform=transform, download=False)
    class_names  = imagenette_class_names(dataset)
    descriptions = [f"an image of a {n}" for n in class_names]
    return dataset, descriptions


def setup_forget_remain_data(class_to_forget, batch_size, image_size,
                              interpolation="bicubic",
                              root="/storage/s25017/Datasets"):
    """
    Build forget and retain DataLoaders from a single Imagenette scan.
    Preferred over calling setup_forget_data + setup_remain_data separately
    because the filesystem scan happens only once.

    Returns
    -------
    forget_loader, remain_loader, descriptions
    """
    dataset, descriptions = _load_imagenette(image_size, interpolation, root)
    assert 0 <= class_to_forget < len(dataset.classes), \
        f"class_to_forget={class_to_forget} out of range"

    forget_idx, remain_idx = [], []
    for i, s in enumerate(dataset._samples):
        if s[1] == class_to_forget:
            forget_idx.append(i)
        else:
            remain_idx.append(i)

    forget_loader = DataLoader(Subset(dataset, forget_idx),
                               batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True)
    remain_loader = DataLoader(Subset(dataset, remain_idx),
                               batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True)
    return forget_loader, remain_loader, descriptions


def setup_remain_data(class_to_forget, batch_size, image_size,
                      interpolation="bicubic", root="/storage/s25017/Datasets"):
    """DataLoader of retain samples (all Imagenette classes except class_to_forget)."""
    dataset, descriptions = _load_imagenette(image_size, interpolation, root)
    assert 0 <= class_to_forget < len(dataset.classes), \
        f"class_to_forget={class_to_forget} out of range"
    remain_idx    = [i for i, s in enumerate(dataset._samples)
                     if s[1] != class_to_forget]
    remain_loader = DataLoader(Subset(dataset, remain_idx),
                               batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True, prefetch_factor=4)
    return remain_loader, descriptions


def setup_forget_data(class_to_forget, batch_size, image_size,
                      interpolation="bicubic", root="/storage/s25017/Datasets"):
    """DataLoader of forget samples (only class_to_forget from Imagenette)."""
    dataset, descriptions = _load_imagenette(image_size, interpolation, root)
    assert 0 <= class_to_forget < len(dataset.classes), \
        f"class_to_forget={class_to_forget} out of range"
    forget_idx    = [i for i, s in enumerate(dataset._samples)
                     if s[1] == class_to_forget]
    forget_loader = DataLoader(Subset(dataset, forget_idx),
                               batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True, prefetch_factor=4)
    return forget_loader, descriptions


def setup_data(class_to_forget, batch_size, image_size,
               interpolation="bicubic", root="/storage/s25017/Datasets"):
    """Full Imagenette training split (all classes)."""
    dataset, descriptions = _load_imagenette(image_size, interpolation, root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True,
                      persistent_workers=True, prefetch_factor=4), descriptions


# ─────────────────────────────────────────────────────────────────────────────
# Objectnette helpers — SD-generated 10-class OBJECT dataset, stored in the same
# ImageFolder layout as imagenette2 (one subdir per class under <root>/train).
# This makes object-concept removal use the EXACT same code path as Imagenette
# class removal: pick a class index to forget, the other 9 become the retain set.
# ─────────────────────────────────────────────────────────────────────────────

OBJECTNETTE_ROOT = "/storage/s25017/Datasets/objectnette2"

# The set of classes to generate (folder names). NOTE: torchvision ImageFolder
# assigns label indices by SORTED folder name, so the authoritative index->name
# map is taken from dataset.classes at load time. This list only declares which
# class folders should exist.
OBJECTNETTE_CLASSES = [
    "dog", "cat", "car", "bicycle", "airplane",
    "bird", "horse", "boat", "truck", "train",
]


def _load_objectnette(image_size, interpolation="bicubic",
                      root=OBJECTNETTE_ROOT):
    """Load the SD-generated objectnette train split (ImageFolder) and return
    (dataset, descriptions). Mirrors _load_imagenette exactly."""
    from torchvision.datasets import ImageFolder
    transform = get_transform(INTERPOLATIONS[interpolation], image_size)
    train_dir = os.path.join(root, "train")
    dataset   = ImageFolder(train_dir, transform=transform)
    descriptions = [f"an image of a {name}" for name in dataset.classes]
    return dataset, descriptions


def setup_objectnette_forget_remain_data(class_to_forget, batch_size, image_size,
                                         interpolation="bicubic",
                                         root=OBJECTNETTE_ROOT):
    """Forget/retain DataLoaders for objectnette — identical interface and
    behaviour to setup_forget_remain_data (Imagenette), just reading the
    SD-generated ImageFolder dataset.

    Returns
    -------
    forget_loader, remain_loader, descriptions
    """
    dataset, descriptions = _load_objectnette(image_size, interpolation, root)
    assert 0 <= class_to_forget < len(dataset.classes), \
        f"class_to_forget={class_to_forget} out of range (0..{len(dataset.classes)-1})"

    forget_idx, remain_idx = [], []
    for i, (_, target) in enumerate(dataset.samples):
        (forget_idx if target == class_to_forget else remain_idx).append(i)

    forget_loader = DataLoader(Subset(dataset, forget_idx),
                               batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True)
    remain_loader = DataLoader(Subset(dataset, remain_idx),
                               batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True)
    return forget_loader, remain_loader, descriptions


# ─────────────────────────────────────────────────────────────────────────────
# NSFW dataset stubs (for future use / NSFW unlearning experiments)
# ─────────────────────────────────────────────────────────────────────────────

class NSFW(Dataset):
    """NSFW image dataset (loads from local HuggingFace cache)."""
    def __init__(self, transform=None):
        self.dataset   = load_dataset("data/nsfw")["train"]
        self.transform = transform

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if self.transform:
            image = self.transform(image)
        return image


class NOT_NSFW(Dataset):
    """Non-NSFW image dataset (loads from local HuggingFace cache)."""
    def __init__(self, transform=None):
        self.dataset   = load_dataset("data/not-nsfw")["train"]
        self.transform = transform

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if self.transform:
            image = self.transform(image)
        return image


# ─────────────────────────────────────────────────────────────────────────────
# File-based NSFW datasets (local image directories)
# ─────────────────────────────────────────────────────────────────────────────

import glob as _glob

class NSFWDataset(Dataset):
    """NSFW images loaded from a local directory (PNG/JPG files, recursive)."""
    def __init__(self, img_dir, transform, image_key="jpg", txt_key="txt", caption=None):
        self.img_dir    = img_dir
        self.all_imgs   = (
            _glob.glob(os.path.join(img_dir, "**/*.png"),  recursive=True) +
            _glob.glob(os.path.join(img_dir, "**/*.jpg"),  recursive=True) +
            _glob.glob(os.path.join(img_dir, "**/*.jpeg"), recursive=True)
        )
        self.caption    = caption or "a photo of a nude person"
        self.captions   = [c.strip() for c in self.caption.split(",")]
        self.image_key  = image_key
        self.txt_key    = txt_key
        self.transform  = transform

    def __len__(self): return len(self.all_imgs)

    def __getitem__(self, idx):
        img_name = self.all_imgs[idx]
        max_retries = 10
        for attempt in range(max_retries):
            try:
                image = Image.open(img_name).convert("RGB")
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to load image after {max_retries} retries: {img_name}")
                idx      = random.randint(0, len(self.all_imgs) - 1)
                img_name = self.all_imgs[idx]
        cap_idx   = int(os.path.basename(img_name).split("_")[0]) % len(self.captions)
        text_cond = self.captions[cap_idx]
        image     = self.transform(image).permute(1, 2, 0)
        return {self.image_key: image, self.txt_key: text_cond}


class NotNSFWDataset(Dataset):
    """Non-NSFW images loaded from a local directory (PNG/JPG files)."""
    def __init__(self, img_dir, transform, image_key="jpg", txt_key="txt", caption=None):
        self.img_dir   = img_dir
        self.all_imgs  = (
            _glob.glob(os.path.join(img_dir, "*.png"))  +
            _glob.glob(os.path.join(img_dir, "*.jpg"))  +
            _glob.glob(os.path.join(img_dir, "*.jpeg"))
        )
        self.caption   = caption or "a photo of a person wearing clothes"
        self.captions  = [c.strip() for c in self.caption.split(",")]
        self.image_key = image_key
        self.txt_key   = txt_key
        self.transform = transform

    def __len__(self): return len(self.all_imgs)

    def __getitem__(self, idx):
        img_name = self.all_imgs[idx]
        max_retries = 10
        for attempt in range(max_retries):
            try:
                image = Image.open(img_name).convert("RGB")
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to load image after {max_retries} retries: {img_name}")
                idx      = random.randint(0, len(self.all_imgs) - 1)
                img_name = self.all_imgs[idx]
        image = self.transform(image).permute(1, 2, 0)
        return {self.image_key: image, self.txt_key: self.captions[0]}


def setup_nsfw_data(batch_size, forget_path, remain_path, image_size,
                    interpolation="bicubic", num_workers=8):
    """DataLoaders for file-based NSFW (forget) and NotNSFW (remain) datasets."""
    transform  = get_transform(INTERPOLATIONS[interpolation], image_size)
    forget_dl  = DataLoader(NSFWDataset(forget_path,  transform), batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4,
                            drop_last=True)
    remain_dl  = DataLoader(NotNSFWDataset(remain_path, transform), batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4,
                            drop_last=True)
    return forget_dl, remain_dl


# ─────────────────────────────────────────────────────────────────────────────
# Object-concept datasets (generic objects: dog / car / bicycle)
#
# Images are generated from SD v1.4 (see Evaluation/objects/generate_objects_*),
# which writes a manifest.csv (filename,prompt) next to the PNGs. Each image is
# returned with the *prompt that generated it* as its caption, so the forget /
# retain conditioning matches the image content. Falls back to `default_caption`
# when no manifest is present.
# ─────────────────────────────────────────────────────────────────────────────

import csv as _csv


def _load_manifest(img_dir):
    """Return {basename: prompt} from <img_dir>/manifest.csv, or {} if absent."""
    path = os.path.join(img_dir, "manifest.csv")
    if not os.path.isfile(path):
        return {}
    mapping = {}
    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            fname = row.get("filename") or row.get("file") or row.get("image")
            prompt = row.get("prompt", "")
            if fname:
                mapping[os.path.basename(fname)] = prompt
    return mapping


class ObjectDataset(Dataset):
    """Generic-object images from a directory, captioned via manifest.csv.

    Returns {"jpg": HWC float tensor in [-1, 1], "txt": prompt} to match the
    LDM ``get_input`` / ``shared_step`` contract used elsewhere in MUKSB.
    """

    def __init__(self, img_dir, transform, default_caption="a photo",
                 image_key="jpg", txt_key="txt"):
        self.img_dir   = img_dir
        self.all_imgs  = sorted(
            _glob.glob(os.path.join(img_dir, "**/*.png"),  recursive=True) +
            _glob.glob(os.path.join(img_dir, "**/*.jpg"),  recursive=True) +
            _glob.glob(os.path.join(img_dir, "**/*.jpeg"), recursive=True)
        )
        if not self.all_imgs:
            raise RuntimeError(f"ObjectDataset: no images found under {img_dir}")
        self.manifest        = _load_manifest(img_dir)
        self.default_caption = default_caption
        self.image_key       = image_key
        self.txt_key         = txt_key
        self.transform       = transform

    def __len__(self):
        return len(self.all_imgs)

    def _caption_for(self, img_name):
        return self.manifest.get(os.path.basename(img_name), self.default_caption)

    def __getitem__(self, idx):
        img_name = self.all_imgs[idx]
        for attempt in range(10):
            try:
                image = Image.open(img_name).convert("RGB")
                break
            except Exception:
                if attempt == 9:
                    raise RuntimeError(f"Failed to load image: {img_name}")
                idx      = random.randint(0, len(self.all_imgs) - 1)
                img_name = self.all_imgs[idx]
        caption = self._caption_for(img_name)
        image   = self.transform(image).permute(1, 2, 0)  # CHW -> HWC
        return {self.image_key: image, self.txt_key: caption}


def setup_object_data(batch_size, forget_path, remain_path, image_size,
                      forget_caption="a photo", remain_caption="a photo",
                      interpolation="bicubic", num_workers=8):
    """DataLoaders for generic-object forget / retain image directories.

    `*_caption` are only used as a fallback when an image has no manifest entry.
    """
    transform = get_transform(INTERPOLATIONS[interpolation], image_size)
    forget_dl = DataLoader(
        ObjectDataset(forget_path, transform, default_caption=forget_caption),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
        drop_last=True)
    remain_dl = DataLoader(
        ObjectDataset(remain_path, transform, default_caption=remain_caption),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
        drop_last=True)
    return forget_dl, remain_dl
