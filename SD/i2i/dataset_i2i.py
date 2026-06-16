"""
i2i/dataset_i2i.py
==================
Image-to-Image NSFW dataset for InstructPix2Pix unlearning under MUKSB.

Each sample returns:
    - "jpg" : target image tensor (HWC, [-1, 1])  — same format as
              MUKSB's T2I dataset (see train_scripts/dataset.NSFWDataset)
    - "src" : source image tensor for I2I conditioning (HWC, [-1, 1])
    - "txt" : edit instruction (string)

Modes:
    - "forget" → NSFW edit instructions cycled per sample
    - "retain" → neutral / clothed-preserving instructions cycled

Source and target are the same image — the simplest, cleanest I2I
setup for this unlearning task.
"""

import glob
import os
import random
import sys

from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# Reuse the SD-level transform so image preprocessing matches the
# T2I MUKSB code exactly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SD_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SD_DIR not in sys.path:
    sys.path.insert(0, _SD_DIR)
from train_scripts.dataset import get_transform, INTERPOLATIONS

ImageFile.LOAD_TRUNCATED_IMAGES = True


FORGET_INSTRUCTIONS = [
    "make this person nude",
    "remove the clothes from the subject",
    "show the person without clothing",
    "edit this into explicit nudity",
]

RETAIN_INSTRUCTIONS = [
    "keep the person fully clothed",
    "preserve the original outfit",
    "maintain the dressed appearance",
    "keep the clothing intact",
]


class I2INSFWDataset(Dataset):
    def __init__(self, img_dir, transform, mode="forget"):
        assert mode in ("forget", "retain"), (
            f"mode must be 'forget' or 'retain', got {mode!r}"
        )
        self.mode = mode
        self.transform = transform

        pngs = glob.glob(os.path.join(img_dir, "**/*.png"), recursive=True)
        if not pngs:
            pngs = glob.glob(os.path.join(img_dir, "*.png"))
        self.all_imgs = pngs

        self.instructions = (
            FORGET_INSTRUCTIONS if mode == "forget" else RETAIN_INSTRUCTIONS
        )

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        path = self.all_imgs[idx]
        while True:
            try:
                image = Image.open(path).convert("RGB")
                break
            except Exception:
                idx = random.randint(0, len(self.all_imgs) - 1)
                path = self.all_imgs[idx]

        img = self.transform(image).permute(1, 2, 0)  # HWC, [-1, 1]
        instruction = self.instructions[idx % len(self.instructions)]
        return {"jpg": img, "src": img.clone(), "txt": instruction}


def setup_i2i_nsfw_data(
    batch_size, forget_path, remain_path, image_size, interpolation="bicubic"
):
    """Mirror of train_scripts.dataset.setup_nsfw_data, yielding I2I dicts."""
    transform = get_transform(INTERPOLATIONS[interpolation], image_size)
    forget_set = I2INSFWDataset(forget_path, transform, mode="forget")
    remain_set = I2INSFWDataset(remain_path, transform, mode="retain")
    forget_dl = DataLoader(forget_set, batch_size=batch_size, shuffle=True, num_workers=2)
    remain_dl = DataLoader(remain_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return forget_dl, remain_dl
