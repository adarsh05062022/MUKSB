"""
IP2P/dataset_i2i.py
===================
Image-to-Image NSFW dataset for InstructPix2Pix unlearning under MUKSB.

Each sample returns:
    - "jpg" : target image tensor (HWC, [-1, 1])
    - "src" : source image tensor for I2I conditioning (HWC, [-1, 1])
    - "txt" : edit instruction (string)

Modes:
    - "forget" → NSFW edit instructions cycled per sample
    - "retain" → diverse benign attribute-edit instructions cycled
                 (used for base-model distillation so general editing
                  capability is preserved during unlearning)

Source and target are the same image — the simplest, cleanest I2I
setup for this unlearning task.
"""

import glob
import os
import random

import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True


INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic":  InterpolationMode.BICUBIC,
    "lanczos":  InterpolationMode.LANCZOS,
}


def _to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    return T.Compose([
        T.Resize(size, interpolation=interpolation),
        T.CenterCrop(size),
        _to_rgb,
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])


FORGET_INSTRUCTIONS = [
    "make this person nude",
    "remove the clothes from the subject",
    "show the person without clothing",
    "edit this into explicit nudity",
]

# Diverse benign attribute edits.  The retain branch distills the trainable
# UNet toward the frozen base model on these instructions, so the model keeps
# its general editing ability instead of collapsing to an identity / copy map.
RETAIN_INSTRUCTIONS = [
    "make the person smile",
    "give the person a happy expression",
    "add sunglasses to the person",
    "make the person wear a hat",
    "change the hair color to blonde",
    "make the person look older",
    "change the background to a forest",
    "change the background to a city street",
    "add a beard to the person",
    "make the lighting warmer",
    "turn this into a watercolor painting",
    "make the person wear a red jacket",
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
    transform = get_transform(INTERPOLATIONS[interpolation], image_size)
    forget_set = I2INSFWDataset(forget_path, transform, mode="forget")
    remain_set = I2INSFWDataset(remain_path, transform, mode="retain")
    forget_dl = DataLoader(forget_set, batch_size=batch_size, shuffle=True, num_workers=2)
    remain_dl = DataLoader(remain_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return forget_dl, remain_dl
