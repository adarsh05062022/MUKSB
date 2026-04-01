import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode


# -------------------------------------------------------------------------
# Utilities (copied minimal dependencies, no NSFW coupling)
# -------------------------------------------------------------------------

INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    return torch_transforms.Compose(
        [
            torch_transforms.Resize(size, interpolation=interpolation),
            torch_transforms.CenterCrop(size),
            _convert_image_to_rgb,
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.5], [0.5]),
        ]
    )


# -------------------------------------------------------------------------
# Style Forget Dataset
# -------------------------------------------------------------------------

class StyleForgetDataset(Dataset):
    """
    Dataset for STYLE FORGETTING.
    Uses a single fixed style prompt for all images.
    """

    def __init__(
        self,
        img_dir,
        transform,
        style_prompt="van gogh style painting",
        image_key="jpg",
        txt_key="txt",
    ):
        super().__init__()
        self.img_dir = img_dir
        self.all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        assert len(self.all_imgs) > 0, f"No images found in {img_dir}"

        self.style_prompt = style_prompt
        self.image_key = image_key
        self.txt_key = txt_key
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = self.all_imgs[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image).permute(1, 2, 0)  # [H, W, C]

        return {
            self.image_key: image,
            self.txt_key: self.style_prompt,
        }


# -------------------------------------------------------------------------
# Style Preserve Dataset
# -------------------------------------------------------------------------

class StylePreserveDataset(Dataset):
    """
    Dataset for PRESERVATION (neutral / photographic).
    Uses a single fixed preserve prompt for all images.
    """

    def __init__(
        self,
        img_dir,
        transform,
        preserve_prompt="a realistic photograph",
        image_key="jpg",
        txt_key="txt",
    ):
        super().__init__()
        self.img_dir = img_dir
        self.all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        assert len(self.all_imgs) > 0, f"No images found in {img_dir}"

        self.preserve_prompt = preserve_prompt
        self.image_key = image_key
        self.txt_key = txt_key
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = self.all_imgs[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image).permute(1, 2, 0)  # [H, W, C]

        return {
            self.image_key: image,
            self.txt_key: self.preserve_prompt,
        }


# -------------------------------------------------------------------------
# Setup function (style replacement for setup_nsfw_data)
# -------------------------------------------------------------------------

def setup_style_data(
    batch_size,
    forget_path,
    preserve_path,
    image_size,
    style_prompt="van gogh style painting",
    preserve_prompt="a realistic photograph",
    interpolation="bicubic",
):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    forget_set = StyleForgetDataset(
        img_dir=forget_path,
        transform=transform,
        style_prompt=style_prompt,
    )

    preserve_set = StylePreserveDataset(
        img_dir=preserve_path,
        transform=transform,
        preserve_prompt=preserve_prompt,
    )

    forget_dl = DataLoader(
        forget_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    preserve_dl = DataLoader(
        preserve_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return forget_dl, preserve_dl
