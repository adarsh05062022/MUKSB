"""
SDXL/train_scripts/dataset.py
SDXL model setup (HuggingFace Diffusers) and NSFW image datasets.
No CompVis/LDM dependency — uses diffusers pipeline components directly.
"""
import glob
import os
import random
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

ImageFile.LOAD_TRUNCATED_IMAGES = True

INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic":  InterpolationMode.BICUBIC,
    "lanczos":  InterpolationMode.LANCZOS,
}


def get_transform(interpolation=InterpolationMode.BICUBIC, size=1024):
    return T.Compose([
        T.Resize(size, interpolation=interpolation),
        T.CenterCrop(size),
        T.Lambda(lambda img: img.convert("RGB")),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# SDXL model loading
# ─────────────────────────────────────────────────────────────────────────────

def setup_sdxl_components(model_id, device, dtype=torch.bfloat16):
    """
    Load SDXL from HuggingFace Hub and return trainable components separately.

    The UNet is returned in training mode with requires_grad=True.
    VAE and text encoders are frozen (inference-only, no grad).

    Returns
    -------
    unet, vae, text_encoder, text_encoder_2,
    tokenizer, tokenizer_2, scheduler
    """
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)

    unet           = pipe.unet
    vae            = pipe.vae
    text_encoder   = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer      = pipe.tokenizer
    tokenizer_2    = pipe.tokenizer_2

    # DDPMScheduler is more stable for fine-tuning than the pipeline default
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze everything except the UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    unet.train()

    return unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, scheduler


# ─────────────────────────────────────────────────────────────────────────────
# SDXL forward-pass helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_text_sdxl(tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompts, device):
    """
    Dual-CLIP text encoding for SDXL.

    Returns
    -------
    encoder_hidden_states : [B, 77, 2048]  (CLIP-L 768 || CLIP-G 1280)
    pooled_embeds         : [B, 1280]      (CLIP-G pooled output)
    """
    # CLIP-L
    tok1 = tokenizer(
        prompts, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        h1 = text_encoder(
            tok1.input_ids.to(device),
            output_hidden_states=True,
        ).hidden_states[-2]  # [B, 77, 768]

    # CLIP-G
    tok2 = tokenizer_2(
        prompts, padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        out2   = text_encoder_2(tok2.input_ids.to(device), output_hidden_states=True)
        h2     = out2.hidden_states[-2]  # [B, 77, 1280]
        pooled = out2[0]                 # [B, 1280]

    enc_hidden = torch.cat([h1, h2], dim=-1)  # [B, 77, 2048]
    return enc_hidden, pooled


def get_time_ids(image_size, batch_size, device, dtype):
    """Standard SDXL time_ids: (orig_h, orig_w, crop_top, crop_left, tgt_h, tgt_w)."""
    single = [image_size, image_size, 0, 0, image_size, image_size]
    return torch.tensor([single] * batch_size, device=device, dtype=dtype)


def encode_images_to_latents(vae, images, device, dtype):
    """VAE-encode images to scaled latents (frozen, no grad)."""
    with torch.no_grad():
        latents = vae.encode(images.to(device, dtype=dtype)).latent_dist.sample()
    return latents * vae.config.scaling_factor


def unet_forward(unet, noisy_latents, t, enc_hidden, pooled, image_size):
    """Single UNet forward pass with SDXL added-cond kwargs."""
    B     = noisy_latents.shape[0]
    dtype = noisy_latents.dtype
    dev   = noisy_latents.device
    time_ids   = get_time_ids(image_size, B, dev, dtype)
    added_cond = {"text_embeds": pooled.to(dtype), "time_ids": time_ids}
    return unet(
        noisy_latents,
        t,
        encoder_hidden_states=enc_hidden.to(dtype),
        added_cond_kwargs=added_cond,
    ).sample


def compute_retain_loss(
    unet, vae, scheduler,
    tokenizer, tokenizer_2, text_encoder, text_encoder_2,
    images, prompts, device, image_size, dtype,
):
    """
    MSE denoising loss on the retain set — replaces model.shared_step().
    Gradients flow only through the UNet.
    """
    latents = encode_images_to_latents(vae, images, device, dtype)
    B       = latents.shape[0]
    noise   = torch.randn_like(latents)
    t       = torch.randint(
        0, scheduler.config.num_train_timesteps, (B,), device=device
    )
    noisy   = scheduler.add_noise(latents, noise, t)
    enc_h, pooled = encode_text_sdxl(
        tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompts, device
    )
    pred = unet_forward(unet, noisy, t, enc_h, pooled, image_size)
    return torch.nn.functional.mse_loss(pred, noise.to(dtype))


# ─────────────────────────────────────────────────────────────────────────────
# NSFW datasets
# ─────────────────────────────────────────────────────────────────────────────

class NSFWDataset(Dataset):
    """NSFW images from a local directory. Returns (image_chw, caption) tuples."""
    def __init__(self, img_dir, transform, caption="a photo of a nude person"):
        self.all_imgs = sorted(
            glob.glob(os.path.join(img_dir, "**/*.png"),  recursive=True) +
            glob.glob(os.path.join(img_dir, "**/*.jpg"),  recursive=True) +
            glob.glob(os.path.join(img_dir, "**/*.jpeg"), recursive=True)
        )
        self.captions  = [c.strip() for c in caption.split(",")]
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = self.all_imgs[idx]
        for _ in range(10):
            try:
                image = Image.open(img_path).convert("RGB")
                break
            except Exception:
                idx      = random.randint(0, len(self.all_imgs) - 1)
                img_path = self.all_imgs[idx]
        cap_idx = int(os.path.basename(img_path).split("_")[0]) % len(self.captions)
        return self.transform(image), self.captions[cap_idx]


class NotNSFWDataset(Dataset):
    """Non-NSFW images from a local directory. Returns (image_chw, caption) tuples."""
    def __init__(self, img_dir, transform, caption="a photo of a person wearing clothes"):
        self.all_imgs = sorted(
            glob.glob(os.path.join(img_dir, "*.png"))  +
            glob.glob(os.path.join(img_dir, "*.jpg"))  +
            glob.glob(os.path.join(img_dir, "*.jpeg"))
        )
        self.caption   = caption
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = self.all_imgs[idx]
        for _ in range(10):
            try:
                image = Image.open(img_path).convert("RGB")
                break
            except Exception:
                idx      = random.randint(0, len(self.all_imgs) - 1)
                img_path = self.all_imgs[idx]
        return self.transform(image), self.caption


def setup_nsfw_data(batch_size, forget_path, remain_path, image_size=1024,
                    interpolation="bicubic", num_workers=8):
    """DataLoaders for file-based NSFW (forget) and NotNSFW (retain) datasets."""
    transform = get_transform(INTERPOLATIONS[interpolation], image_size)
    forget_dl = DataLoader(
        NSFWDataset(forget_path, transform),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )
    remain_dl = DataLoader(
        NotNSFWDataset(remain_path, transform),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )
    return forget_dl, remain_dl
