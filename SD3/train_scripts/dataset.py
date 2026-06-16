"""
SD3/train_scripts/dataset.py
SD3 model setup (HuggingFace Diffusers) and NSFW image datasets.

Key SD3 differences vs SDXL:
  - SD3Transformer2DModel  (DiT)  instead of UNet2DConditionModel
  - Three text encoders: CLIP-L, CLIP-G, T5-XXL
  - 16-channel VAE latents with shift_factor
  - Flow-matching noise via sigmas  (not DDPM add_noise)
  - Velocity target: (noise - latents)  (not noise)
"""
import glob
import os
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

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
# SD3 model loading
# ─────────────────────────────────────────────────────────────────────────────

def setup_sd3_components(model_id, device, dtype=torch.bfloat16, skip_t5=False):
    """
    Load SD3 from HuggingFace Hub and return trainable components separately.

    Parameters
    ----------
    skip_t5 : bool
        If True, T5-XXL is not loaded — saves ~9.4 GB but reduces text understanding.
        CLIP-L + CLIP-G are always loaded.

    Returns
    -------
    transformer, vae, pipe (kept alive for encode_prompt),
    scheduler
    """
    kwargs = dict(torch_dtype=dtype, use_safetensors=True)
    if skip_t5:
        kwargs["text_encoder_3"] = None
        kwargs["tokenizer_3"]    = None

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, **kwargs).to(device)

    transformer = pipe.transformer
    vae         = pipe.vae
    scheduler   = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # Freeze everything except the transformer
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    if pipe.text_encoder_3 is not None:
        pipe.text_encoder_3.requires_grad_(False)

    transformer.train()

    return transformer, vae, pipe, scheduler


# ─────────────────────────────────────────────────────────────────────────────
# SD3 forward-pass helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_text_sd3(pipe, prompts, device):
    """
    Encode prompts through all SD3 text encoders (CLIP-L + CLIP-G + T5).
    Uses the pipeline's encode_prompt to handle the 3-encoder logic correctly.

    Returns
    -------
    prompt_embeds        : [B, seq_len, 4096]  (joint CLIP+T5 embeddings)
    pooled_prompt_embeds : [B, 2048]           (CLIP-L + CLIP-G pooled)
    """
    with torch.no_grad():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            device=device,
        )
    return prompt_embeds, pooled_prompt_embeds


def encode_images_to_latents_sd3(vae, images, device, dtype):
    """
    VAE-encode images to SD3 scaled latents (frozen, no grad).
    SD3 uses 16-channel latents with a shift_factor before scaling.
    """
    with torch.no_grad():
        latents = vae.encode(images.to(device, dtype=dtype)).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents


def get_sigmas_for_timesteps(scheduler, indices, device, dtype, n_dim):
    """
    Get flow-matching sigma values for a batch of random timestep indices.
    Expands to match latent tensor dimensions for broadcasting.
    """
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    sigma  = sigmas[indices]
    for _ in range(n_dim - 1):
        sigma = sigma.unsqueeze(-1)
    return sigma


def transformer_forward_sd3(transformer, noisy_latents, timesteps,
                             prompt_embeds, pooled_prompt_embeds):
    """Single SD3 transformer forward pass (predicts velocity)."""
    return transformer(
        hidden_states      = noisy_latents,
        timestep           = timesteps,
        encoder_hidden_states   = prompt_embeds,
        pooled_projections = pooled_prompt_embeds,
        return_dict        = False,
    )[0]


def compute_retain_loss_sd3(
    transformer, vae, scheduler,
    pipe,
    images, prompts,
    device, image_size, dtype,
):
    """
    Flow-matching denoising loss on the retain set.
    Loss target = velocity = (noise - latents).
    Gradients flow only through the transformer.
    """
    latents = encode_images_to_latents_sd3(vae, images, device, dtype)
    B       = latents.shape[0]
    noise   = torch.randn_like(latents)

    # Sample random timestep indices and get corresponding sigmas
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)
    indices   = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
    timesteps = scheduler.timesteps[indices]
    sigmas    = get_sigmas_for_timesteps(scheduler, indices, device, dtype, latents.ndim)

    noisy = (1.0 - sigmas) * latents + sigmas * noise

    prompt_embeds, pooled = encode_text_sd3(pipe, prompts, device)

    pred   = transformer_forward_sd3(transformer, noisy, timesteps, prompt_embeds, pooled)
    target = noise - latents  # flow matching velocity target
    return F.mse_loss(pred.float(), target.float())


# ─────────────────────────────────────────────────────────────────────────────
# NSFW datasets  (identical to SDXL version — no model dependency)
# ─────────────────────────────────────────────────────────────────────────────

class NSFWDataset(Dataset):
    def __init__(self, img_dir, transform, caption="a photo of a nude person"):
        self.all_imgs = sorted(
            glob.glob(os.path.join(img_dir, "**/*.png"),  recursive=True) +
            glob.glob(os.path.join(img_dir, "**/*.jpg"),  recursive=True) +
            glob.glob(os.path.join(img_dir, "**/*.jpeg"), recursive=True)
        )
        self.captions  = [c.strip() for c in caption.split(",")]
        self.transform = transform

    def __len__(self): return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = self.all_imgs[idx]
        for _ in range(10):
            try:
                image = Image.open(img_path).convert("RGB"); break
            except Exception:
                idx = random.randint(0, len(self.all_imgs) - 1)
                img_path = self.all_imgs[idx]
        cap_idx = int(os.path.basename(img_path).split("_")[0]) % len(self.captions)
        return self.transform(image), self.captions[cap_idx]


class NotNSFWDataset(Dataset):
    def __init__(self, img_dir, transform, caption="a photo of a person wearing clothes"):
        self.all_imgs = sorted(
            glob.glob(os.path.join(img_dir, "*.png"))  +
            glob.glob(os.path.join(img_dir, "*.jpg"))  +
            glob.glob(os.path.join(img_dir, "*.jpeg"))
        )
        self.caption   = caption
        self.transform = transform

    def __len__(self): return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = self.all_imgs[idx]
        for _ in range(10):
            try:
                image = Image.open(img_path).convert("RGB"); break
            except Exception:
                idx = random.randint(0, len(self.all_imgs) - 1)
                img_path = self.all_imgs[idx]
        return self.transform(image), self.caption


def setup_nsfw_data(batch_size, forget_path, remain_path, image_size=1024,
                    interpolation="bicubic", num_workers=8):
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
