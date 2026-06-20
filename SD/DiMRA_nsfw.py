"""
DiMRA_nsfw.py — Diffusion Model Relearning Attack (DiMRA) for NSFW unlearning.
STANDALONE: depends only on torch / diffusers / transformers / PIL (no repo imports).

Reference: "Towards Irreversible Machine Unlearning for Diffusion Models"
           (arXiv:2512.03564).

Idea
----
Unlearning only nudges the UNet weights a small distance from the original
SD-v1.4 minimum. Fine-tuning the *unlearned* model with the ordinary denoising
loss on a BENIGN auxiliary set -- one that contains NONE of the forgotten
concept -- pulls the weights back toward the original minimum and reawakens the
erased capability. The attacker never sees forget-concept data and never needs
to know what was erased; it only needs the unlearned weights and the prompt
format.

    L_DiMRA(theta) = E_{t, (x0,c)~D_aux, eps} [ || eps - eps_theta(x_t | c) ||^2 ]

This script runs in the diffusers framework because the unlearned MUKSB NSFW
checkpoint is saved only as a diffusers UNet state_dict. It loads VAE /
tokenizer / text-encoder / UNet the same way the eval generation script does,
fine-tunes the UNet on the auxiliary set, and saves periodic checkpoints in the
SAME diffusers .pt format so each one is drop-in loadable by the existing
generation/eval pipeline.

Example
-------
python DiMRA_nsfw.py \
  --unlearned_ckpt models/compvis-nsfw-MUKSB-g0.5-method_full-lr_1e-05_E5_U800_MAGNITUDE/diffusers-nsfw-MUKSB-g0.5-method_full-lr_1e-05_E5_U800_MAGNITUDE-epoch_1.pt \
  --aux_path /storage/s25017/Datasets/NSFW_removal/with_dress \
  --lr 1e-5 --max_steps 2000 --batch_size 4 --save_every 250 --device 0
"""

import argparse
import glob
import logging
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

# SD v1.x VAE latent scaling factor.
LATENT_SCALE = 0.18215


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_dir="logs", name="DiMRA_nsfw"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{ts}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt)
    ch = logging.StreamHandler();        ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger, log_path


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary data — benign, forget-concept-FREE (the DiMRA requirement)
# ─────────────────────────────────────────────────────────────────────────────

def get_transform(size=512):
    return T.Compose([
        T.Resize(size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        lambda im: im.convert("RGB"),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # -> [-1, 1]
    ])


class AuxImageDataset(Dataset):
    """Benign images from a flat directory, all captioned with one prompt.

    Returns (CHW float tensor in [-1, 1], caption_str).
    """

    def __init__(self, img_dir, transform, caption):
        self.all_imgs = (
            glob.glob(os.path.join(img_dir, "*.png")) +
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.jpeg"))
        )
        if not self.all_imgs:
            raise RuntimeError(f"No images found under {img_dir}")
        self.transform = transform
        self.caption = caption

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        name = self.all_imgs[idx]
        for attempt in range(10):
            try:
                img = Image.open(name).convert("RGB")
                break
            except Exception:
                if attempt == 9:
                    raise RuntimeError(f"Failed to load image: {name}")
                idx = random.randint(0, len(self.all_imgs) - 1)
                name = self.all_imgs[idx]
        return self.transform(img), self.caption


def build_aux_loader(aux_path, caption, batch_size, image_size, num_workers):
    ds = AuxImageDataset(aux_path, get_transform(image_size), caption)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None, drop_last=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (mirrors the diffusers load recipe used for generation)
# ─────────────────────────────────────────────────────────────────────────────

def load_unlearned_pipeline(unlearned_ckpt, sd_id, device, logger):
    """Load frozen VAE + CLIP text encoder and the unlearned UNet (theta_u)."""
    logger.info(f"Loading frozen VAE / tokenizer / text-encoder from {sd_id}")
    vae = AutoencoderKL.from_pretrained(sd_id, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    logger.info(f"Loading base UNet from {sd_id}, then unlearned weights")
    unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")

    if not os.path.exists(unlearned_ckpt):
        raise FileNotFoundError(f"Unlearned checkpoint not found: {unlearned_ckpt}")
    state_dict = torch.load(unlearned_ckpt, map_location="cpu")
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded theta_u | missing={len(missing)} unexpected={len(unexpected)}")

    # Freeze everything except the UNet (the attack updates eps_theta only).
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    vae.to(device).eval()
    text_encoder.to(device).eval()
    unet.to(device).train()
    return vae, tokenizer, text_encoder, unet


def encode_text(tokenizer, text_encoder, captions, device):
    tok = tokenizer(
        list(captions), padding="max_length",
        max_length=tokenizer.model_max_length, truncation=True,
        return_tensors="pt",
    )
    return text_encoder(tok.input_ids.to(device))[0]


# ─────────────────────────────────────────────────────────────────────────────
# Save helper — diffusers UNet state_dict (.pt), same format as the unlearned ckpt
# ─────────────────────────────────────────────────────────────────────────────

def save_unet(unet, folder, file_stub, label, logger):
    """Save a diffusers UNet state_dict (identical format to the unlearned ckpt)."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{file_stub}-{label}.pt")
    torch.save(unet.state_dict(), path)
    logger.info(f"[save] {label} -> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# DiMRA relearning attack
# ─────────────────────────────────────────────────────────────────────────────

def dimra(
    unlearned_ckpt, aux_path, caption, sd_id, lr, epochs, batch_size,
    image_size, save_every, num_workers, out_dir, device, logger,
):
    t0 = time.time()
    logger.info("======== DiMRA RELEARNING ATTACK STARTED ========")

    vae, tokenizer, text_encoder, unet = load_unlearned_pipeline(
        unlearned_ckpt, sd_id, device, logger
    )
    unet.enable_gradient_checkpointing()

    scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    aux_dl = build_aux_loader(aux_path, caption, batch_size, image_size, num_workers)
    steps_per_epoch = len(aux_dl)
    total_steps = steps_per_epoch * epochs
    logger.info(f"Auxiliary set: {len(aux_dl.dataset)} images from {aux_path}")
    logger.info(f"Caption: '{caption}'  | aux MUST contain none of the forget concept")
    logger.info(f"Epochs={epochs} | steps/epoch={steps_per_epoch} | total steps={total_steps}")

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    # Mirror the unlearned model's layout so the result drops straight into the
    # generation/eval pipeline: a folder named compvis-... holding a
    # diffusers-...-epoch_{N}.pt UNet state_dict (same as the unlearned ckpt).
    aux_tag = os.path.basename(aux_path.rstrip("/"))
    run_tag = f"compvis-nsfw-MUKSB-DiMRA-aux_{aux_tag}-lr_{lr}-E{epochs}"
    folder = os.path.join(out_dir, run_tag)
    file_stub = run_tag.replace("compvis-", "diffusers-", 1)
    logger.info(f"run_tag = {run_tag}")
    logger.info(f"Saving to: {folder}/{file_stub}-epoch_*.pt")

    # Step-0 (pre-attack) baseline so recovery can be measured from the start.
    save_unet(unet, folder, file_stub, "step_0", logger)

    step = 0
    final_path = None
    pbar = tqdm(total=total_steps, desc="DiMRA")
    for epoch in range(epochs):
        for imgs, captions in aux_dl:
            imgs = imgs.to(device, non_blocking=True)  # NCHW in [-1, 1]

            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample() * LATENT_SCALE
                enc = encode_text(tokenizer, text_encoder, captions, device)

            noise = torch.randn_like(latents)
            t = torch.randint(0, scheduler.config.num_train_timesteps,
                              (latents.shape[0],), device=device).long()
            noisy = scheduler.add_noise(latents, noise, t)

            pred = unet(noisy, t, encoder_hidden_states=enc).sample
            loss = F.mse_loss(pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % 20 == 0:
                logger.info(f"epoch={epoch+1}/{epochs} step={step}/{total_steps}  loss={loss.item():.4f}")
            if save_every > 0 and step % save_every == 0:
                unet.eval()
                save_unet(unet, folder, file_stub, f"step_{step}", logger)
                unet.train()
        # End-of-epoch checkpoint, named like the unlearned model (epoch_{N}).
        unet.eval()
        final_path = save_unet(unet, folder, file_stub, f"epoch_{epoch+1}", logger)
        unet.train()
    pbar.close()

    dt = time.time() - t0
    logger.info(f"======== DiMRA FINISHED in {dt/60:.2f} min | {epochs} epoch(s), {step} steps ========")
    logger.info(f"FINAL relearned model: {final_path}")
    logger.info("Generate with it exactly like the unlearned model: "
                f"--model_name {final_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger, log_file = setup_logger(name="DiMRA_nsfw")

    p = argparse.ArgumentParser(
        description="DiMRA: relearning attack on an unlearned NSFW SD model"
    )
    p.add_argument("--unlearned_ckpt", type=str,
                   default="models/compvis-nsfw-MUKSB-g0.5-method_full-lr_1e-05_E5_U800_MAGNITUDE/"
                           "diffusers-nsfw-MUKSB-g0.5-method_full-lr_1e-05_E5_U800_MAGNITUDE-epoch_1.pt",
                   help="Diffusers-format unlearned UNet state_dict (theta_u)")
    p.add_argument("--aux_path", type=str,
                   default="/storage/s25017/Datasets/NSFW_removal/with_dress",
                   help="Benign auxiliary set — MUST contain none of the forget concept")
    p.add_argument("--caption", type=str,
                   default="a photo of a person wearing clothes")
    p.add_argument("--sd_id", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1,
                   help="Passes over the auxiliary set (mirror the unlearning run)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--save_every", type=int, default=50,
                   help="Intra-epoch checkpoint every N steps (0 disables; "
                        "an end-of-epoch checkpoint is always saved)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="models",
                   help="A run folder (compvis-nsfw-MUKSB-DiMRA-...) is created here")
    p.add_argument("--device", type=str, default="0")
    args = p.parse_args()

    logger.info(f"Log file : {log_file}")
    logger.info(f"Args     : {vars(args)}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    dimra(
        unlearned_ckpt = args.unlearned_ckpt,
        aux_path       = args.aux_path,
        caption        = args.caption,
        sd_id          = args.sd_id,
        lr             = args.lr,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        image_size     = args.image_size,
        save_every     = args.save_every,
        num_workers    = args.num_workers,
        out_dir        = args.out_dir,
        device         = f"cuda:{int(args.device)}",
        logger         = logger,
    )
