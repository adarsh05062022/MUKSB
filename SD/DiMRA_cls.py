"""
DiMRA_cls.py — Diffusion Model Relearning Attack (DiMRA) for Imagenette-10
class-removal unlearning.  STANDALONE (torch / diffusers / transformers / PIL /
torchvision only — no repo imports).

Reference: "Towards Irreversible Machine Unlearning for Diffusion Models"
           (arXiv:2512.03564).

Why the class setting?
----------------------
For NSFW the erased concept is diffuse (it is spread over skin, colour, pose,
composition...), so almost any benign fine-tuning reawakens it and the attack is
hard to interpret. A discrete Imagenette class (e.g. "tench", "church") is a
clean, classifier-measurable concept — this is exactly how the DiMRA paper
evaluates (CIFAR-10 / UnlearnCanvas class removal).

Attack
------
Take the unlearned class model theta_u (it erased class c). Fine-tune it with the
ordinary denoising loss on an AUXILIARY set that contains NONE of class c — here
the Imagenette *retain* set (the other 9 classes), each captioned with its true
class. Because unlearning only nudged the weights a small distance from the
original SD-v1.4 minimum, fine-tuning on retain data pulls them back and the
erased class reappears when you then condition generation on class c's prompt.

    L_DiMRA(theta) = E_{t, (x0,c)~D_aux, eps} [ || eps - eps_theta(x_t | c) ||^2 ]

The attacker never sees class-c images and never needs to know which class was
erased; it only needs the unlearned weights and the prompt format
("an image of a {class}").

Runs in the diffusers framework (the unlearned class checkpoint is a diffusers
UNet state_dict). Saves periodic checkpoints in the SAME diffusers .pt layout as
the unlearned model so each is drop-in loadable by the existing generation/eval
pipeline.

Example
-------
python DiMRA_cls.py \
  --class_to_forget 0 \
  --unlearned_ckpt models/compvis-cls_0-MUKSB-g0.5-method_full-lr_5e-06_E10_U963_/diffusers-cls_0-MUKSB-g0.5-method_full-lr_5e-06_E10_U963_-epoch_5.pt \
  --lr 1e-5 --epochs 1 --batch_size 8 --device 0
"""

import argparse
import logging
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision.transforms as TT
from PIL import ImageFile
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Imagenette
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

def setup_logger(log_dir="logs", name="DiMRA_cls"):
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
# Auxiliary data — Imagenette RETAIN set (the 9 non-forgotten classes).
# Contains none of the forgotten class -> the DiMRA requirement.
# ─────────────────────────────────────────────────────────────────────────────

def get_transform(size=256):
    return TT.Compose([
        TT.Resize(size, interpolation=InterpolationMode.BICUBIC),
        TT.CenterCrop(size),
        lambda im: im.convert("RGB"),
        TT.ToTensor(),
        TT.Normalize([0.5], [0.5]),  # -> [-1, 1]
    ])


def imagenette_descriptions(dataset):
    """['an image of a tench', 'an image of a English springer', ...] — matches
    the caption format used during unlearning (MUKSB_cls.py)."""
    names = []
    for cls in dataset.classes:
        names.append(cls[0] if isinstance(cls, (tuple, list)) else cls)
    return [f"an image of a {n}" for n in names]


def build_retain_loader(root, class_to_forget, batch_size, image_size,
                        num_workers):
    base = Imagenette(root=root, split="train",
                      transform=get_transform(image_size), download=False)
    descriptions = imagenette_descriptions(base)
    retain_idx = [i for i, s in enumerate(base._samples) if s[1] != class_to_forget]
    loader = DataLoader(
        Subset(base, retain_idx), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None, drop_last=True,
    )
    return loader, descriptions


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (diffusers UNet state_dict, same recipe as generation)
# ─────────────────────────────────────────────────────────────────────────────

def load_unlearned_pipeline(unlearned_ckpt, sd_id, device, logger):
    """Frozen VAE + CLIP text encoder + the unlearned UNet (theta_u)."""
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
# Save helper — diffusers UNet state_dict (.pt), same layout as the unlearned ckpt
# ─────────────────────────────────────────────────────────────────────────────

def save_unet(unet, folder, file_stub, label, logger):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{file_stub}-{label}.pt")
    torch.save(unet.state_dict(), path)
    logger.info(f"[save] {label} -> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# DiMRA relearning attack (class removal)
# ─────────────────────────────────────────────────────────────────────────────

def dimra(
    class_to_forget, unlearned_ckpt, imagenette_root, sd_id, lr, max_steps,
    batch_size, image_size, save_every, num_workers, out_dir, device, logger,
):
    t0 = time.time()
    logger.info("======== DiMRA CLASS RELEARNING ATTACK STARTED ========")
    logger.info(f"class_to_forget = {class_to_forget}")

    vae, tokenizer, text_encoder, unet = load_unlearned_pipeline(
        unlearned_ckpt, sd_id, device, logger
    )
    unet.enable_gradient_checkpointing()

    scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    aux_dl, descriptions = build_retain_loader(
        imagenette_root, class_to_forget, batch_size, image_size, num_workers
    )
    forget_desc = descriptions[class_to_forget]
    logger.info(f"Auxiliary RETAIN set: {len(aux_dl.dataset)} images "
                f"(all Imagenette classes except #{class_to_forget})")
    logger.info(f"Forgotten class prompt (NOT in aux): '{forget_desc}'")
    logger.info(f"max_steps={max_steps} | batch_size={batch_size} | "
                f"mixed shuffled batches from the retain set "
                f"(~{max_steps * batch_size} images seen, << one epoch)")

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    # Mirror the unlearned model's layout: a compvis-... folder holding a
    # diffusers-...-step_{N}.pt UNet state_dict (same format as the unlearned ckpt).
    run_tag = f"compvis-cls_{class_to_forget}-MUKSB-DiMRA-aux_retain-lr_{lr}-S{max_steps}"
    folder = os.path.join(out_dir, run_tag)
    file_stub = run_tag.replace("compvis-", "diffusers-", 1)
    logger.info(f"run_tag = {run_tag}")
    logger.info(f"Saving to: {folder}/{file_stub}-step_*.pt")

    # Step-0 (pre-attack) baseline so recovery can be measured from the start.
    save_unet(unet, folder, file_stub, "step_0", logger)

    aux_iter = iter(aux_dl)
    step = 0
    pbar = tqdm(total=max_steps, desc="DiMRA-cls")
    while step < max_steps:
        try:
            imgs, labels = next(aux_iter)
        except StopIteration:               # cycle the loader (mixed reshuffle)
            aux_iter = iter(aux_dl)
            imgs, labels = next(aux_iter)

        imgs = imgs.to(device, non_blocking=True)  # NCHW in [-1, 1]
        captions = [descriptions[int(l)] for l in labels]

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
            logger.info(f"step={step}/{max_steps}  loss={loss.item():.4f}")
        # Intra-run checkpoints for the recovery curve (skip max_steps; saved below).
        if save_every > 0 and step % save_every == 0 and step < max_steps:
            unet.eval()
            save_unet(unet, folder, file_stub, f"step_{step}", logger)
            unet.train()
    pbar.close()

    # Final relearned model.
    unet.eval()
    final_path = save_unet(unet, folder, file_stub, f"step_{step}", logger)

    dt = time.time() - t0
    logger.info(f"======== DiMRA FINISHED in {dt/60:.2f} min | {step} steps ========")
    logger.info(f"FINAL relearned model: {final_path}")
    logger.info("Generate with it exactly like the unlearned class model "
                f"(condition on '{forget_desc}' and check if class #{class_to_forget} returns).")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger, log_file = setup_logger(name="DiMRA_cls")

    p = argparse.ArgumentParser(
        description="DiMRA: relearning attack on an unlearned Imagenette class SD model"
    )
    p.add_argument("--class_to_forget", type=int, default=0,
                   help="Imagenette class index that was erased (0–9)")
    p.add_argument("--unlearned_ckpt", type=str,
                   default="models/compvis-cls_0-MUKSB-g0.5-method_full-lr_5e-06_E10_U963_/"
                           "diffusers-cls_0-MUKSB-g0.5-method_full-lr_5e-06_E10_U963_-epoch_5.pt",
                   help="Diffusers-format unlearned UNet state_dict (theta_u) for that class")
    p.add_argument("--imagenette_root", type=str, default="/storage/s25017/Datasets",
                   help="Root containing the imagenette2 folder")
    p.add_argument("--sd_id", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_steps", type=int, default=200,
                   help="Total gradient steps of mixed shuffled retain batches "
                        "(NOT a full epoch over the retain set)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256,
                   help="Match the resolution the class model was unlearned at (256)")
    p.add_argument("--save_every", type=int, default=50,
                   help="Intra-run checkpoint every N steps (0 disables; "
                        "the final step is always saved)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="models")
    p.add_argument("--device", type=str, default="0")
    args = p.parse_args()

    logger.info(f"Log file : {log_file}")
    logger.info(f"Args     : {vars(args)}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    dimra(
        class_to_forget = args.class_to_forget,
        unlearned_ckpt  = args.unlearned_ckpt,
        imagenette_root = args.imagenette_root,
        sd_id           = args.sd_id,
        lr              = args.lr,
        max_steps       = args.max_steps,
        batch_size      = args.batch_size,
        image_size      = args.image_size,
        save_every      = args.save_every,
        num_workers     = args.num_workers,
        out_dir         = args.out_dir,
        device          = f"cuda:{int(args.device)}",
        logger          = logger,
    )
