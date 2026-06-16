"""
SDXL/MUKSB_nsfw.py
===================
MUKSB (Kalai-Smorodinsky Bargaining) NSFW concept unlearning for
Stable Diffusion XL (HuggingFace Diffusers — no CompVis/LDM dependency).

Usage
-----
python MUKSB_nsfw.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --forget_path /storage/s25017/Datasets/NSFW_removal/nude \
    --remain_path /storage/s25017/Datasets/NSFW_removal/with_dress \
    --train_method full \
    --epochs 5 \
    --lr 1e-5 \
    --device 0
"""

import argparse
import gc
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from logger.logger import setup_logger
from train_scripts.dataset import (
    setup_sdxl_components,
    setup_nsfw_data,
    encode_text_sdxl,
    encode_images_to_latents,
    unet_forward,
    compute_retain_loss,
)
from mask_variants import build_mask_nsfw, MASK_VARIANT_CHOICES


# ─────────────────────────────────────────────────────────────────────────────
# KS bargaining core  (identical to SD version — pure gradient math)
# ─────────────────────────────────────────────────────────────────────────────

def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # torch.dot uses cuBLAS which is limited to INT_MAX (~2.1B) elements.
    # Element-wise multiply + sum handles arbitrarily large tensors (e.g. full SDXL UNet).
    return (a * b).sum()


def ks_step(gr_flat: torch.Tensor, gf_flat: torch.Tensor, eps: float = 1e-8):
    norm_gr = torch.clamp(gr_flat.norm(), min=1e-6)
    norm_gf = torch.clamp(gf_flat.norm(), min=1e-6)

    cos_phi = torch.clamp(
        _dot(gr_flat, gf_flat) / (norm_gr * norm_gf),
        -1.0 + eps, 1.0 - eps,
    )

    g_hat_r  = gr_flat / norm_gr
    g_hat_f  = gf_flat / norm_gf
    g_sum    = g_hat_r + g_hat_f
    norm_sum = g_sum.norm()

    if norm_sum < 1e-6:
        zero = torch.zeros_like(gr_flat)
        return (
            torch.tensor(0.0, device=gr_flat.device),
            cos_phi,
            zero,
            torch.tensor(0.0, device=gr_flat.device),
        )

    g_star          = g_sum / norm_sum
    effective_scale = 2.0 * norm_gr * norm_gf / (norm_gr + norm_gf)
    lambda_ks       = _dot(g_hat_r, g_star)

    return lambda_ks, cos_phi, g_star, effective_scale


def _flatten_grads(params, grads):
    return torch.cat([
        g.detach().view(-1) if g is not None
        else torch.zeros(p.numel(), device=p.device)
        for p, g in zip(params, grads)
    ])


def _unpack_to_grads(params, flat_vec: torch.Tensor):
    offset = 0
    for p in params:
        n     = p.numel()
        p.grad = flat_vec[offset: offset + n].view(p.shape).clone()
        offset += n


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def l1_regularization(parameters):
    # Per-param sum avoids building a 2.57B-element flat tensor (INT_MAX exceeded).
    return sum(p.abs().sum() for p in parameters)


def select_parameters(unet, train_method):
    """Select trainable UNet parameters by method name.

    SDXL UNet attention naming mirrors SD1.x:
      attn1 = self-attention, attn2 = cross-attention.
    """
    parameters = []
    for name, param in unet.named_parameters():
        keep = False
        if train_method == "full":
            keep = True
        elif train_method == "xattn":
            keep = "attn2" in name
        elif train_method == "selfattn":
            keep = "attn1" in name
        elif train_method == "noxattn":
            keep = not ("attn2" in name or "time_embedding" in name)
        elif train_method == "notime":
            keep = "time_embedding" not in name
        if keep:
            parameters.append(param)
    return parameters


def save_model(unet, name, num, output_dir="models", logger=None):
    """Save the fine-tuned UNet in Diffusers format."""
    epoch_tag = f"-epoch_{num}" if num is not None else ""
    save_path = os.path.join(output_dir, name, f"unet{epoch_tag}")
    os.makedirs(save_path, exist_ok=True)
    unet.save_pretrained(save_path)
    if logger:
        logger.info(f"UNet saved → {save_path}")


def save_history(losses, name, output_dir="models"):
    folder = os.path.join(output_dir, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "loss.txt"), "w") as f:
        f.writelines([str(v) + "\n" for v in losses])
    if len(losses) >= 3:
        v = np.convolve(losses, np.ones(3) / 3, mode="valid")
    else:
        v = losses
    plt.figure()
    plt.plot(v, label="nsfw_loss")
    plt.legend(loc="upper left")
    plt.title("MUKSB SDXL training loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "loss.png"))
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main unlearning function
# ─────────────────────────────────────────────────────────────────────────────

def MUKSB(
    model_id,
    train_method,
    batch_size,
    epochs,
    lr,
    mask_variant,
    mask_density,
    lambda_tradeoff,
    device,
    image_size,
    with_l1,
    alpha,
    beta,
    forget_path,
    remain_path,
    dtype,
    logger,
):
    total_start = time.time()
    logger.info("======== MUKSB SDXL NSFW TRAINING STARTED ========")
    logger.info(f"model_id={model_id}  train_method={train_method}  dtype={dtype}")

    # ── load model components ────────────────────────────────────────────────
    logger.info(f"Loading SDXL from: {model_id}")
    unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, scheduler = \
        setup_sdxl_components(model_id, device, dtype=dtype)

    logger.info(f"UNet params total: {sum(p.numel() for p in unet.parameters()):,}")

    # ── data ─────────────────────────────────────────────────────────────────
    forget_dl, remain_dl = setup_nsfw_data(
        batch_size=batch_size,
        forget_path=forget_path,
        remain_path=remain_path,
        image_size=image_size,
    )
    num_forget = len(forget_dl.dataset)
    logger.info(f"Forget: {num_forget} samples | Remain: {len(remain_dl.dataset)} samples")

    # ── parameter selection ──────────────────────────────────────────────────
    parameters = select_parameters(unet, train_method)
    logger.info(f"Trainable params ({train_method}): {sum(p.numel() for p in parameters):,}")

    # ── importance mask (optional) ───────────────────────────────────────────
    if mask_variant is not None:
        logger.info(f"Building mask: variant={mask_variant}  density={mask_density}")
        unet.eval()
        mask = build_mask_nsfw(
            variant         = mask_variant,
            unet            = unet,
            vae             = vae,
            scheduler       = scheduler,
            tokenizer       = tokenizer,
            tokenizer_2     = tokenizer_2,
            text_encoder    = text_encoder,
            text_encoder_2  = text_encoder_2,
            parameters      = parameters,
            forget_dl       = forget_dl,
            remain_dl       = remain_dl,
            beta            = beta,
            device          = device,
            image_size      = image_size,
            dtype           = dtype,
            target_density  = mask_density,
            max_batches     = len(forget_dl),
            lambda_tradeoff = lambda_tradeoff,
            logger          = logger,
        )
        active = mask.sum().item()
        total  = mask.numel()
        logger.info(f"[Mask] active={active:,}/{total:,}  density={active/total:.4f}")
        run_tag = (
            f"sdxl-nsfw-MUKSB-{mask_variant}"
            f"-rho{int(mask_density*100)}pct"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}"
        )
    else:
        mask    = None
        run_tag = (
            f"sdxl-nsfw-MUKSB"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}"
        )

    logger.info(f"Run tag: {run_tag}")

    # ── training ─────────────────────────────────────────────────────────────
    unet.train()
    optimizer     = torch.optim.Adam(parameters, lr=lr)
    criteria      = torch.nn.MSELoss()
    losses        = []
    step          = 0
    skipped_steps = 0
    total_steps   = 0
    cos_phi_hist  = []

    word_wear = "a photo of a person wearing clothes"

    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch + 1}/{epochs} started")
        remain_iter = iter(remain_dl)

        with tqdm(total=len(forget_dl), desc=f"Epoch {epoch + 1}") as pbar:
            for forget_images, forget_captions in forget_dl:
                unet.train()
                total_steps += 1

                try:
                    remain_images, remain_captions = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_images, remain_captions = next(remain_iter)

                forget_images = forget_images.to(device)
                remain_images = remain_images.to(device)
                B             = forget_images.shape[0]

                # ── retain loss ───────────────────────────────────────────────
                loss_r = compute_retain_loss(
                    unet, vae, scheduler,
                    tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                    remain_images, list(remain_captions),
                    device, image_size, dtype,
                )

                # ── forget loss ───────────────────────────────────────────────
                # Same image → same latent for forget and pseudo passes;
                # only text conditioning differs.
                latents = encode_images_to_latents(vae, forget_images, device, dtype)
                noise   = torch.randn_like(latents)
                t       = torch.randint(
                    0, scheduler.config.num_train_timesteps, (B,), device=device
                )
                noisy = scheduler.add_noise(latents, noise, t)

                f_enc_h, f_pooled = encode_text_sdxl(
                    tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                    list(forget_captions), device,
                )
                p_enc_h, p_pooled = encode_text_sdxl(
                    tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                    [word_wear] * B, device,
                )

                forget_out = unet_forward(unet, noisy, t, f_enc_h, f_pooled, image_size)
                with torch.no_grad():
                    pseudo_out = unet_forward(unet, noisy, t, p_enc_h, p_pooled, image_size)

                loss_u = criteria(forget_out, pseudo_out.detach()) * beta

                # ── compute separate gradients for KS bargaining ──────────────
                # Compute forget grads first (no graph retention needed),
                # then retain grads.
                grads_f = torch.autograd.grad(
                    loss_u, parameters, retain_graph=False, allow_unused=True
                )
                grads_r = torch.autograd.grad(
                    loss_r, parameters, retain_graph=False, allow_unused=True
                )

                gr_flat = _flatten_grads(parameters, grads_r)
                gf_flat = _flatten_grads(parameters, grads_f)
                del grads_r, grads_f

                # ── project onto masked subspace if mask provided ──────────────
                gr_input = gr_flat[mask] if mask is not None else gr_flat
                gf_input = gf_flat[mask] if mask is not None else gf_flat

                # ── KS bargaining ─────────────────────────────────────────────
                lambda_ks, cos_phi, g_star, effective_scale = ks_step(gr_input, gf_input)
                cos_phi_hist.append(cos_phi.item())

                if torch.norm(g_star).item() < 1e-6:
                    skipped_steps += 1
                    logger.debug(
                        f"step={step}: anti-parallel gradients "
                        f"(cos_φ={cos_phi.item():.3f}), skipping update"
                    )
                    del gr_flat, gf_flat, gr_input, gf_input, g_star
                    pbar.update(1)
                    continue

                g_star_scaled = effective_scale * g_star
                del gr_input, gf_input

                if mask is not None:
                    update_full          = torch.zeros_like(gr_flat)
                    update_full[mask]    = g_star_scaled
                else:
                    update_full = g_star_scaled

                del gr_flat, gf_flat, g_star, g_star_scaled

                optimizer.zero_grad()
                _unpack_to_grads(parameters, update_full)
                del update_full

                if with_l1:
                    current_alpha = alpha * (1 - epoch / epochs)
                    l1_loss  = current_alpha * l1_regularization(parameters)
                    l1_grads = torch.autograd.grad(l1_loss, parameters)
                    for p, lg in zip(parameters, l1_grads):
                        if p.grad is not None and lg is not None:
                            p.grad += lg.detach()

                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                combined = loss_r + loss_u
                losses.append(combined.item() / batch_size)
                step += 1

                if step % 10 == 0:
                    avg_cos   = float(np.mean(cos_phi_hist[-10:])) if cos_phi_hist else 0.0
                    skip_rate = skipped_steps / max(total_steps, 1)
                    logger.info(
                        f"step={step}"
                        f"  λ_KS={lambda_ks.item():.4f}"
                        f"  cos_φ={cos_phi.item():.4f}"
                        f"  avg_cos_φ(10)={avg_cos:.4f}"
                        f"  eff_scale={effective_scale.item():.4e}"
                        f"  skip_rate={skip_rate:.3f}"
                        f"  loss_r={loss_r.item():.4f}"
                        f"  loss_u={loss_u.item():.4f}"
                    )
                    save_history(losses, run_tag)

                pbar.set_postfix(
                    loss_r=f"{loss_r.item():.4f}",
                    loss_u=f"{loss_u.item():.4f}",
                    lam=f"{lambda_ks.item():.3f}",
                    cos=f"{cos_phi.item():.2f}",
                )
                pbar.update(1)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1} done | {epoch_time:.1f}s ({epoch_time/60:.2f} min) | "
            f"skip_rate={skipped_steps/max(total_steps,1):.3f}"
        )

        unet.eval()
        if (epoch + 1) % 1 == 0 and epoch != epochs - 1:
            save_model(unet, run_tag, epoch + 1, logger=logger)
        torch.cuda.empty_cache()
        gc.collect()

    # ── final save ───────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    logger.info("======== MUKSB SDXL NSFW TRAINING FINISHED ========")
    logger.info(
        f"Total: {total_time:.1f}s ({total_time/60:.2f} min | {total_time/3600:.2f} hrs)"
    )
    if cos_phi_hist:
        logger.info(
            f"cos_φ stats: mean={np.mean(cos_phi_hist):.4f}  "
            f"min={np.min(cos_phi_hist):.4f}  "
            f"max={np.max(cos_phi_hist):.4f}"
        )
    logger.info(
        f"Anti-parallel skips: {skipped_steps}/{total_steps} "
        f"({skipped_steps/max(total_steps,1)*100:.1f}%)"
    )

    unet.eval()
    save_model(unet, run_tag, epochs, logger=logger)
    save_history(losses, run_tag)
    logger.info(f"Model saved under: models/{run_tag}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _mask_type(v):
        return None if v in (None, "None", "none") else str(v)

    parser = argparse.ArgumentParser(
        description="MUKSB: KS-Bargaining NSFW unlearning for Stable Diffusion XL"
    )
    parser.add_argument("--model_id",        type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="HuggingFace model ID or local path to SDXL")
    parser.add_argument("--train_method",    type=str, default="full",
                        choices=["full", "xattn", "selfattn", "noxattn", "notime"])
    parser.add_argument("--batch_size",      type=int,   default=1)
    parser.add_argument("--epochs",          type=int,   default=15)
    parser.add_argument("--lr",              type=float, default=1e-5)
    parser.add_argument("--mask_variant",    type=_mask_type, default=None,
                        choices=list(MASK_VARIANT_CHOICES) + [None],
                        help="Parameter saliency mask: random / forget_fisher / salun / dual_fisher")
    parser.add_argument("--mask_density",    type=float, default=0.5,
                        help="Fraction ρ of parameters to update when using a mask")
    parser.add_argument("--lambda_tradeoff", type=float, default=1.0,
                        help="λ in dual_fisher score (S_diff = F̂_f − λ·F̂_r)")
    parser.add_argument("--device",         type=str,   default="5")
    parser.add_argument("--image_size",     type=int,   default=512,
                        help="Training resolution (1024 = SDXL native; use 512 to save memory)")
    parser.add_argument("--with_l1",        action="store_true", default=False)
    parser.add_argument("--alpha",          type=float, default=1e-4,
                        help="L1 regularisation coefficient")
    parser.add_argument("--beta",           type=float, default=100.0,
                        help="Scale factor for the forget loss")
    parser.add_argument("--forget_path",    type=str,
                        default="/storage/s25017/Datasets/NSFW_removal/nude")
    parser.add_argument("--remain_path",    type=str,
                        default="/storage/s25017/Datasets/NSFW_removal/with_dress")
    parser.add_argument("--dtype",          type=str,   default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (bfloat16 recommended for training stability)")
    args = parser.parse_args()

    logger, log_file = setup_logger(name="MUKSB_sdxl_nsfw")
    logger.info("======== MUKSB SDXL NSFW STARTED ========")
    logger.info(f"Log: {log_file}")
    logger.info(f"Args: {vars(args)}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    _dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype  = _dtype_map[args.dtype]
    device = f"cuda:{int(args.device)}"

    MUKSB(
        model_id        = args.model_id,
        train_method    = args.train_method,
        batch_size      = args.batch_size,
        epochs          = args.epochs,
        lr              = args.lr,
        mask_variant    = args.mask_variant,
        mask_density    = args.mask_density,
        lambda_tradeoff = args.lambda_tradeoff,
        device          = device,
        image_size      = args.image_size,
        with_l1         = args.with_l1,
        alpha           = args.alpha,
        beta            = args.beta,
        forget_path     = args.forget_path,
        remain_path     = args.remain_path,
        dtype           = dtype,
        logger          = logger,
    )
