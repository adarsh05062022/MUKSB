"""
SD3/MUKSB_nsfw.py
==================
MUKSB (Kalai-Smorodinsky Bargaining) NSFW concept unlearning for
Stable Diffusion 3 (HuggingFace Diffusers, flow-matching).

Key differences vs SDXL version:
  - SD3Transformer2DModel  (DiT)  instead of UNet
  - Three text encoders; pipe.encode_prompt() handles the complexity
  - 16-channel VAE latents with shift_factor
  - Flow-matching noise: noisy = (1-σ)*latents + σ*noise
  - Forget loss compares velocity predictions (not noise predictions)
  - Retain loss target = noise - latents  (velocity)

Usage
-----
conda run -n munba3_sd3 python3 MUKSB_nsfw.py \
    --model_id stabilityai/stable-diffusion-3-medium-diffusers \
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
    setup_sd3_components,
    setup_nsfw_data,
    encode_text_sd3,
    encode_images_to_latents_sd3,
    transformer_forward_sd3,
    compute_retain_loss_sd3,
    get_sigmas_for_timesteps,
)
from mask_variants import build_mask_nsfw, MASK_VARIANT_CHOICES


# ─────────────────────────────────────────────────────────────────────────────
# KS bargaining core  (same as SDXL — pure gradient math)
# ─────────────────────────────────────────────────────────────────────────────

def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # torch.dot is limited to INT_MAX elements; element-wise handles arbitrary sizes.
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
        n      = p.numel()
        p.grad = flat_vec[offset: offset + n].view(p.shape).clone()
        offset += n


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def l1_regularization(parameters):
    return sum(p.abs().sum() for p in parameters)


def select_parameters(transformer, train_method):
    """
    Select trainable transformer parameters.

    SD3 JointTransformerBlock naming:
      attn.*         — joint self-attention (image + text streams)
      ff.net.*       — image-stream feed-forward
      ff_context.*   — text-stream feed-forward
      norm*          — layer norms

    SingleTransformerBlock naming:
      attn.*         — attention
      proj_mlp       — MLP projection
    """
    parameters = []
    for name, param in transformer.named_parameters():
        keep = False
        if train_method == "full":
            keep = True
        elif train_method == "attn":
            keep = "attn" in name
        elif train_method == "ff":
            keep = ("ff." in name or "ff_context" in name or "proj_mlp" in name)
        elif train_method == "joint_blocks":
            keep = "transformer_blocks." in name and "single_transformer_blocks" not in name
        elif train_method == "single_blocks":
            keep = "single_transformer_blocks." in name
        if keep:
            parameters.append(param)
    return parameters


def save_model(transformer, name, num, output_dir="models", logger=None):
    epoch_tag = f"-epoch_{num}" if num is not None else ""
    save_path = os.path.join(output_dir, name, f"transformer{epoch_tag}")
    os.makedirs(save_path, exist_ok=True)
    transformer.save_pretrained(save_path)
    if logger:
        logger.info(f"Transformer saved → {save_path}")


def save_history(losses, name, output_dir="models"):
    folder = os.path.join(output_dir, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "loss.txt"), "w") as f:
        f.writelines([str(v) + "\n" for v in losses])
    v = np.convolve(losses, np.ones(3) / 3, mode="valid") if len(losses) >= 3 else losses
    plt.figure()
    plt.plot(v, label="nsfw_loss")
    plt.legend(loc="upper left")
    plt.title("MUKSB SD3 training loss")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.tight_layout()
    plt.savefig(os.path.join(folder, "loss.png"))
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main unlearning function
# ─────────────────────────────────────────────────────────────────────────────

def MUKSB(
    model_id, train_method, batch_size, epochs, lr,
    mask_variant, mask_density, lambda_tradeoff,
    device, image_size, with_l1, alpha, beta,
    forget_path, remain_path, dtype, skip_t5, logger,
):
    total_start = time.time()
    logger.info("======== MUKSB SD3 NSFW TRAINING STARTED ========")
    logger.info(f"model_id={model_id}  train_method={train_method}  dtype={dtype}  skip_t5={skip_t5}")

    # ── load components ──────────────────────────────────────────────────────
    logger.info(f"Loading SD3 from: {model_id}")
    transformer, vae, pipe, scheduler = setup_sd3_components(
        model_id, device, dtype=dtype, skip_t5=skip_t5
    )
    logger.info(f"Transformer params: {sum(p.numel() for p in transformer.parameters()):,}")

    # ── data ─────────────────────────────────────────────────────────────────
    forget_dl, remain_dl = setup_nsfw_data(
        batch_size=batch_size, forget_path=forget_path,
        remain_path=remain_path, image_size=image_size,
    )
    num_forget = len(forget_dl.dataset)
    logger.info(f"Forget: {num_forget} | Remain: {len(remain_dl.dataset)}")

    # ── parameter selection ──────────────────────────────────────────────────
    parameters = select_parameters(transformer, train_method)
    logger.info(f"Trainable params ({train_method}): {sum(p.numel() for p in parameters):,}")

    # ── importance mask ──────────────────────────────────────────────────────
    if mask_variant is not None:
        logger.info(f"Building mask: variant={mask_variant}  density={mask_density}")
        transformer.eval()
        mask = build_mask_nsfw(
            variant         = mask_variant,
            transformer     = transformer,
            vae             = vae,
            scheduler       = scheduler,
            pipe            = pipe,
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
        logger.info(f"[Mask] active={active:,}/{mask.numel():,}  density={active/mask.numel():.4f}")
        run_tag = (
            f"sd3-nsfw-MUKSB-{mask_variant}"
            f"-rho{int(mask_density*100)}pct"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}"
        )
    else:
        mask    = None
        run_tag = f"sd3-nsfw-MUKSB-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}"

    logger.info(f"Run tag: {run_tag}")

    # ── training ─────────────────────────────────────────────────────────────
    transformer.train()
    optimizer     = torch.optim.Adam(parameters, lr=lr)
    criteria      = torch.nn.MSELoss()
    losses        = []
    step          = 0
    skipped_steps = 0
    total_steps   = 0
    cos_phi_hist  = []

    word_wear = "a photo of a person wearing clothes"
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)

    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch + 1}/{epochs} started")
        remain_iter = iter(remain_dl)

        with tqdm(total=len(forget_dl), desc=f"Epoch {epoch + 1}") as pbar:
            for forget_images, forget_captions in forget_dl:
                transformer.train()
                total_steps += 1

                try:
                    remain_images, remain_captions = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_images, remain_captions = next(remain_iter)

                forget_images = forget_images.to(device)
                remain_images = remain_images.to(device)
                B = forget_images.shape[0]

                # ── retain loss (flow-matching velocity objective) ─────────────
                loss_r = compute_retain_loss_sd3(
                    transformer, vae, scheduler, pipe,
                    remain_images, list(remain_captions),
                    device, image_size, dtype,
                )

                # ── forget loss ───────────────────────────────────────────────
                # Same image → same latent for both forget and pseudo passes;
                # only text conditioning differs.
                latents = encode_images_to_latents_sd3(vae, forget_images, device, dtype)
                noise   = torch.randn_like(latents)
                indices   = torch.randint(
                    0, scheduler.config.num_train_timesteps, (B,), device=device
                )
                timesteps = scheduler.timesteps[indices]
                sigmas    = get_sigmas_for_timesteps(
                    scheduler, indices, device, dtype, latents.ndim
                )
                noisy = (1.0 - sigmas) * latents + sigmas * noise

                f_embeds, f_pooled = encode_text_sd3(pipe, list(forget_captions), device)
                p_embeds, p_pooled = encode_text_sd3(pipe, [word_wear] * B, device)

                forget_out = transformer_forward_sd3(
                    transformer, noisy, timesteps, f_embeds, f_pooled
                )
                with torch.no_grad():
                    pseudo_out = transformer_forward_sd3(
                        transformer, noisy, timesteps, p_embeds, p_pooled
                    )

                loss_u = criteria(forget_out, pseudo_out.detach()) * beta

                # ── KS gradient merge ─────────────────────────────────────────
                grads_f = torch.autograd.grad(
                    loss_u, parameters, retain_graph=False, allow_unused=True
                )
                grads_r = torch.autograd.grad(
                    loss_r, parameters, retain_graph=False, allow_unused=True
                )

                gr_flat = _flatten_grads(parameters, grads_r)
                gf_flat = _flatten_grads(parameters, grads_f)
                del grads_r, grads_f

                gr_input = gr_flat[mask] if mask is not None else gr_flat
                gf_input = gf_flat[mask] if mask is not None else gf_flat

                lambda_ks, cos_phi, g_star, effective_scale = ks_step(gr_input, gf_input)
                cos_phi_hist.append(cos_phi.item())

                if g_star.norm().item() < 1e-6:
                    skipped_steps += 1
                    del gr_flat, gf_flat, gr_input, gf_input, g_star
                    pbar.update(1)
                    continue

                g_star_scaled = effective_scale * g_star
                del gr_input, gf_input

                if mask is not None:
                    update_full       = torch.zeros_like(gr_flat)
                    update_full[mask] = g_star_scaled
                else:
                    update_full = g_star_scaled

                del gr_flat, gf_flat, g_star, g_star_scaled

                optimizer.zero_grad()
                _unpack_to_grads(parameters, update_full)
                del update_full

                if with_l1:
                    current_alpha = alpha * (1 - epoch / epochs)
                    l1_grads = torch.autograd.grad(
                        current_alpha * l1_regularization(parameters), parameters
                    )
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
            f"Epoch {epoch+1} done | {epoch_time:.1f}s ({epoch_time/60:.2f} min) | "
            f"skip_rate={skipped_steps/max(total_steps,1):.3f}"
        )
        transformer.eval()
        if (epoch + 1) % 1 == 0 and epoch != epochs - 1:
            save_model(transformer, run_tag, epoch + 1, logger=logger)
        torch.cuda.empty_cache(); gc.collect()

    # ── final save ───────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    logger.info("======== MUKSB SD3 NSFW TRAINING FINISHED ========")
    logger.info(
        f"Total: {total_time:.1f}s ({total_time/60:.2f} min | {total_time/3600:.2f} hrs)"
    )
    if cos_phi_hist:
        logger.info(
            f"cos_φ stats: mean={np.mean(cos_phi_hist):.4f}  "
            f"min={np.min(cos_phi_hist):.4f}  max={np.max(cos_phi_hist):.4f}"
        )
    logger.info(
        f"Anti-parallel skips: {skipped_steps}/{total_steps} "
        f"({skipped_steps/max(total_steps,1)*100:.1f}%)"
    )
    transformer.eval()
    save_model(transformer, run_tag, epochs, logger=logger)
    save_history(losses, run_tag)
    logger.info(f"Model saved under: models/{run_tag}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _mask_type(v):
        return None if v in (None, "None", "none") else str(v)

    parser = argparse.ArgumentParser(
        description="MUKSB: KS-Bargaining NSFW unlearning for Stable Diffusion 3"
    )
    parser.add_argument("--model_id",        type=str,
                        default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--train_method",    type=str, default="full",
                        choices=["full", "attn", "ff", "joint_blocks", "single_blocks"],
                        help="full = all transformer params; attn = attention only (saves ~8 GB Adam states)")
    parser.add_argument("--batch_size",      type=int,   default=1)
    parser.add_argument("--epochs",          type=int,   default=15)
    parser.add_argument("--lr",              type=float, default=1e-5)
    parser.add_argument("--mask_variant",    type=_mask_type, default=None,
                        choices=list(MASK_VARIANT_CHOICES) + [None])
    parser.add_argument("--mask_density",    type=float, default=0.5)
    parser.add_argument("--lambda_tradeoff", type=float, default=1.0)
    parser.add_argument("--device",         type=str,   default="5")
    parser.add_argument("--image_size",     type=int,   default=512)
    parser.add_argument("--with_l1",        action="store_true", default=False)
    parser.add_argument("--alpha",          type=float, default=1e-4)
    parser.add_argument("--beta",           type=float, default=100.0)
    parser.add_argument("--forget_path",    type=str,
                        default="/storage/s25017/Datasets/NSFW_removal/nude")
    parser.add_argument("--remain_path",    type=str,
                        default="/storage/s25017/Datasets/NSFW_removal/with_dress")
    parser.add_argument("--dtype",          type=str,   default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--skip_t5",        action="store_true", default=False,
                        help="Skip T5-XXL encoder to save ~9.4 GB VRAM (uses only CLIP-L + CLIP-G)")
    args = parser.parse_args()

    logger, log_file = setup_logger(name="MUKSB_sd3_nsfw")
    logger.info("======== MUKSB SD3 NSFW STARTED ========")
    logger.info(f"Log: {log_file}")
    logger.info(f"Args: {vars(args)}")

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    np.random.seed(42); random.seed(42)
    torch.backends.cudnn.deterministic = True

    _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    MUKSB(
        model_id        = args.model_id,
        train_method    = args.train_method,
        batch_size      = args.batch_size,
        epochs          = args.epochs,
        lr              = args.lr,
        mask_variant    = args.mask_variant,
        mask_density    = args.mask_density,
        lambda_tradeoff = args.lambda_tradeoff,
        device          = f"cuda:{int(args.device)}",
        image_size      = args.image_size,
        with_l1         = args.with_l1,
        alpha           = args.alpha,
        beta            = args.beta,
        forget_path     = args.forget_path,
        remain_path     = args.remain_path,
        dtype           = _dtype_map[args.dtype],
        skip_t5         = args.skip_t5,
        logger          = logger,
    )
