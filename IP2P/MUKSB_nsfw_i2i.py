"""
IP2P/MUKSB_nsfw_i2i.py
=======================
MUKSB (Magnitude-Aware Kalai-Smorodinsky Bargaining) NSFW concept
unlearning under **Image-to-Image** (InstructPix2Pix).

The bargaining math (`ks_step`), L1 regularisation, and mask-driven
sparse update are identical to SD/MUKSB_nsfw.py; the forward pass goes
through the diffusers IP2P UNet with 8-channel inputs
(noisy target latent ⊕ source-image latent).

Mask variants
-------------
  none          — no mask (all selected params updated under KS bargaining)
  random        — uniform random top-k%
  forget_fisher — single-sided forget Fisher (F_f)
  salun         — gradient-magnitude saliency |∇L_f|
  dual_fisher   — dual Fisher combined score (proposed)

Usage
-----
  python MUKSB_nsfw_i2i.py --mask_variant dual_fisher --epochs 5 --device 0
"""

import argparse
import copy
import gc
import os
import random
import sys
import time
from time import sleep

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from logger.logger import setup_logger
from dataset_i2i import setup_i2i_nsfw_data
from mask_variants_i2i import build_mask, MASK_VARIANT_CHOICES

# Semantic "clothed" anchors for the forget objective.  Instead of redirecting
# the NSFW instruction onto an identity / "keep unchanged" map (which collapses
# editing and causes flicker), we redirect it onto the FROZEN base model's
# behaviour under one of these clothed anchors, sampled at random per image.
CLOTHED_ANCHORS = [
    "a photo of a person wearing clothes",
    "the same person fully dressed",
    "a person in normal everyday clothing",
    "keep the subject fully clothed",
]
VAE_SCALE = 0.18215

EXTRA = "i2i"


# ─────────────────────────────────────────────────────────────────────────────
# KS bargaining core
# ─────────────────────────────────────────────────────────────────────────────

def ks_step(gr_flat: torch.Tensor, gf_flat: torch.Tensor, eps: float = 1e-8):
    norm_gr = torch.clamp(torch.norm(gr_flat), min=1e-6)
    norm_gf = torch.clamp(torch.norm(gf_flat), min=1e-6)

    cos_phi = torch.clamp(
        torch.dot(gr_flat, gf_flat) / (norm_gr * norm_gf),
        -1.0 + eps, 1.0 - eps,
    )

    g_hat_r = gr_flat / norm_gr
    g_hat_f = gf_flat / norm_gf

    g_sum = g_hat_r + g_hat_f
    norm_sum = torch.norm(g_sum)

    if norm_sum < 1e-6:
        zero = torch.zeros_like(gr_flat)
        return (
            torch.tensor(0.0, device=gr_flat.device),
            cos_phi,
            zero,
            torch.tensor(0.0, device=gr_flat.device),
        )

    g_star = g_sum / norm_sum
    effective_scale = 2.0 * norm_gr * norm_gf / (norm_gr + norm_gf)
    lambda_ks = torch.dot(g_hat_r, g_star)

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
        n = p.numel()
        p.grad = flat_vec[offset: offset + n].view(p.shape).clone()
        offset += n


def l1_regularization(parameters):
    return torch.linalg.norm(
        torch.cat([p.view(-1) for p in parameters]), ord=1
    )


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────────────────────
# I2I-specific helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_i2i_model(ckpt_path, device):
    """Load InstructPix2Pix; freeze VAE/text encoder; keep UNet trainable."""
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe


def select_parameters(unet, train_method):
    """Diffusers UNet parameter selection — mirrors the T2I MUKSB choices."""
    parameters, param_names = [], []
    for name, param in unet.named_parameters():
        keep = False
        if train_method == "full":
            keep = True
        elif train_method == "noxattn":
            keep = not (
                name.startswith("conv_out.")
                or "attn2" in name
                or "time_embedding" in name
            )
        elif train_method == "selfattn":
            keep = "attn1" in name
        elif train_method == "xattn":
            keep = "attn2" in name
        elif train_method == "notime":
            keep = not (name.startswith("conv_out.") or "time_embedding" in name)
        elif train_method == "xlayer":
            keep = "attn2" in name and (
                "up_blocks.2." in name or "up_blocks.3." in name
            )
        elif train_method == "selflayer":
            keep = "attn1" in name and (
                "down_blocks.1." in name or "down_blocks.2." in name
            )
        if keep:
            param.requires_grad_(True)
            parameters.append(param)
            param_names.append(name)
    return parameters, param_names


def _encode_text(text_encoder, tokenizer, prompts, device):
    toks = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(toks.input_ids.to(device))[0]


def _encode_image(vae, images_hwc, device):
    imgs = images_hwc.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        latent = vae.encode(imgs).latent_dist.sample() * VAE_SCALE
    return latent


def i2i_forget_loss(pipe, base_unet, forget_batch, device, beta, criteria):
    """Redirect the NSFW instruction onto the FROZEN base model's behaviour
    under a randomized *clothed* anchor.

    The trainable UNet (forget instruction) is matched to base_unet (clothed
    anchor) on the same noisy input, so the model learns to respond to the
    NSFW instruction as if asked to keep the subject clothed.  Using the frozen
    base — rather than the moving model under an identity prompt — gives a
    stable, non-self-referential target, which removes the editing collapse and
    the flicker that the old `keep the image unchanged` objective produced.
    """
    n_imgs = forget_batch["jpg"].shape[0]
    tgt_lat = _encode_image(pipe.vae, forget_batch["jpg"], device)
    src_lat = _encode_image(pipe.vae, forget_batch["src"], device)

    t = torch.randint(
        0, pipe.scheduler.config.num_train_timesteps,
        (tgt_lat.shape[0],), device=device,
    ).long()
    noise = torch.randn_like(tgt_lat)
    noisy = pipe.scheduler.add_noise(tgt_lat, noise, t)
    cat = torch.cat([noisy, src_lat], dim=1)  # 8 channels

    emb_forget = _encode_text(
        pipe.text_encoder, pipe.tokenizer, forget_batch["txt"], device
    )
    anchor_prompts = [random.choice(CLOTHED_ANCHORS) for _ in range(n_imgs)]
    emb_anchor = _encode_text(
        pipe.text_encoder, pipe.tokenizer, anchor_prompts, device
    )

    f_out = pipe.unet(cat, t, encoder_hidden_states=emb_forget).sample
    with torch.no_grad():
        a_out = base_unet(cat, t, encoder_hidden_states=emb_anchor).sample
    return criteria(f_out, a_out) * beta


def i2i_retain_loss(pipe, base_unet, retain_batch, device, criteria):
    """Distil the trainable UNet toward the FROZEN base UNet on diverse benign
    attribute edits (smile, sunglasses, background, ...).

    This preserves general editing capability *without* needing real
    (source, instruction, edited-target) triplets: the base model's own noise
    prediction is the target, so matching it on benign prompts keeps the
    trainable model functionally identical to base outside the forget concept.
    """
    src_lat = _encode_image(pipe.vae, retain_batch["src"], device)

    t = torch.randint(
        0, pipe.scheduler.config.num_train_timesteps,
        (src_lat.shape[0],), device=device,
    ).long()
    noise = torch.randn_like(src_lat)
    noisy = pipe.scheduler.add_noise(src_lat, noise, t)
    cat = torch.cat([noisy, src_lat], dim=1)

    emb = _encode_text(
        pipe.text_encoder, pipe.tokenizer, retain_batch["txt"], device
    )
    pred = pipe.unet(cat, t, encoder_hidden_states=emb).sample
    with torch.no_grad():
        ref = base_unet(cat, t, encoder_hidden_states=emb).sample
    return criteria(pred, ref)


def save_model_diffusers(pipe, name, num):
    """Save the whole diffusers pipeline directory."""
    folder_path = (
        f"models/{name}/epoch_{num}" if num is not None else f"models/{name}"
    )
    os.makedirs(folder_path, exist_ok=True)
    pipe.save_pretrained(folder_path)


def save_history(losses, name):
    import matplotlib.pyplot as plt
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(v) + "\n" for v in losses])
    if len(losses) >= 3:
        v = np.convolve(losses, np.ones(3) / 3, mode="valid")
        plt.figure()
        plt.plot(v, label="nsfw_i2i_loss")
        plt.legend(loc="upper left")
        plt.title("Training loss (moving avg, n=3)")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(f"{folder_path}/loss.png")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main MUKSB I2I training loop
# ─────────────────────────────────────────────────────────────────────────────

def MUKSB_i2i(
    train_method,
    batch_size,
    epochs,
    lr,
    ckpt_path,
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
    logger,
):
    total_start = time.time()
    logger.info("======== MUKSB NSFW-I2I (KS Bargaining) TRAINING STARTED ========")
    logger.info(f"EXTRA = {EXTRA}  mask_variant={mask_variant}  density={mask_density}")

    pipe = setup_i2i_model(ckpt_path, device)

    # Frozen base UNet — stable anchor for the forget objective and target for
    # the retain distillation.  Snapshot BEFORE any training modifies pipe.unet.
    base_unet = copy.deepcopy(pipe.unet)
    base_unet.requires_grad_(False)
    base_unet.eval()
    logger.info("Frozen base UNet snapshot created (anchor + retain-distill target)")

    criteria = torch.nn.MSELoss()

    forget_dl, remain_dl = setup_i2i_nsfw_data(
        batch_size, forget_path, remain_path, image_size
    )
    fisher_dl, _ = setup_i2i_nsfw_data(
        batch_size, forget_path, remain_path, image_size
    )
    num_forget = len(forget_dl.dataset)
    logger.info(
        f"Forget samples: {num_forget} | Remain samples: {len(remain_dl.dataset)}"
    )

    parameters, param_names = select_parameters(pipe.unet, train_method)
    logger.info(f"Trainable params: {sum(p.numel() for p in parameters):,}")

    # ── build importance mask (if requested) ────────────────────────────────
    max_batches_forget = len(forget_dl)
    max_batches_retain = min(len(remain_dl), max_batches_forget * 3)

    pipe.unet.eval()
    if mask_variant is None or mask_variant == "none":
        mask = None
        run_tag = (
            f"i2p-nsfw-MUKSB-i2i"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{EXTRA}"
        )
        logger.info("[Mask] no mask — all selected params active")
    else:
        mask = build_mask(
            variant=mask_variant,
            model=pipe.unet,
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            scheduler=pipe.scheduler,
            parameters=parameters,
            forget_dl=fisher_dl,
            remain_dl=remain_dl,
            beta=beta,
            device=device,
            target_density=mask_density,
            max_batches_forget=max_batches_forget,
            max_batches_retain=max_batches_retain,
            lambda_tradeoff=lambda_tradeoff,
            logger=logger,
        )
        active = mask.sum().item()
        total = mask.numel()
        logger.info(
            f"[Mask] active={active:,} / {total:,}  density={active/total:.4f}"
        )
        run_tag = (
            f"i2p-nsfw-MUKSB-i2i-{mask_variant}"
            f"-rho{int(mask_density*100)}pct"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{EXTRA}"
        )

    pipe.unet.train()
    optimizer = torch.optim.Adam(parameters, lr=lr)
    losses = []
    step = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch + 1}/{epochs} started")

        remain_iter = iter(remain_dl)

        with tqdm(total=len(forget_dl), desc=f"Epoch {epoch + 1}") as pbar:
            for forget_batch in forget_dl:

                try:
                    remain_batch = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_batch = next(remain_iter)

                # ── retain loss (distil benign edits from frozen base) ─────
                loss_r = i2i_retain_loss(pipe, base_unet, remain_batch, device, criteria)

                # ── forget loss (redirect NSFW → frozen-base clothed anchor) ─
                loss_u = i2i_forget_loss(pipe, base_unet, forget_batch, device, beta, criteria)

                # ── KS gradient merge ────────────────────────────────────
                grads_r = torch.autograd.grad(
                    loss_r, parameters, retain_graph=True, allow_unused=True
                )
                grads_f = torch.autograd.grad(
                    loss_u, parameters, allow_unused=True
                )

                gr_flat = _flatten_grads(parameters, grads_r)
                gf_flat = _flatten_grads(parameters, grads_f)

                if mask is not None:
                    gr_masked = gr_flat[mask]
                    gf_masked = gf_flat[mask]
                else:
                    gr_masked = gr_flat
                    gf_masked = gf_flat

                lambda_ks, cos_phi, g_star, effective_scale = ks_step(
                    gr_masked, gf_masked
                )

                if torch.norm(g_star).item() < 1e-6:
                    logger.info(
                        f"step={step}: anti-parallel gradients "
                        f"(cos_φ={cos_phi.item():.3f}), skipping update"
                    )
                    del gr_flat, gf_flat, gr_masked, gf_masked
                    del grads_r, grads_f, g_star
                    pbar.update(1)
                    continue

                g_star_scaled = effective_scale * g_star
                del gr_masked, gf_masked, grads_r, grads_f

                if mask is not None:
                    update_full = torch.zeros_like(gr_flat)
                    update_full[mask] = g_star_scaled
                else:
                    update_full = g_star_scaled

                del gr_flat, gf_flat, g_star, g_star_scaled

                optimizer.zero_grad()
                _unpack_to_grads(parameters, update_full)
                del update_full

                if with_l1:
                    l1_loss = alpha * l1_regularization(parameters)
                    l1_grads = torch.autograd.grad(l1_loss, parameters)
                    for p, lg in zip(parameters, l1_grads):
                        if p.grad is not None and lg is not None:
                            p.grad += lg.detach()

                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                losses.append((loss_r + loss_u).item() / batch_size)
                step += 1

                if step % 10 == 0:
                    logger.info(
                        f"step={step}"
                        f"  λ_KS={lambda_ks.item():.4f}"
                        f"  cos_φ={cos_phi.item():.4f}"
                        f"  loss_r={loss_r.item():.4f}"
                        f"  loss_u={loss_u.item():.4f}"
                    )
                    save_history(losses, run_tag)
                if step % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                pbar.set_postfix(
                    loss_r=f"{loss_r:.4f}", lam=f"{lambda_ks.item():.3f}"
                )
                sleep(0.05)
                pbar.update(1)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1} done | {epoch_time:.1f}s ({epoch_time/60:.2f} min)"
        )

        if (epoch + 1) % 1 == 0 and epoch != epochs - 1:
            pipe.unet.eval()
            save_model_diffusers(pipe, run_tag, epoch + 1)
            pipe.unet.train()

        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - total_start
    logger.info("======== MUKSB NSFW-I2I TRAINING FINISHED ========")
    logger.info(
        f"Total: {total_time:.1f}s "
        f"({total_time/60:.2f} min | {total_time/3600:.2f} hrs)"
    )
    pipe.unet.eval()
    save_model_diffusers(pipe, run_tag, epochs)
    save_history(losses, run_tag)
    logger.info(f"Model and loss history saved under: models/{run_tag}/")
    return run_tag


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MUKSB I2I (InstructPix2Pix): KS-Bargaining NSFW concept unlearning"
    )
    parser.add_argument("--train_method", type=str, default="xattn",
                        choices=["full", "noxattn", "xattn", "selfattn",
                                 "notime", "xlayer", "selflayer"],
                        help="xattn (cross-attention only) localises erasure to "
                             "the text->image association; avoids the global "
                             "denoiser damage that 'full' causes.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ckpt_path", type=str,
                        default="timbrooks/instruct-pix2pix",
                        help="Local IP2P diffusers dir or HF id.")
    parser.add_argument(
        "--mask_variant", type=str, default="none",
        choices=list(MASK_VARIANT_CHOICES) + ["none", "None"],
    )
    parser.add_argument("--mask_density", type=float, default=0.5)
    parser.add_argument("--lambda_tradeoff", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--with_l1", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=100.0,
                        help="Forget-loss scale. Kept small (was 100) so the "
                             "forget term no longer dominates the KS merge.")
    parser.add_argument(
        "--forget_path", type=str,
        default="/storage/s25017/Datasets/NSFW_removal/nude",
    )
    parser.add_argument(
        "--remain_path", type=str,
        default="/storage/s25017/Datasets/NSFW_removal/with_dress",
    )
    args = parser.parse_args()

    log_name = f"MUKSB_nsfw_i2i_{args.mask_variant}"
    logger, log_file = setup_logger(
        log_dir=os.path.join(_THIS_DIR, "logs"),
        name=log_name,
    )
    logger.info(f"Log: {log_file}")
    logger.info(f"Args: {vars(args)}")
    setup_seed(42)

    mask_variant = args.mask_variant
    if mask_variant in ("None", "none"):
        mask_variant = "none"

    MUKSB_i2i(
        train_method=args.train_method,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        ckpt_path=args.ckpt_path,
        mask_variant=mask_variant,
        mask_density=args.mask_density,
        lambda_tradeoff=args.lambda_tradeoff,
        device=f"cuda:{int(args.device)}",
        image_size=args.image_size,
        with_l1=args.with_l1,
        alpha=args.alpha,
        beta=args.beta,
        forget_path=args.forget_path,
        remain_path=args.remain_path,
        logger=logger,
    )
