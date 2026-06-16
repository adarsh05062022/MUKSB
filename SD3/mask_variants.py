"""
SD3/mask_variants.py
====================
Parameter-selection (saliency mask) strategies for SD3 MUKSB NSFW unlearning.
Mirrors SDXL/mask_variants.py but uses the SD3 flow-matching API.

Strategies: random, forget_fisher, salun, dual_fisher
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_scripts.dataset import (
    encode_text_sd3,
    encode_images_to_latents_sd3,
    transformer_forward_sd3,
    compute_retain_loss_sd3,
    get_sigmas_for_timesteps,
)

PSEUDO_CAPTION_NSFW  = "a photo of a person wearing clothes"
MASK_VARIANT_CHOICES = ("random", "forget_fisher", "salun", "dual_fisher")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _z(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp(min=1e-10)


def _topk_mask(score: torch.Tensor, target_density: float) -> torch.Tensor:
    k         = max(1, int(target_density * score.numel()))
    top_idx   = torch.topk(score, k).indices
    mask_flat = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    mask_flat[top_idx] = True
    return mask_flat


# ─────────────────────────────────────────────────────────────────────────────
# Fisher / grad-mag accumulation — forget set (SD3 flow-matching version)
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate_forget_fisher_and_gradmag_nsfw(
    transformer, vae, scheduler, pipe,
    forget_dl, parameters,
    pseudo_caption, beta,
    device, image_size, dtype,
    max_batches,
):
    criteria  = torch.nn.MSELoss()
    transformer.eval()
    accum_f   = [torch.zeros_like(p) for p in parameters]
    accum_mag = [torch.zeros_like(p) for p in parameters]
    n_batches = 0

    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)

    for batch_idx, (forget_images, forget_captions) in enumerate(
        tqdm(forget_dl, desc="[Mask/Fisher] forget batches")
    ):
        if batch_idx >= max_batches:
            break

        forget_images = forget_images.to(device)
        B             = forget_images.shape[0]

        latents = encode_images_to_latents_sd3(vae, forget_images, device, dtype)
        noise   = torch.randn_like(latents)
        indices   = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
        timesteps = scheduler.timesteps[indices]
        sigmas    = get_sigmas_for_timesteps(scheduler, indices, device, dtype, latents.ndim)
        noisy     = (1.0 - sigmas) * latents + sigmas * noise

        f_embeds, f_pooled = encode_text_sd3(pipe, list(forget_captions), device)
        p_embeds, p_pooled = encode_text_sd3(pipe, [pseudo_caption] * B, device)

        transformer.train()
        f_pred = transformer_forward_sd3(transformer, noisy, timesteps, f_embeds, f_pooled)
        with torch.no_grad():
            p_pred = transformer_forward_sd3(transformer, noisy, timesteps, p_embeds, p_pooled)

        loss_f = criteria(f_pred, p_pred.detach()) * beta

        grads = torch.autograd.grad(loss_f, parameters, retain_graph=False, allow_unused=True)
        for i, g in enumerate(grads):
            if g is not None:
                accum_f[i]   += g.detach() ** 2
                accum_mag[i] += g.detach().abs()

        del f_pred, p_pred, latents, noisy, loss_f, grads
        n_batches += 1

    n = max(n_batches, 1)
    for i in range(len(accum_f)):
        accum_f[i]   /= n
        accum_mag[i] /= n

    return accum_f, accum_mag


# ─────────────────────────────────────────────────────────────────────────────
# Fisher accumulation — retain set
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate_retain_fisher_nsfw(
    transformer, vae, scheduler, pipe,
    remain_dl, parameters,
    device, image_size, dtype,
    max_batches,
):
    transformer.eval()
    accum_r     = [torch.zeros_like(p) for p in parameters]
    remain_iter = iter(remain_dl)
    n_batches   = 0

    for batch_idx in tqdm(range(max_batches), desc="[Mask/Fisher] retain batches"):
        try:
            remain_images, remain_captions = next(remain_iter)
        except StopIteration:
            break

        transformer.train()
        loss_r = compute_retain_loss_sd3(
            transformer, vae, scheduler, pipe,
            remain_images, list(remain_captions),
            device, image_size, dtype,
        )
        grads_r = torch.autograd.grad(loss_r, parameters, retain_graph=False, allow_unused=True)
        for i, g in enumerate(grads_r):
            if g is not None:
                accum_r[i] += g.detach() ** 2
        del loss_r, grads_r
        n_batches += 1

    n = max(n_batches, 1)
    for i in range(len(accum_r)):
        accum_r[i] /= n

    return accum_r


# ─────────────────────────────────────────────────────────────────────────────
# (a) Random mask
# ─────────────────────────────────────────────────────────────────────────────

def compute_random_mask(parameters, target_density, device, logger=None):
    total     = sum(p.numel() for p in parameters)
    k         = max(1, int(target_density * total))
    perm      = torch.randperm(total, device=device)
    mask_flat = torch.zeros(total, dtype=torch.bool, device=device)
    mask_flat[perm[:k]] = True
    if logger:
        logger.info(f"[Mask/random]  active={k:,}/{total:,}  density={k/total:.4f}")
    return mask_flat


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def build_mask_nsfw(
    variant,
    transformer, vae, scheduler, pipe,
    parameters,
    forget_dl, remain_dl,
    beta, device, image_size, dtype,
    target_density, max_batches,
    pseudo_caption=PSEUDO_CAPTION_NSFW,
    lambda_tradeoff=1.0,
    logger=None,
):
    if variant not in MASK_VARIANT_CHOICES:
        raise ValueError(f"Unknown mask variant: {variant!r}. Choose from {MASK_VARIANT_CHOICES}.")

    if logger:
        logger.info(f"[Mask/NSFW] variant='{variant}'  density={target_density}  max_batches={max_batches}")

    if variant == "random":
        return compute_random_mask(parameters, target_density, device, logger)

    accum_f, accum_mag = _accumulate_forget_fisher_and_gradmag_nsfw(
        transformer, vae, scheduler, pipe,
        forget_dl, parameters,
        pseudo_caption, beta,
        device, image_size, dtype, max_batches,
    )

    if variant == "forget_fisher":
        score     = _z(torch.cat([f.reshape(-1) for f in accum_f]))
        mask_flat = _topk_mask(score, target_density)
        if logger:
            logger.info(f"[Mask/NSFW/forget_fisher]  active={mask_flat.sum():,}/{mask_flat.numel():,}")
        del accum_f, accum_mag; torch.cuda.empty_cache()
        return mask_flat

    if variant == "salun":
        score     = _z(torch.cat([m.reshape(-1) for m in accum_mag]))
        mask_flat = _topk_mask(score, target_density)
        if logger:
            logger.info(f"[Mask/NSFW/salun]  active={mask_flat.sum():,}/{mask_flat.numel():,}")
        del accum_f, accum_mag; torch.cuda.empty_cache()
        return mask_flat

    # dual_fisher
    accum_r  = _accumulate_retain_fisher_nsfw(
        transformer, vae, scheduler, pipe,
        remain_dl, parameters,
        device, image_size, dtype, max_batches,
    )
    global_f = torch.cat([f.reshape(-1) for f in accum_f]).clamp(min=1e-10)
    global_r = torch.cat([r.reshape(-1) for r in accum_r]).clamp(min=1e-10)

    log_ratio = torch.log(global_f) - torch.log(global_r)
    diff      = (global_f / global_f.std().clamp(1e-10)) \
              - lambda_tradeoff * (global_r / global_r.std().clamp(1e-10))
    score     = 0.5 * _z(log_ratio) + 0.5 * _z(diff)

    mask_flat = _topk_mask(score, target_density)
    if logger:
        logger.info(
            f"[Mask/NSFW/dual_fisher]  active={mask_flat.sum():,}/{mask_flat.numel():,}"
            f"  density={mask_flat.float().mean():.4f}"
        )
    del accum_f, accum_mag, accum_r; torch.cuda.empty_cache()
    return mask_flat
