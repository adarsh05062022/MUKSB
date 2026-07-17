
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_scripts.dataset import (
    encode_text_sdxl,
    encode_images_to_latents,
    unet_forward,
    compute_retain_loss,
)

PSEUDO_CAPTION_NSFW = "a photo of a person wearing clothes"

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
# Fisher / gradient-magnitude accumulation — forget set
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate_forget_fisher_and_gradmag_nsfw(
    unet, vae, scheduler,
    tokenizer, tokenizer_2, text_encoder, text_encoder_2,
    forget_dl, parameters,
    pseudo_caption, beta,
    device, image_size, dtype,
    max_batches,
):
    """
    Single forward/backward pass over the NSFW forget set.

    Returns
    -------
    accum_f   : list[Tensor]  — per-param squared gradient (Fisher approx)
    accum_mag : list[Tensor]  — per-param gradient magnitude |∇L_f| (SalUn-style)
    """
    criteria  = torch.nn.MSELoss()
    unet.eval()
    accum_f   = [torch.zeros_like(p) for p in parameters]
    accum_mag = [torch.zeros_like(p) for p in parameters]
    n_batches = 0

    for batch_idx, (forget_images, forget_captions) in enumerate(
        tqdm(forget_dl, desc="[Mask/Fisher] forget batches")
    ):
        if batch_idx >= max_batches:
            break

        forget_images = forget_images.to(device)
        B             = forget_images.shape[0]
        pseudo_caps   = [pseudo_caption] * B

        # Encode images to latents once (same image → same latent for both passes)
        latents = encode_images_to_latents(vae, forget_images, device, dtype)
        noise   = torch.randn_like(latents)
        t       = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,), device=device
        )
        noisy = scheduler.add_noise(latents, noise, t)

        # Text embeddings (frozen)
        f_enc_h, f_pooled = encode_text_sdxl(
            tokenizer, tokenizer_2, text_encoder, text_encoder_2,
            list(forget_captions), device,
        )
        p_enc_h, p_pooled = encode_text_sdxl(
            tokenizer, tokenizer_2, text_encoder, text_encoder_2,
            pseudo_caps, device,
        )

        # UNet forward passes
        unet.train()
        f_pred = unet_forward(unet, noisy, t, f_enc_h, f_pooled, image_size)
        with torch.no_grad():
            p_pred = unet_forward(unet, noisy, t, p_enc_h, p_pooled, image_size)

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
    unet, vae, scheduler,
    tokenizer, tokenizer_2, text_encoder, text_encoder_2,
    remain_dl, parameters,
    device, image_size, dtype,
    max_batches,
):
    """Accumulate retain Fisher F_r over the non-NSFW retain set."""
    unet.eval()
    accum_r     = [torch.zeros_like(p) for p in parameters]
    remain_iter = iter(remain_dl)
    n_batches   = 0

    for batch_idx in tqdm(range(max_batches), desc="[Mask/Fisher] retain batches"):
        try:
            remain_images, remain_captions = next(remain_iter)
        except StopIteration:
            break

        unet.train()
        loss_r = compute_retain_loss(
            unet, vae, scheduler,
            tokenizer, tokenizer_2, text_encoder, text_encoder_2,
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

def compute_random_mask(parameters, target_density: float, device, logger=None):
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
    variant: str,
    unet, vae, scheduler,
    tokenizer, tokenizer_2, text_encoder, text_encoder_2,
    parameters,
    forget_dl,
    remain_dl,
    beta,
    device,
    image_size,
    dtype,
    target_density: float,
    max_batches: int,
    pseudo_caption: str = PSEUDO_CAPTION_NSFW,
    lambda_tradeoff: float = 1.0,
    logger=None,
) -> torch.Tensor:
    """
    Dispatcher for SDXL NSFW mask building.

    Parameters
    ----------
    variant : str
        One of "random", "forget_fisher", "salun", "dual_fisher".
    """
    if variant not in MASK_VARIANT_CHOICES:
        raise ValueError(f"Unknown mask variant: {variant!r}. Choose from {MASK_VARIANT_CHOICES}.")

    if logger:
        logger.info(f"[Mask/NSFW] variant='{variant}'  density={target_density}  max_batches={max_batches}")

    if variant == "random":
        return compute_random_mask(parameters, target_density, device, logger)

    # Accumulate forget Fisher + grad-mag
    accum_f, accum_mag = _accumulate_forget_fisher_and_gradmag_nsfw(
        unet, vae, scheduler,
        tokenizer, tokenizer_2, text_encoder, text_encoder_2,
        forget_dl, parameters,
        pseudo_caption, beta,
        device, image_size, dtype,
        max_batches,
    )

    if variant == "forget_fisher":
        global_f  = torch.cat([f.reshape(-1) for f in accum_f])
        score     = _z(global_f)
        mask_flat = _topk_mask(score, target_density)
        if logger:
            active = mask_flat.sum().item()
            logger.info(f"[Mask/NSFW/forget_fisher]  active={active:,}/{mask_flat.numel():,}")
        del accum_f, accum_mag
        torch.cuda.empty_cache()
        return mask_flat

    if variant == "salun":
        global_mag = torch.cat([m.reshape(-1) for m in accum_mag])
        score      = _z(global_mag)
        mask_flat  = _topk_mask(score, target_density)
        if logger:
            active = mask_flat.sum().item()
            logger.info(f"[Mask/NSFW/salun]  active={active:,}/{mask_flat.numel():,}")
        del accum_f, accum_mag
        torch.cuda.empty_cache()
        return mask_flat

    # dual_fisher
    accum_r = _accumulate_retain_fisher_nsfw(
        unet, vae, scheduler,
        tokenizer, tokenizer_2, text_encoder, text_encoder_2,
        remain_dl, parameters,
        device, image_size, dtype,
        max_batches,
    )
    global_f = torch.cat([f.reshape(-1) for f in accum_f]).clamp(min=1e-10)
    global_r = torch.cat([r.reshape(-1) for r in accum_r]).clamp(min=1e-10)

    log_ratio = torch.log(global_f) - torch.log(global_r)
    f_std     = global_f.std().clamp(min=1e-10)
    r_std     = global_r.std().clamp(min=1e-10)
    diff      = (global_f / f_std) - lambda_tradeoff * (global_r / r_std)
    score     = 0.5 * _z(log_ratio) + 0.5 * _z(diff)

    mask_flat = _topk_mask(score, target_density)
    if logger:
        active = mask_flat.sum().item()
        logger.info(
            f"[Mask/NSFW/dual_fisher]  active={active:,}/{mask_flat.numel():,}"
            f"  density={active/mask_flat.numel():.4f}"
        )
    del accum_f, accum_mag, accum_r
    torch.cuda.empty_cache()
    return mask_flat
