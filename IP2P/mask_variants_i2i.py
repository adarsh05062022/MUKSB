"""
IP2P/mask_variants_i2i.py
=========================
Four parameter-selection strategies for the MUKSB NSFW *I2I* ablation.

Adapted from the SD T2I MUKSB mask_variants by routing forward passes
through the diffusers UNet using InstructPix2Pix 8-channel inputs
(noisy target latent concatenated with the source-image latent along
the channel dim).

All variants return a flat bool tensor of the requested density ρ.

Strategies
----------
(a) random        — uniform random top-k%
(b) forget_fisher — score = Z(F_f)
(c) salun         — score = Z(mean |∇L_f|) (SalUn-style gradient saliency)
(d) dual_fisher   — score = ½Z(log F_f − log F_r) + ½Z(F_f/σ_f − λF_r/σ_r)
"""

import torch
import torch.utils.data as tud
from tqdm import tqdm


PSEUDO_INSTRUCTION = "keep the image unchanged"
VAE_SCALE = 0.18215

MASK_VARIANT_CHOICES = ("random", "forget_fisher", "salun", "dual_fisher")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _z(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp(min=1e-10)


def _topk_mask(score: torch.Tensor, target_density: float) -> torch.Tensor:
    k = max(1, int(target_density * score.numel()))
    top_idx = torch.topk(score, k).indices
    mask_flat = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    mask_flat[top_idx] = True
    return mask_flat


def _encode_text(text_encoder, tokenizer, prompts, device):
    toks = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(toks.input_ids.to(device))[0]


def _encode_image_to_latent(vae, images_hwc, device):
    imgs = images_hwc.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        latent = vae.encode(imgs).latent_dist.sample() * VAE_SCALE
    return latent


def _i2i_forget_loss(
    unet, vae, text_encoder, tokenizer, scheduler,
    forget_batch, device, beta,
):
    criteria = torch.nn.MSELoss()
    n_imgs = forget_batch["jpg"].shape[0]

    tgt_lat = _encode_image_to_latent(vae, forget_batch["jpg"], device)
    src_lat = _encode_image_to_latent(vae, forget_batch["src"], device)

    t = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (tgt_lat.shape[0],), device=device,
    ).long()
    noise = torch.randn_like(tgt_lat)
    noisy = scheduler.add_noise(tgt_lat, noise, t)
    cat = torch.cat([noisy, src_lat], dim=1)  # 8 channels

    emb_forget = _encode_text(text_encoder, tokenizer, forget_batch["txt"], device)
    emb_pseudo = _encode_text(
        text_encoder, tokenizer, [PSEUDO_INSTRUCTION] * n_imgs, device
    )

    f_out = unet(cat, t, encoder_hidden_states=emb_forget).sample
    p_out = unet(cat, t, encoder_hidden_states=emb_pseudo).sample.detach()
    return criteria(f_out, p_out) * beta


def _i2i_retain_loss(
    unet, vae, text_encoder, tokenizer, scheduler,
    retain_batch, device,
):
    criteria = torch.nn.MSELoss()

    tgt_lat = _encode_image_to_latent(vae, retain_batch["jpg"], device)
    src_lat = _encode_image_to_latent(vae, retain_batch["src"], device)

    t = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (tgt_lat.shape[0],), device=device,
    ).long()
    noise = torch.randn_like(tgt_lat)
    noisy = scheduler.add_noise(tgt_lat, noise, t)
    cat = torch.cat([noisy, src_lat], dim=1)

    emb = _encode_text(text_encoder, tokenizer, retain_batch["txt"], device)
    pred = unet(cat, t, encoder_hidden_states=emb).sample
    return criteria(pred, noise)


# ─────────────────────────────────────────────────────────────────────────────
# Fisher / gradient accumulators
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate_forget_fisher_and_gradmag(
    unet, vae, text_encoder, tokenizer, scheduler,
    forget_dl, parameters, beta, device, max_batches,
):
    unet.eval()
    accum_f = [torch.zeros_like(p) for p in parameters]
    accum_mag = [torch.zeros_like(p) for p in parameters]
    n_batches = 0

    for batch_idx, forget_batch in enumerate(
        tqdm(forget_dl, desc="[MUKSB Fisher/I2I] forget batches")
    ):
        if batch_idx >= max_batches:
            break

        loss_f = _i2i_forget_loss(
            unet, vae, text_encoder, tokenizer, scheduler,
            forget_batch, device, beta,
        )
        grads = torch.autograd.grad(
            loss_f, parameters, retain_graph=False, allow_unused=True
        )
        for i, g in enumerate(grads):
            if g is not None:
                accum_f[i] += g.detach() ** 2
                accum_mag[i] += g.detach().abs()

        del loss_f, grads
        n_batches += 1

    n = max(n_batches, 1)
    for i in range(len(accum_f)):
        accum_f[i] /= n
        accum_mag[i] /= n
    return accum_f, accum_mag


def _accumulate_retain_fisher(
    unet, vae, text_encoder, tokenizer, scheduler,
    remain_dl, parameters, device, max_batches,
):
    unet.eval()
    accum_r = [torch.zeros_like(p) for p in parameters]
    n_batches = 0

    dataset = remain_dl.dataset
    n_samples = min(len(dataset), max_batches * remain_dl.batch_size)
    indices = torch.randperm(len(dataset))[:n_samples].tolist()
    subset_dl = tud.DataLoader(
        tud.Subset(dataset, indices),
        batch_size=remain_dl.batch_size,
        shuffle=True,
        num_workers=getattr(remain_dl, "num_workers", 0),
        drop_last=False,
        pin_memory=False,
    )

    for batch_idx, remain_batch in enumerate(
        tqdm(subset_dl, desc="[MUKSB Fisher/I2I] retain batches")
    ):
        if batch_idx >= max_batches:
            break

        loss_r = _i2i_retain_loss(
            unet, vae, text_encoder, tokenizer, scheduler,
            remain_batch, device,
        )
        grads_r = torch.autograd.grad(
            loss_r, parameters, retain_graph=False, allow_unused=True
        )
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
# Mask variants
# ─────────────────────────────────────────────────────────────────────────────

def compute_random_mask(parameters, target_density, device, logger=None):
    total = sum(p.numel() for p in parameters)
    k = max(1, int(target_density * total))
    perm = torch.randperm(total, device=device)
    mask_flat = torch.zeros(total, dtype=torch.bool, device=device)
    mask_flat[perm[:k]] = True
    if logger:
        logger.info(
            f"[Mask/random]  active={k:,} / {total:,}  density={k/total:.4f}"
        )
    return mask_flat


def compute_forget_fisher_mask(
    unet, vae, text_encoder, tokenizer, scheduler,
    forget_dl, parameters, beta, device,
    target_density, max_batches, logger=None,
):
    accum_f, _ = _accumulate_forget_fisher_and_gradmag(
        unet, vae, text_encoder, tokenizer, scheduler,
        forget_dl, parameters, beta, device, max_batches,
    )
    global_f = torch.cat([f.reshape(-1) for f in accum_f if f is not None])
    score = _z(global_f)
    mask_flat = _topk_mask(score, target_density)
    if logger:
        active = mask_flat.sum().item()
        total = mask_flat.numel()
        logger.info(
            f"[Mask/forget_fisher]  active={active:,} / {total:,}"
            f"  density={active/total:.4f}"
        )
    del accum_f
    torch.cuda.empty_cache()
    return mask_flat


def compute_salun_mask(
    unet, vae, text_encoder, tokenizer, scheduler,
    forget_dl, parameters, beta, device,
    target_density, max_batches, logger=None,
):
    _, accum_mag = _accumulate_forget_fisher_and_gradmag(
        unet, vae, text_encoder, tokenizer, scheduler,
        forget_dl, parameters, beta, device, max_batches,
    )
    global_mag = torch.cat([m.reshape(-1) for m in accum_mag if m is not None])
    score = _z(global_mag)
    mask_flat = _topk_mask(score, target_density)
    if logger:
        active = mask_flat.sum().item()
        total = mask_flat.numel()
        logger.info(
            f"[Mask/salun]  active={active:,} / {total:,}"
            f"  density={active/total:.4f}"
        )
    del accum_mag
    torch.cuda.empty_cache()
    return mask_flat


def compute_dual_fisher_mask(
    unet, vae, text_encoder, tokenizer, scheduler,
    forget_dl, remain_dl, parameters, beta, device,
    target_density, max_batches_forget, max_batches_retain,
    lambda_tradeoff=1.0, logger=None,
):
    accum_f, _ = _accumulate_forget_fisher_and_gradmag(
        unet, vae, text_encoder, tokenizer, scheduler,
        forget_dl, parameters, beta, device, max_batches_forget,
    )
    accum_r = _accumulate_retain_fisher(
        unet, vae, text_encoder, tokenizer, scheduler,
        remain_dl, parameters, device, max_batches_retain,
    )

    global_f = torch.cat([f.reshape(-1) for f in accum_f if f is not None]).clamp(min=1e-10)
    global_r = torch.cat([r.reshape(-1) for r in accum_r if r is not None]).clamp(min=1e-10)

    log_ratio = torch.log(global_f) - torch.log(global_r)
    f_std = global_f.std().clamp(min=1e-10)
    r_std = global_r.std().clamp(min=1e-10)
    diff = (global_f / f_std) - lambda_tradeoff * (global_r / r_std)
    score = 0.5 * _z(log_ratio) + 0.5 * _z(diff)

    mask_flat = _topk_mask(score, target_density)
    if logger:
        active = mask_flat.sum().item()
        total = mask_flat.numel()
        logger.info(
            f"[Mask/dual_fisher]  active={active:,} / {total:,}"
            f"  density={active/total:.4f}"
            f"  log_ratio range=[{log_ratio.min():.2f}, {log_ratio.max():.2f}]"
            f"  diff range=[{diff.min():.2f}, {diff.max():.2f}]"
        )
    del accum_f, accum_r
    torch.cuda.empty_cache()
    return mask_flat


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def build_mask(
    variant: str,
    model,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    parameters,
    forget_dl,
    remain_dl,
    beta,
    device,
    target_density: float,
    max_batches_forget: int,
    max_batches_retain: int,
    lambda_tradeoff: float = 1.0,
    logger=None,
) -> torch.Tensor:
    if variant not in MASK_VARIANT_CHOICES:
        raise ValueError(
            f"Unknown mask variant: {variant!r}. Choose from {MASK_VARIANT_CHOICES}."
        )

    if logger:
        logger.info(
            f"[Mask] variant='{variant}'  density={target_density}"
            f"  max_batches_forget={max_batches_forget}"
            f"  max_batches_retain={max_batches_retain}"
        )

    if variant == "random":
        return compute_random_mask(parameters, target_density, device, logger)

    if variant == "forget_fisher":
        return compute_forget_fisher_mask(
            model, vae, text_encoder, tokenizer, scheduler,
            forget_dl, parameters, beta, device,
            target_density, max_batches_forget, logger,
        )

    if variant == "salun":
        return compute_salun_mask(
            model, vae, text_encoder, tokenizer, scheduler,
            forget_dl, parameters, beta, device,
            target_density, max_batches_forget, logger,
        )

    return compute_dual_fisher_mask(
        model, vae, text_encoder, tokenizer, scheduler,
        forget_dl, remain_dl, parameters, beta, device,
        target_density, max_batches_forget, max_batches_retain,
        lambda_tradeoff, logger,
    )
