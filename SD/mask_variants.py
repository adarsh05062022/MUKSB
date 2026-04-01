"""
Experiments/ablation/mask_variants.py
======================================
Four parameter-selection strategies for the SSU ablation study.
All variants return a flat bool tensor ``mask_flat`` of the same
target density ρ, so the downstream Nash gradient update is unchanged.

Strategies
----------
(a) random        — uniform random top-k% selection (no Fisher at all)
(b) forget_fisher — single-sided: score = Z(F_f)  [eq. 7 only]
(c) salun         — gradient-magnitude saliency: score = Z(mean |∇L_f|)
                    mirrors SalUn (https://arxiv.org/abs/2310.12508)
(d) dual_fisher   — SSU full dual score: ½Z(S_ratio) + ½Z(S_diff)
                    [eqs. 10-13, the proposed method]

Each public function has the signature:
    compute_<name>_mask(...) -> (mask_flat: BoolTensor)

The Fisher accumulation helpers are shared where possible to keep
the comparison fair (same number of gradient passes per batch).
"""

import random

import torch
from tqdm import tqdm

# ── Pseudo-concepts for the forget loss ───────────────────────────────────────
# Semantically unrelated to all Imagenette-10 classes so the model is not
# just redirected to a neighbouring in-domain concept.
PSEUDO_CONCEPTS = [
    "a red apple on a wooden table",
    "a cup of coffee with steam",
    "a green leaf on a branch",
    "a sandy beach at sunset",
    "a snow-covered mountain peak",
    "a brick wall with ivy growing on it",
    "a yellow sunflower in a field",
    "a glass of water on a desk",
    "a fluffy white cloud in a blue sky",
    "a lit candle in a dark room",
    "a loaf of bread on a cutting board",
    "a stack of books on a wooden shelf",
    "a potted plant by a sunny window",
    "a ceramic bowl filled with fruit",
    "a gravel path through a garden",
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared Fisher / gradient accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate_forget_fisher_and_gradmag(
    model,
    forget_dl,
    parameters,
    descriptions,
    class_to_forget,
    beta,
    device,
    max_batches,
):
    """
    Single forward/backward pass over the forget set.

    Returns
    -------
    accum_f   : list[Tensor]  — per-param squared gradient (Fisher approx, eq. 7)
    accum_mag : list[Tensor]  — per-param gradient magnitude |∇L_f| (SalUn-style)
    """
    criteria = torch.nn.MSELoss()
    model.eval()

    accum_f   = [torch.zeros_like(p) for p in parameters]
    accum_mag = [torch.zeros_like(p) for p in parameters]
    n_batches = 0

    for batch_idx, (forget_images, forget_labels) in enumerate(
        tqdm(forget_dl, desc="[Ablation Fisher] forget batches")
    ):
        if batch_idx >= max_batches:
            break

        forget_images  = forget_images.to(device)
        forget_prompts = [descriptions[int(lbl)] for lbl in forget_labels]
        # pseudo_prompts = [random.choice(PSEUDO_CONCEPTS) for _ in forget_labels]
        next_cls       = (class_to_forget + 1) % len(descriptions)
        pseudo_prompts = [descriptions[next_cls] for _ in forget_labels]

        forget_batch = {"jpg": forget_images.permute(0, 2, 3, 1), "txt": forget_prompts}
        pseudo_batch = {"jpg": forget_images.permute(0, 2, 3, 1), "txt": pseudo_prompts}

        f_in, f_emb = model.get_input(forget_batch, model.first_stage_key)
        p_in, p_emb = model.get_input(pseudo_batch, model.first_stage_key)

        t     = torch.randint(0, model.num_timesteps, (f_in.shape[0],), device=device).long()
        noise = torch.randn_like(f_in)

        f_out = model.apply_model(model.q_sample(f_in, t, noise), t, f_emb)
        p_out = model.apply_model(model.q_sample(p_in, t, noise), t, p_emb).detach()
        loss_f = criteria(f_out, p_out) * beta

        grads = torch.autograd.grad(loss_f, parameters, retain_graph=False, allow_unused=True)
        for i, g in enumerate(grads):
            if g is not None:
                accum_f[i]   += g.detach() ** 2          # Fisher: g²
                accum_mag[i] += g.detach().abs()          # SalUn:  |g|

        del f_out, p_out, f_in, p_in, loss_f, grads
        n_batches += 1

    n = max(n_batches, 1)
    for i in range(len(accum_f)):
        accum_f[i]   /= n
        accum_mag[i] /= n

    return accum_f, accum_mag


def _accumulate_retain_fisher(
    model,
    remain_dl,
    parameters,
    descriptions,
    device,
    max_batches,
):
    """Accumulate retain Fisher F_r (eq. 8)."""
    model.eval()
    accum_r   = [torch.zeros_like(p) for p in parameters]
    remain_iter = iter(remain_dl)
    n_batches   = 0

    for batch_idx in tqdm(range(max_batches), desc="[Ablation Fisher] retain batches"):
        try:
            remain_images, remain_labels = next(remain_iter)
        except StopIteration:
            break

        remain_images  = remain_images.to(device)
        remain_prompts = [descriptions[int(lbl)] for lbl in remain_labels]
        remain_batch   = {"jpg": remain_images.permute(0, 2, 3, 1), "txt": remain_prompts}

        loss_r  = model.shared_step(remain_batch)[0]
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
# Z-score normalisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _z(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp(min=1e-10)


def _topk_mask(score: torch.Tensor, target_density: float) -> torch.Tensor:
    k = max(1, int(target_density * score.numel()))
    top_idx   = torch.topk(score, k).indices
    mask_flat = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    mask_flat[top_idx] = True
    return mask_flat


# ─────────────────────────────────────────────────────────────────────────────
# (a) Random mask
# ─────────────────────────────────────────────────────────────────────────────

def compute_random_mask(parameters, target_density: float, device, logger=None) -> torch.Tensor:
    """
    Variant (a): random parameter selection.

    Selects top-k% parameters uniformly at random — no gradient information
    used. Acts as a lower-bound baseline.
    """
    total     = sum(p.numel() for p in parameters)
    k         = max(1, int(target_density * total))
    perm      = torch.randperm(total, device=device)
    mask_flat = torch.zeros(total, dtype=torch.bool, device=device)
    mask_flat[perm[:k]] = True

    if logger:
        logger.info(f"[Mask/random]  active={k:,} / {total:,}  density={k/total:.4f}")
    return mask_flat


# ─────────────────────────────────────────────────────────────────────────────
# (b) Single-sided forget Fisher
# ─────────────────────────────────────────────────────────────────────────────

def compute_forget_fisher_mask(
    model, forget_dl, parameters, descriptions,
    class_to_forget, beta, device,
    target_density: float, max_batches: int,
    logger=None,
) -> torch.Tensor:
    """
    Variant (b): forget Fisher only — no retain information.

    Score = Z(F_f)   where F_f = E[(∂L_f/∂θ)²]   (eq. 7)

    This isolates the effect of the dual-Fisher design: by dropping F_r
    we lose the retain-aware suppression and risk over-erasing.
    """
    accum_f, _ = _accumulate_forget_fisher_and_gradmag(
        model, forget_dl, parameters, descriptions,
        class_to_forget, beta, device, max_batches,
    )

    global_f  = torch.cat([f.reshape(-1) for f in accum_f if f is not None])
    score     = _z(global_f)
    mask_flat = _topk_mask(score, target_density)

    if logger:
        active = mask_flat.sum().item()
        total  = mask_flat.numel()
        logger.info(f"[Mask/forget_fisher]  active={active:,} / {total:,}  density={active/total:.4f}")

    del accum_f
    torch.cuda.empty_cache()
    return mask_flat


# ─────────────────────────────────────────────────────────────────────────────
# (c) Gradient magnitude saliency (SalUn-style)
# ─────────────────────────────────────────────────────────────────────────────

def compute_salun_mask(
    model, forget_dl, parameters, descriptions,
    class_to_forget, beta, device,
    target_density: float, max_batches: int,
    logger=None,
) -> torch.Tensor:
    """
    Variant (c): gradient-magnitude saliency (SalUn-style).

    Score = Z(mean |∇L_f|)

    Unlike Fisher (which uses g²), SalUn uses the raw gradient magnitude.
    This is less sensitive to gradient scale but retains directionality.
    Reference: Fan et al., "Salun: Empowering Machine Unlearning via
    Gradient-Based Weight Saliency", ICLR 2024.
    """
    _, accum_mag = _accumulate_forget_fisher_and_gradmag(
        model, forget_dl, parameters, descriptions,
        class_to_forget, beta, device, max_batches,
    )

    global_mag = torch.cat([m.reshape(-1) for m in accum_mag if m is not None])
    score      = _z(global_mag)
    mask_flat  = _topk_mask(score, target_density)

    if logger:
        active = mask_flat.sum().item()
        total  = mask_flat.numel()
        logger.info(f"[Mask/salun]  active={active:,} / {total:,}  density={active/total:.4f}")

    del accum_mag
    torch.cuda.empty_cache()
    return mask_flat


# ─────────────────────────────────────────────────────────────────────────────
# (d) Dual Fisher combined score (SSU — proposed method)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dual_fisher_mask(
    model, forget_dl, remain_dl, parameters, descriptions,
    class_to_forget, beta, device,
    target_density: float, max_batches: int,
    lambda_tradeoff: float = 1.0,
    logger=None,
) -> torch.Tensor:
    """
    Variant (d): SSU dual Fisher combined score (proposed method).

    S = ½ Z(S_ratio) + ½ Z(S_diff)

    where:
        S_ratio = F_f / (F_r + ε)                      (eq. 10)
        S_diff  = F̂_f − λ·F̂_r                         (eq. 12)

    Both retain and forget Fisher are used so that salient forget
    parameters with low retain importance are prioritised.
    """
    # Forget Fisher + grad-mag (we only use Fisher here)
    accum_f, _ = _accumulate_forget_fisher_and_gradmag(
        model, forget_dl, parameters, descriptions,
        class_to_forget, beta, device, max_batches,
    )

    # Retain Fisher
    accum_r = _accumulate_retain_fisher(
        model, remain_dl, parameters, descriptions, device, max_batches,
    )

    # Build score (eqs. 10-13)
    global_f = torch.cat([f.reshape(-1) for f in accum_f if f is not None]).clamp(min=1e-10)
    global_r = torch.cat([r.reshape(-1) for r in accum_r if r is not None]).clamp(min=1e-10)

    # ratio  = global_f / (global_r + 1e-10)                          # eq. 10
    log_ratio = torch.log(global_f) - torch.log(global_r)
    f_std  = global_f.std().clamp(min=1e-10)
    r_std  = global_r.std().clamp(min=1e-10)
    diff   = (global_f / f_std) - lambda_tradeoff * (global_r / r_std)  # eq. 12
    # score  = 0.5 * _z(ratio) + 0.5 * _z(diff)                       # eq. 13
    score  = 0.5 * _z(log_ratio) + 0.5 * _z(diff)                       # eq. 13

    mask_flat = _topk_mask(score, target_density)

    if logger:
        active = mask_flat.sum().item()
        total  = mask_flat.numel()
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

MASK_VARIANT_CHOICES = ("random", "forget_fisher", "salun", "dual_fisher")


def build_mask(
    variant: str,
    model,
    parameters,
    forget_dl,
    remain_dl,
    descriptions,
    class_to_forget,
    beta,
    device,
    target_density: float,
    max_batches: int,
    lambda_tradeoff: float = 1.0,
    logger=None,
) -> torch.Tensor:
    """
    Dispatcher — returns mask_flat for the requested variant.

    Parameters
    ----------
    variant : str
        One of "random", "forget_fisher", "salun", "dual_fisher".
    """
    if variant not in MASK_VARIANT_CHOICES:
        raise ValueError(f"Unknown mask variant: {variant!r}. "
                         f"Choose from {MASK_VARIANT_CHOICES}.")

    if logger:
        logger.info(f"[Mask] Building mask with variant='{variant}'  "
                    f"density={target_density}  max_batches={max_batches}")

    if variant == "random":
        return compute_random_mask(parameters, target_density, device, logger)

    elif variant == "forget_fisher":
        return compute_forget_fisher_mask(
            model, forget_dl, parameters, descriptions,
            class_to_forget, beta, device,
            target_density, max_batches, logger,
        )

    elif variant == "salun":
        return compute_salun_mask(
            model, forget_dl, parameters, descriptions,
            class_to_forget, beta, device,
            target_density, max_batches, logger,
        )

    else:  # "dual_fisher"
        return compute_dual_fisher_mask(
            model, forget_dl, remain_dl, parameters, descriptions,
            class_to_forget, beta, device,
            target_density, max_batches, lambda_tradeoff, logger,
        )


# ─────────────────────────────────────────────────────────────────────────────
# NSFW variants
# (batches already carry 'txt'; no class labels or descriptions list needed)
# ─────────────────────────────────────────────────────────────────────────────

PSEUDO_CAPTION_NSFW = "a photo of a person wearing clothes"


def _accumulate_forget_fisher_and_gradmag_nsfw(
    model,
    forget_dl,
    parameters,
    pseudo_caption,
    beta,
    device,
    max_batches,
):
    """
    Fisher / grad-mag accumulation for NSFW forget set.

    Batches are dicts with 'jpg' and 'txt' already set (no label lookup).
    The pseudo prompt is a fixed caption that redirects the model.

    Returns
    -------
    accum_f   : list[Tensor]  — per-param squared gradient (Fisher approx)
    accum_mag : list[Tensor]  — per-param gradient magnitude |∇L_f|
    """
    criteria = torch.nn.MSELoss()
    model.eval()

    accum_f   = [torch.zeros_like(p) for p in parameters]
    accum_mag = [torch.zeros_like(p) for p in parameters]
    n_batches = 0

    for batch_idx, forget_batch in enumerate(
        tqdm(forget_dl, desc="[Mask/NSFW Fisher] forget batches")
    ):
        if batch_idx >= max_batches:
            break

        n_imgs       = forget_batch["jpg"].shape[0]
        pseudo_batch = {"jpg": forget_batch["jpg"], "txt": [pseudo_caption] * n_imgs}

        f_in, f_emb = model.get_input(forget_batch, model.first_stage_key)
        p_in, p_emb = model.get_input(pseudo_batch,  model.first_stage_key)

        t     = torch.randint(0, model.num_timesteps, (f_in.shape[0],), device=device).long()
        noise = torch.randn_like(f_in)

        f_out  = model.apply_model(model.q_sample(f_in, t, noise), t, f_emb)
        p_out  = model.apply_model(model.q_sample(p_in, t, noise), t, p_emb).detach()
        loss_f = criteria(f_out, p_out) * beta

        grads = torch.autograd.grad(loss_f, parameters, retain_graph=False, allow_unused=True)
        for i, g in enumerate(grads):
            if g is not None:
                accum_f[i]   += g.detach() ** 2
                accum_mag[i] += g.detach().abs()

        del f_out, p_out, f_in, p_in, loss_f, grads
        n_batches += 1

    n = max(n_batches, 1)
    for i in range(len(accum_f)):
        accum_f[i]   /= n
        accum_mag[i] /= n

    return accum_f, accum_mag


def _accumulate_retain_fisher_nsfw(
    model,
    remain_dl,
    parameters,
    device,
    max_batches,
):
    """Accumulate retain Fisher for NSFW (batches carry their own 'txt')."""
    model.eval()
    accum_r     = [torch.zeros_like(p) for p in parameters]
    remain_iter = iter(remain_dl)
    n_batches   = 0

    for batch_idx in tqdm(range(max_batches), desc="[Mask/NSFW Fisher] retain batches"):
        try:
            remain_batch = next(remain_iter)
        except StopIteration:
            break

        loss_r  = model.shared_step(remain_batch)[0]
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


def build_mask_nsfw(
    variant: str,
    model,
    parameters,
    forget_dl,
    remain_dl,
    beta,
    device,
    target_density: float,
    max_batches: int,
    pseudo_caption: str = PSEUDO_CAPTION_NSFW,
    lambda_tradeoff: float = 1.0,
    logger=None,
) -> torch.Tensor:
    """
    Dispatcher for NSFW mask building — same variants as build_mask but
    uses NSFW-specific Fisher accumulation (no class labels / descriptions).

    Parameters
    ----------
    variant : str
        One of "random", "forget_fisher", "salun", "dual_fisher".
    pseudo_caption : str
        Caption used as the forget redirect target (default: clothed person).
    """
    if variant not in MASK_VARIANT_CHOICES:
        raise ValueError(f"Unknown mask variant: {variant!r}. "
                         f"Choose from {MASK_VARIANT_CHOICES}.")

    if logger:
        logger.info(f"[Mask/NSFW] variant='{variant}'  "
                    f"density={target_density}  max_batches={max_batches}")

    if variant == "random":
        return compute_random_mask(parameters, target_density, device, logger)

    # ── accumulate Fisher / grad-mag on forget set ────────────────────────────
    accum_f, accum_mag = _accumulate_forget_fisher_and_gradmag_nsfw(
        model, forget_dl, parameters, pseudo_caption, beta, device, max_batches,
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
        model, remain_dl, parameters, device, max_batches,
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
