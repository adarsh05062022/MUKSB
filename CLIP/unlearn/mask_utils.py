import clip
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# ---------------------------------------------------------
# Fisher accumulation
# ---------------------------------------------------------

def _accumulate_fisher(
    model,
    forget_dl,
    remain_dl,
    parameters,
    texts,
    beta,
    device,
    max_batches=80
):

    criterion = nn.CrossEntropyLoss()

    model.eval()

    if isinstance(texts, list):
        texts = clip.tokenize(texts).to(device)

    accum_fisher_f = [torch.zeros_like(p) for p in parameters]
    accum_fisher_r = [torch.zeros_like(p) for p in parameters]

    remain_iter = iter(remain_dl)

    logit_scale = 100

    for batch_idx, (forget_images, forget_labels) in enumerate(
        tqdm(forget_dl, desc="[Mask] Accumulating Fisher")
    ):

        if batch_idx >= max_batches:
            break

        forget_images = forget_images.to(device)
        forget_labels = forget_labels.to(device)

        try:
            remain_images, remain_labels = next(remain_iter)
        except StopIteration:
            remain_iter = iter(remain_dl)
            remain_images, remain_labels = next(remain_iter)

        remain_images = remain_images.to(device)
        remain_labels = remain_labels.to(device)

        # -------------------------------------------------
        # FORGET FISHER
        # -------------------------------------------------

        image_features = model.encode_image(forget_images)
        text_features = model.encode_text(texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        pseudo_labels = torch.randint(
            0,
            logits.shape[1],
            forget_labels.shape,
            device=device,
        )

        loss_f = criterion(logits, pseudo_labels) * beta

        grads_f = torch.autograd.grad(
            loss_f,
            parameters,
            retain_graph=False,
            allow_unused=True
        )

        for i, g in enumerate(grads_f):
            if g is not None:
                accum_fisher_f[i] += g.detach() ** 2

        del loss_f, grads_f

        # -------------------------------------------------
        # RETAIN FISHER
        # -------------------------------------------------

        image_features = model.encode_image(remain_images)
        
        text_features = model.encode_text(texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        loss_r = criterion(logits, remain_labels)

        grads_r = torch.autograd.grad(
            loss_r,
            parameters,
            retain_graph=False,
            allow_unused=True
        )

        for i, g in enumerate(grads_r):
            if g is not None:
                accum_fisher_r[i] += g.detach() ** 2

        del loss_r, grads_r

    n = min(len(forget_dl), max_batches)

    for i in range(len(accum_fisher_f)):
        accum_fisher_f[i] /= n
        accum_fisher_r[i] /= n

    return accum_fisher_f, accum_fisher_r


# ---------------------------------------------------------
# Build mask
# ---------------------------------------------------------

def _build_mask_from_fisher(
    parameters,
    accum_fisher_f,
    accum_fisher_r,
    target_density,
    lambda_tradeoff,
    importance_variant,
    param_names=None,
    logger=None
):

    flat_f = []
    flat_r = []

    valid_indices = []

    for i, (f, r) in enumerate(zip(accum_fisher_f, accum_fisher_r)):

        if f is None or r is None:
            continue

        flat_f.append(f.reshape(-1))
        flat_r.append(r.reshape(-1))

        valid_indices.append(i)

    global_f = torch.cat(flat_f)
    global_r = torch.cat(flat_r)

    ratio = global_f / (global_r + 1e-10)

    f_std = global_f.std().clamp(min=1e-10)
    r_std = global_r.std().clamp(min=1e-10)

    diff = (global_f / f_std) - lambda_tradeoff * (global_r / r_std)

    def z(x):
        return (x - x.mean()) / (x.std().clamp(min=1e-10))

    ratio_z = z(ratio)
    diff_z = z(diff)

    if importance_variant == "ratio":
        score = ratio_z

    elif importance_variant == "difference":
        score = diff_z

    else:
        score = 0.5 * ratio_z + 0.5 * diff_z

    k = max(1, int(target_density * score.numel()))

    top_indices = torch.topk(score, k).indices

    mask_flat = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    mask_flat[top_indices] = True

    # -------------------------------------------------
    # rebuild masks per layer
    # -------------------------------------------------

    masks = [None] * len(accum_fisher_f)

    offset = 0

    for i, f in enumerate(accum_fisher_f):

        if f is None:
            continue

        n = f.numel()

        masks[i] = mask_flat[offset:offset + n].reshape(f.shape)

        offset += n

    if logger:

        active = mask_flat.sum().item()
        total = mask_flat.numel()

        logger.info(
            f"[Mask] density={active/total:.4f} "
            f"active={active} total={total}"
        )

    return masks, mask_flat


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def compute_dual_importance_mask(
    model,
    forget_dl,
    remain_dl,
    parameters,
    param_names,
    descriptions,
    class_to_forget,
    beta,
    device,
    target_density=0.10,
    lambda_tradeoff=1.0,
    importance_variant="both",
    previous_mask_flat=None,
    ema_alpha=0.3,
    logger=None,
    max_fisher_batches=80
):

    if logger:
        logger.info("[Mask] Computing Fisher importance")

    accum_fisher_f, accum_fisher_r = _accumulate_fisher(
        model,
        forget_dl,
        remain_dl,
        parameters,
        descriptions,
        beta,
        device,
        max_batches=max_fisher_batches
    )

    masks, mask_flat = _build_mask_from_fisher(
        parameters,
        accum_fisher_f,
        accum_fisher_r,
        target_density,
        lambda_tradeoff,
        importance_variant,
        param_names=param_names,
        logger=logger
    )

    # -------------------------------------------------
    # EMA smoothing
    # -------------------------------------------------

    if previous_mask_flat is not None:

        soft_new = mask_flat.float()
        soft_prev = previous_mask_flat.float()

        blended = ema_alpha * soft_new + (1 - ema_alpha) * soft_prev

        k = mask_flat.sum().item()

        top_indices = torch.topk(blended, k).indices

        mask_flat_ema = torch.zeros_like(mask_flat)
        mask_flat_ema[top_indices] = True

        masks_ema = [None] * len(masks)

        offset = 0

        for i, m in enumerate(masks):

            if m is None:
                continue

            n = m.numel()

            masks_ema[i] = mask_flat_ema[offset:offset + n].reshape(m.shape)

            offset += n

        return masks_ema, mask_flat_ema

    return masks, mask_flat