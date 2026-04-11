"""
Classification/unlearn/SalUn.py
================================
SalUn — Saliency-based Machine Unlearning
Fan et al., "SalUn: Empowering Machine Unlearning via Gradient-Based
Weight Saliency in Both Image Classification and Generation", ICLR 2024.

Algorithm
---------
1. Compute per-parameter |∇L_forget| over the forget set.
2. Threshold at density ρ (default 0.5) → binary saliency mask (top-ρ by magnitude).
3. Epoch loop:
   a. Minimise CE(random-label) on forget set — only masked params updated.
   b. Minimise CE(true-label)   on retain set — only masked params updated.
"""

import gc
import time

import torch
import torch.nn as nn
import utils


# ─────────────────────────────────────────────────────────────────────────────
# Saliency mask generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_saliency_mask(model, forget_loader, criterion, device, density=0.5):
    """
    Accumulate |∇L_forget| over the full forget set, then keep the top-ρ
    fraction of parameters by gradient magnitude.

    Returns
    -------
    mask : dict  param_name → binary float Tensor (same shape as param)
                 1 = update allowed, 0 = frozen.
    """
    gradients = {n: torch.zeros_like(p, device=device)
                 for n, p in model.named_parameters()}
    model.eval()

    for images, targets in forget_loader:
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        # gradient ascent direction: negate loss
        loss = -criterion(output, targets)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    gradients[n] += p.grad.data.abs()

    # global threshold at (1 - density) quantile
    all_grads = torch.cat([g.flatten() for g in gradients.values()])
    threshold = torch.quantile(all_grads, 1.0 - density)

    mask = {}
    for n, g in gradients.items():
        mask[n] = (g >= threshold).float()

    kept  = sum(m.sum().item() for m in mask.values())
    total = sum(m.numel()      for m in mask.values())
    print(f"[SalUn] Mask: density={density:.2f}, "
          f"kept {kept:.0f}/{total} params ({100 * kept / total:.1f}%)")
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Main unlearning function
# ─────────────────────────────────────────────────────────────────────────────

def salun(data_loaders, model, criterion, args, mask=None):
    """
    SalUn unlearning for classification models.

    Parameters
    ----------
    data_loaders : dict   — keys: 'forget', 'retain', 'val', 'test'
    model        : nn.Module
    criterion    : callable  (CrossEntropyLoss)
    args         : Namespace
        Uses: gpu, unlearn_lr, momentum, weight_decay, decreasing_lr,
              unlearn_epochs, num_classes, print_freq,
              salun_density (default 0.5).
    mask         : ignored — SalUn generates its own saliency mask.
    """
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device        = torch.device(f"cuda:{int(args.gpu)}")
    density       = getattr(args, "salun_density", 0.5)
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    # ── Step 1: build saliency mask ───────────────────────────────────────────
    print("[SalUn] Building saliency mask from forget set …")
    saliency_mask = _build_saliency_mask(
        model, forget_loader, criterion, device, density
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(), args.unlearn_lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )

    losses = utils.AverageMeter()
    top1   = utils.AverageMeter()
    loader_len = len(retain_loader)

    for epoch in range(args.unlearn_epochs):
        start_time = time.time()
        model.train()
        print(f"[SalUn] Epoch #{epoch}, "
              f"LR: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")

        # ── Step 2: random-label fine-tune on forget set ─────────────────────
        for images, targets in forget_loader:
            images    = images.to(device)
            targets_rl = torch.randint(
                0, args.num_classes, targets.shape, device=device
            )
            optimizer.zero_grad()
            output = model(images)
            loss   = criterion(output, targets_rl)
            loss.backward()

            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.grad.mul_(saliency_mask[n])

            optimizer.step()

        # ── Step 3: retain fine-tune ─────────────────────────────────────────
        start = time.time()
        for i, (images, targets) in enumerate(retain_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss   = criterion(output, targets)
            loss.backward()

            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.grad.mul_(saliency_mask[n])

            optimizer.step()

            with torch.no_grad():
                prec = utils.accuracy(output.float().data, targets)[0]
                losses.update(loss.item(), images.size(0))
                top1.update(prec.item(), images.size(0))
                torch.cuda.empty_cache()
                gc.collect()

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Retain Acc {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, loader_len, end - start,
                        loss=losses, top1=top1,
                    )
                )
                start = time.time()

        scheduler.step()
        print(f"[SalUn] Epoch {epoch} done: retain_acc={top1.avg:.3f}  "
              f"({time.time() - start_time:.1f}s)")

    return top1.avg
