"""
Classification/unlearn/IU.py
=============================
IU — Influence-Function-based Unlearning
Izzo et al., "Approximate Data Deletion from Machine Learning Models",
AISTATS 2021.  (reference [32] in MUNBa paper, Table 1)

Algorithm
---------
The goal is to reverse the training contribution of the forget set.
Using a diagonal empirical Fisher approximation for the Hessian:

    θ_new = θ  +  (F_retain + λI)^{-1}  ·  ∇L_forget(θ)

Where:
    F_retain  = diagonal empirical Fisher on the retain set
              = E_{(x,y) ~ D_r} [ (∇ log p(y|x))^2 ]   (element-wise)
    ∇L_forget = gradient of the cross-entropy loss over the forget set
    λ         = damping term for numerical stability (args.iu_damping, default 1e-3)

This is a ONE-SHOT method — no training loop is needed.
The Newton step adds back the forget contribution, effectively un-training.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Diagonal Fisher Information Matrix
# ─────────────────────────────────────────────────────────────────────────────

def _diagonal_fisher(model, retain_loader, device):
    """
    Compute the diagonal empirical Fisher Information Matrix on the retain set.

    F[i] = (1/N) * Σ  (∂ log p(y|x) / ∂ θ_i)^2

    Returns list of Tensors, one per parameter, same shape as parameter.
    """
    model.eval()
    fisher = [torch.zeros_like(p) for p in model.parameters()]
    total  = 0

    for images, targets in tqdm(retain_loader, desc="[IU] Fisher (retain)"):
        images, targets = images.to(device), targets.to(device)
        log_probs = torch.log_softmax(model(images), dim=-1)
        bs = images.size(0)

        for i in range(bs):
            grad = torch.autograd.grad(
                log_probs[i, targets[i]],
                model.parameters(),
                retain_graph=True,
                create_graph=False,
            )
            with torch.no_grad():
                for j, g in enumerate(grad):
                    fisher[j] += g ** 2

        total += bs

    with torch.no_grad():
        for j in range(len(fisher)):
            fisher[j] /= total

    return fisher


# ─────────────────────────────────────────────────────────────────────────────
# Main unlearning function
# ─────────────────────────────────────────────────────────────────────────────

def IU(data_loaders, model, criterion, args, mask=None):
    """
    IU — Influence Unlearning (one-shot Newton step, no training loop).

    Parameters
    ----------
    data_loaders : dict   — keys: 'forget', 'retain', 'val', 'test'
    model        : nn.Module
    criterion    : callable  (CrossEntropyLoss)
    args         : Namespace
        Uses: gpu, num_classes,
              iu_damping (default 1e-3) — Tikhonov regularisation λ,
              iu_scale   (default 1.0)  — scale on the Newton step.
    mask         : ignored
    """
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device        = torch.device(f"cuda:{int(args.gpu)}")
    damping       = getattr(args, "iu_damping", 1e-3)
    scale         = getattr(args, "iu_scale",   1.0)

    # ── Step 1: diagonal Fisher on retain set ────────────────────────────────
    print("[IU] Computing diagonal Fisher on retain set …")
    fisher = _diagonal_fisher(model, retain_loader, device)

    # ── Step 2: gradient of forget loss ─────────────────────────────────────
    print("[IU] Computing forget gradient …")
    model.eval()
    model.zero_grad()
    n_forget = 0

    for images, targets in tqdm(forget_loader, desc="[IU] Forget grad"):
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        loss   = criterion(output, targets) * images.size(0)
        loss.backward()
        n_forget += images.size(0)

    # ── Step 3: Newton step  θ ← θ + (F + λI)^{-1} · g_f ───────────────────
    print("[IU] Applying Newton step …")
    with torch.no_grad():
        for j, p in enumerate(model.parameters()):
            if p.grad is not None:
                g_f   = p.grad / n_forget                   # mean gradient
                h_inv = 1.0 / (fisher[j] + damping)         # diagonal H^{-1}
                p.data += scale * h_inv * g_f
            p.grad = None

    print("[IU] Done — one-shot update applied.")
    return 0.0
