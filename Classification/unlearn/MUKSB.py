"""
Classification/unlearn/MUKSB.py
================================
MUKSB — Machine Unlearning via Kalai-Smorodinsky Bargaining
Classification component (ResNet / VGG on CIFAR-10, Tiny-ImageNet, CelebA, etc.)

Gradient Merge — Kalai-Smorodinsky (KS) Bargaining
----------------------------------------------------
Unlike Nash (MUNBa), KS satisfies the Monotonicity axiom rather than IIA.
For sequential unlearning — where the feasible gradient set changes every step
— Monotonicity is the natural requirement: if the feasible set expands,
neither player should lose ground.

KS closed-form solution (equal proportional-gain condition):

    g* ∝  ĝ_r + ĝ_f    where ĝ_i = g_i / ||g_i||

i.e. the normalised sum of the two unit gradient vectors. This bisects the
angle between gradients in normalised space, eliminating gradient dominance
by construction (not by post-hoc compensation).

Common proportional gain at the solution:

    λ_KS = ĝ_r · g̃* = ĝ_f · g̃*  = (1 + cos φ) / ||ĝ_r + ĝ_f||

Edge case (φ → π, anti-parallel gradients):
    ||ĝ_r + ĝ_f|| → 0  →  g* = 0  (no Pareto-improving update exists)
"""

import gc
import time
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
import utils

from .impl import iterative_unlearn
from .sam import SAM


def l1_regularization(model):
    params_vec = [param.view(-1) for param in model.parameters()]
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


# ─────────────────────────────────────────────────────────────────────────────
# KS bargaining core
# ─────────────────────────────────────────────────────────────────────────────

def ks_step(gr_flat: torch.Tensor, gf_flat: torch.Tensor, eps: float = 1e-8):
    """
    Kalai-Smorodinsky bargaining solution for gradient merging.

    Parameters
    ----------
    gr_flat : Tensor, shape (D,)
        Flattened retain gradient vector.
    gf_flat : Tensor, shape (D,)
        Flattened forget gradient vector.
    eps : float
        Numerical stability floor.

    Returns
    -------
    lambda_ks : Tensor (scalar)
        Common proportional gain: λ = ĝ_r · g̃* = ĝ_f · g̃*.
        Analogous role to alpha_r / alpha_f in Nash.
    cos_phi : Tensor (scalar)
        Cosine of the angle between the two gradients.
        Negative values indicate gradient conflict.
    g_star : Tensor, shape (D,)
        The KS-merged update direction (unit vector).
        Zero vector iff the gradients are exactly anti-parallel.
    """
    norm_gr = torch.clamp(torch.norm(gr_flat), min=1e-6)
    norm_gf = torch.clamp(torch.norm(gf_flat), min=1e-6)

    cos_phi = torch.clamp(
        torch.dot(gr_flat, gf_flat) / (norm_gr * norm_gf),
        -1.0 + eps, 1.0 - eps,
    )

    g_hat_r = gr_flat / norm_gr
    g_hat_f = gf_flat / norm_gf

    # KS equi-proportional-gain condition: bisect in normalised space.
    g_sum    = g_hat_r + g_hat_f
    norm_sum = torch.norm(g_sum)

    if norm_sum < 1e-6:
        # Gradients are anti-parallel — retain and forget are in direct conflict.
        # In this case the forget gradient alone is the KS-optimal update:
        # any step along g_r hurts forgetting, so we follow g_f exclusively.
        g_star    = g_hat_f          # unit vector in forget direction
        lambda_ks = torch.tensor(1.0, device=gr_flat.device)
    else:
        g_star_unit = g_sum / norm_sum
        lambda_ks   = torch.dot(g_hat_r, g_star_unit)
        g_star      = g_star_unit

    return lambda_ks, cos_phi, g_star


def _flatten_grads(params, grads):
    parts = []
    for p, g in zip(params, grads):
        parts.append(
            g.detach().reshape(-1) if g is not None
            else torch.zeros(p.numel(), device=p.device)
        )
    return torch.cat(parts)


def _unpack_to_grads(params, flat_vec):
    """Write flat_vec back into param.grad for each parameter."""
    offset = 0
    for p in params:
        n = p.numel()
        chunk = flat_vec[offset: offset + n].view_as(p)
        p.grad = chunk.clone() if p.grad is None else p.grad.copy_(chunk)
        offset += n


# ─────────────────────────────────────────────────────────────────────────────
# Main unlearning function
# ─────────────────────────────────────────────────────────────────────────────

def muksb(data_loaders, model, criterion, args, mask=None):
    """
    MUKSB unlearning for classification models.

    Replaces Nash bargaining (MUNBa) with KS bargaining.
    All other components (data split, optimizer, pruning mask) remain
    identical to MUNBa so results are directly comparable.

    Parameters
    ----------
    data_loaders : dict
        Keys: 'forget', 'retain', 'val', 'test'.
    model : nn.Module
        Pre-trained classification model to unlearn.
    criterion : callable
        Loss function (e.g. CrossEntropyLoss).
    args : Namespace
        Parsed argument namespace (see arg_parser.py).
    mask : dict or None
        Optional parameter-name → binary mask for sparse unlearning.
    """
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device = torch.device(f"cuda:{int(args.gpu)}")

    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    if not args.sam:
        optimizer = torch.optim.SGD(
            model.parameters(), args.unlearn_lr,
            momentum=args.momentum, weight_decay=args.weight_decay,
        )
    else:
        optimizer = SAM(
            filter(lambda p: p.requires_grad, model.parameters()),
            torch.optim.SGD, rho=0.05, adaptive=False,
            lr=args.unlearn_lr, momentum=args.momentum, weight_decay=5e-4,
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )

    losses = utils.AverageMeter()
    top1   = utils.AverageMeter()
    top1_u = utils.AverageMeter()
    loader_len = max(len(forget_loader), len(retain_loader))

    # Collect trainable parameters once for gradient manipulation
    params = [p for p in model.parameters() if p.requires_grad]

    print("KS bargaining initialised (no convex solver required).")

    for epoch in range(args.unlearn_epochs):
        start_time = time.time()
        model.train()
        print("Epoch #{}, Learning rate: {}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))

        i = 0
        start = time.time()
        for data_r, data_u in zip_longest(retain_loader, forget_loader, fillvalue=None):
            i += 1
            if (data_r is None) and (data_u is None):
                break
            elif (data_u is None) and (data_r is not None):
                # Only retain samples remain — plain retain step (same as MUNBa).
                image_r, target_r = data_r
                image_r, target_r = image_r.to(device), target_r.to(device)

                optimizer.zero_grad()
                output_r = model(image_r)
                loss = criterion(output_r, target_r)

                if args.with_l1:
                    current_alpha = args.alpha * (1 - epoch / args.unlearn_epochs)
                    loss = loss + current_alpha * l1_regularization(model)
                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.step()

                with torch.no_grad():
                    output_r = output_r.float()
                    loss = loss.float()
                    prec_r = utils.accuracy(output_r.data, target_r)[0]
                    losses.update(loss.item(), image_r.size(0))
                    top1.update(prec_r.item(), image_r.size(0))
                    torch.cuda.empty_cache()
                    gc.collect()

                if (i + 1) % 10 == 0:
                    print(f'Batch: {i+1:4d}, prec_r: {top1.val:.3f} ({top1.avg:.3f}), loss: {loss:.4f}')

            else:
                # Both retain and forget batches — KS bargaining update.
                image_r, target_r = data_r
                image_u, target_u = data_u
                image_r, target_r = image_r.to(device), target_r.to(device)
                image_u, target_u = image_u.to(device), target_u.to(device)

                # Random label (same forget objective as MUNBa)
                target_u_rl = torch.randint(0, args.num_classes, target_u.shape, device=device)

                optimizer.zero_grad()
                output_r = model(image_r)
                output_u = model(image_u)
                loss_r = criterion(output_r, target_r)
                loss_u = criterion(output_u, target_u_rl)

                # ── Compute per-task gradients ────────────────────────────────
                grads_r = torch.autograd.grad(loss_r, params, retain_graph=True)
                grads_f = torch.autograd.grad(loss_u, params, retain_graph=True)

                gr_flat = _flatten_grads(params, grads_r)
                gf_flat = _flatten_grads(params, grads_f)

                # ── KS bargaining: replace Nash weights with KS direction ─────
                lambda_ks, cos_phi, g_star = ks_step(gr_flat, gf_flat)
                print(f'lambda_ks: [{lambda_ks.item():.4f}]  cos_phi: {cos_phi.item():.4f}')

                del gr_flat, gf_flat, grads_r, grads_f

                # ── Apply merged gradient ─────────────────────────────────────
                optimizer.zero_grad()
                _unpack_to_grads(params, g_star)
                del g_star

                if args.with_l1:
                    current_alpha = args.alpha * (1 - epoch / args.unlearn_epochs)
                    l1_loss = current_alpha * l1_regularization(model)
                    l1_grads = torch.autograd.grad(l1_loss, params)
                    for p, lg in zip(params, l1_grads):
                        if p.grad is not None and lg is not None:
                            p.grad += lg.detach()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.step()

                with torch.no_grad():
                    output_r = output_r.float()
                    output_u = output_u.float()
                    loss = (loss_r + loss_u).float()
                    prec_r = utils.accuracy(output_r.data, target_r)[0]
                    prec_u = utils.accuracy(output_u.data, target_u)[0]
                    losses.update(loss.item(), image_r.size(0) + image_u.size(0))
                    top1.update(prec_r.item(), image_r.size(0))
                    top1_u.update(prec_u.item(), image_u.size(0))
                    torch.cuda.empty_cache()
                    gc.collect()

                if (i + 1) % 10 == 0:
                    print(f'Batch: {i+1:4d}, prec_u: {top1_u.val:.3f} ({top1_u.avg:.3f}), loss_u: {loss_u:.4f}, loss_r: {loss_r:.4f}')

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                          epoch, i, loader_len, end - start, loss=losses, top1=top1))
                start = time.time()

        scheduler.step()
        print("one epoch duration:{}".format(time.time() - start_time))

    return top1.avg
