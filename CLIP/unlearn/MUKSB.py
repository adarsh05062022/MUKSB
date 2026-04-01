"""
CLIP/unlearn/MUKSB.py
======================
MUKSB — Machine Unlearning via Kalai-Smorodinsky Bargaining
CLIP component (ViT-B/32 on OxfordPets or similar vision-language datasets)

Gradient Merge — Kalai-Smorodinsky (KS) Bargaining
----------------------------------------------------
Replaces Nash bargaining (MUNBa) with the KS closed-form solution:

    g* ∝  ĝ_r + ĝ_f    where ĝ_i = g_i / ||g_i||

KS satisfies Monotonicity (not IIA), making it the natural choice for
sequential unlearning where the feasible gradient set shifts at every step.

Common proportional gain:
    λ_KS = ĝ_r · g̃* = ĝ_f · g̃*  = (1 + cos φ) / ||ĝ_r + ĝ_f||

No convex solver (cvxpy) required.
"""

import gc
import time
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
import utils


def l1_regularization(parameters):
    params_vec = [p.view(-1) for p in parameters]
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
        Flattened retain gradient.
    gf_flat : Tensor, shape (D,)
        Flattened forget gradient.
    eps : float
        Numerical stability floor.

    Returns
    -------
    lambda_ks : Tensor (scalar)
        Common proportional gain λ_KS.
    cos_phi : Tensor (scalar)
        Cosine of gradient angle (negative = conflict).
    g_star : Tensor, shape (D,)
        KS-merged update direction (unit vector; zero if anti-parallel).
    """
    norm_gr = torch.clamp(torch.norm(gr_flat), min=1e-6)
    norm_gf = torch.clamp(torch.norm(gf_flat), min=1e-6)

    cos_phi = torch.clamp(
        torch.dot(gr_flat, gf_flat) / (norm_gr * norm_gf),
        -1.0 + eps, 1.0 - eps,
    )

    g_hat_r = gr_flat / norm_gr
    g_hat_f = gf_flat / norm_gf

    g_sum    = g_hat_r + g_hat_f
    norm_sum = torch.norm(g_sum)

    if norm_sum < 1e-6:
        g_star    = torch.zeros_like(gr_flat)
        lambda_ks = torch.tensor(0.0, device=gr_flat.device)
    else:
        g_star    = g_sum / norm_sum
        lambda_ks = torch.dot(g_hat_r, g_star)

    return lambda_ks, cos_phi, g_star


def _flatten_grads(parameters, grads):
    parts = []
    for p, g in zip(parameters, grads):
        parts.append(
            g.detach().reshape(-1) if g is not None
            else torch.zeros(p.numel(), device=p.device)
        )
    return torch.cat(parts)


def _unpack_to_grads(parameters, flat_vec):
    offset = 0
    for p in parameters:
        n = p.numel()
        chunk = flat_vec[offset: offset + n].view_as(p)
        p.grad = chunk.clone() if p.grad is None else p.grad.copy_(chunk)
        offset += n


# ─────────────────────────────────────────────────────────────────────────────
# Main unlearning function
# ─────────────────────────────────────────────────────────────────────────────

def muksb(texts, data_loaders, model, args, class_name):
    """
    MUKSB unlearning for CLIP vision-language models.

    Replaces Nash bargaining (MUNBa) with KS bargaining.
    All other components (data split, optimizer, CLIP encoding) remain
    identical to MUNBa so results are directly comparable.

    Parameters
    ----------
    texts : Tensor
        Tokenised text prompts for all classes.
    data_loaders : dict
        Keys: 'forget', 'retain', 'val', 'test'.
    model : CLIP model
        Pre-trained CLIP model to unlearn.
    args : Namespace
        Parsed argument namespace (see arg_parser.py).
    class_name : list[str]
        Human-readable class names.
    """
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device = torch.device(f"cuda:{int(args.gpu)}")

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    # ── choose parameters to train ───────────────────────────────────────────
    parameters = []
    for param in model.parameters():
        param.requires_grad = False

    if args.mode == "text":
        print("Unfreezing text encoder (attention layers)")
        for name, param in model.transformer.named_parameters():
            if "attn" in name:
                param.requires_grad = True
                parameters.append(param)
    elif args.mode == "image":
        print("Unfreezing visual encoder (attention layers)")
        for name, param in model.visual.transformer.named_parameters():
            if "attn" in name:
                param.requires_grad = True
                parameters.append(param)
    elif args.mode == "all":
        print("Unfreezing all attention layers")
        for name, param in model.named_parameters():
            if "attn" in name:
                param.requires_grad = True
                parameters.append(param)

    optimizer = torch.optim.SGD(
        parameters, args.unlearn_lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )

    losses = utils.AverageMeter()
    top1   = utils.AverageMeter()
    top1_u = utils.AverageMeter()
    loader_len = max(len(forget_loader), len(retain_loader))
    logit_scale = 100

    print("KS bargaining initialised (no convex solver required).")

    for epoch in range(args.unlearn_epochs):
        start_time = time.time()
        model.train()
        print(f"Epoch #{epoch}, LR: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")

        i = 0
        start = time.time()
        for data_r, data_u in zip_longest(retain_loader, forget_loader, fillvalue=None):
            i += 1
            if data_r is None and data_u is None:
                break

            elif data_u is None:
                # Only retain samples — plain retain step.
                image_r, target_r = data_r
                image_r = image_r.to(device)
                target_r = target_r.to(device)

                optimizer.zero_grad()
                if args.mode == "text":
                    with torch.no_grad():
                        image_features = model.encode_image(image_r)
                    text_features = model.encode_text(texts)
                elif args.mode == "image":
                    image_features = model.encode_image(image_r)
                    with torch.no_grad():
                        text_features = model.encode_text(texts)
                else:
                    image_features = model.encode_image(image_r)
                    text_features  = model.encode_text(texts)

                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features  = text_features  / text_features.norm(dim=1, keepdim=True)
                cosine_similarity = logit_scale * image_features @ text_features.t()

                loss = criterion(cosine_similarity, target_r)
                if args.with_l1:
                    current_alpha = args.alpha * (1 - epoch / args.unlearn_epochs)
                    loss = loss + current_alpha * l1_regularization(parameters)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    prec_r = utils.accuracy(cosine_similarity, target_r)[0]
                    losses.update(loss.item(), image_r.size(0))
                    top1.update(prec_r.item(), image_r.size(0))
                    torch.cuda.empty_cache(); gc.collect()

                if (i + 1) % 10 == 0:
                    print(f"Batch: {i+1:4d}, prec_r: {top1.val:.3f} ({top1.avg:.3f}), loss: {loss:.4f}")

            else:
                # Both retain and forget batches — KS update.
                image_r, target_r = data_r
                image_u, target_u = data_u
                image_r, target_r = image_r.to(device), target_r.to(device)
                image_u, target_u = image_u.to(device), target_u.to(device)

                # Forget: assign random label (misclassification objective)
                target_u_rl = torch.randint(
                    0, args.num_classes, target_u.shape, device=device
                )

                # Concatenate for a single forward pass
                images = torch.cat((image_r, image_u), dim=0)
                bs = image_r.size(0)

                optimizer.zero_grad()
                if args.mode == "text":
                    with torch.no_grad():
                        image_features = model.encode_image(images)
                    text_features = model.encode_text(texts)
                elif args.mode == "image":
                    image_features = model.encode_image(images)
                    with torch.no_grad():
                        text_features = model.encode_text(texts)
                else:
                    image_features = model.encode_image(images)
                    text_features  = model.encode_text(texts)

                text_features    = text_features / text_features.norm(dim=-1, keepdim=True)
                image_features_r = image_features[:bs] / image_features[:bs].norm(dim=-1, keepdim=True)
                image_features_u = image_features[bs:] / image_features[bs:].norm(dim=-1, keepdim=True)

                cosine_similarity_r = logit_scale * image_features_r @ text_features.t()
                cosine_similarity_u = logit_scale * image_features_u @ text_features.t()

                loss_r = criterion(cosine_similarity_r, target_r)
                loss_u = args.beta * criterion(cosine_similarity_u, target_u_rl)

                # ── Compute per-task gradients ────────────────────────────────
                grads_r = torch.autograd.grad(loss_r, parameters, retain_graph=True)
                grads_f = torch.autograd.grad(loss_u, parameters, retain_graph=True)

                gr_flat = _flatten_grads(parameters, grads_r)
                gf_flat = _flatten_grads(parameters, grads_f)

                # ── KS bargaining: unit-vector sum ────────────────────────────
                lambda_ks, cos_phi, g_star = ks_step(gr_flat, gf_flat)

                if lambda_ks.item() == 0.0:
                    # Anti-parallel gradients — skip update
                    del gr_flat, gf_flat, grads_r, grads_f, g_star
                    torch.cuda.empty_cache(); gc.collect()
                    continue

                if (i + 1) % 2 == 0:
                    print(
                        f"Batch: {i+1:4d} | λ_KS={lambda_ks.item():.4f}"
                        f"  cos_φ={cos_phi.item():.4f}"
                        f"  loss_r={loss_r.item():.4f}  loss_u={loss_u.item():.4f}"
                    )

                del gr_flat, gf_flat, grads_r, grads_f

                # ── Apply KS gradient ────────────────────────────────────────
                optimizer.zero_grad()
                _unpack_to_grads(parameters, g_star)
                del g_star

                if args.with_l1:
                    current_alpha = args.alpha * (1 - epoch / args.unlearn_epochs)
                    l1_loss  = current_alpha * l1_regularization(parameters)
                    l1_grads = torch.autograd.grad(l1_loss, parameters)
                    for p, lg in zip(parameters, l1_grads):
                        if p.grad is not None and lg is not None:
                            p.grad += lg.detach()

                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                with torch.no_grad():
                    prec_r = utils.accuracy(cosine_similarity_r, target_r)[0]
                    prec_u = utils.accuracy(cosine_similarity_u, target_u)[0]
                    combined_loss = loss_r + loss_u
                    losses.update(combined_loss.item(), image_r.size(0) + image_u.size(0))
                    top1.update(prec_r.item(), image_r.size(0))
                    top1_u.update(prec_u.item(), image_u.size(0))
                    torch.cuda.empty_cache(); gc.collect()

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, loader_len, end - start,
                        loss=losses, top1=top1,
                    )
                )
                start = time.time()

        scheduler.step()
        print(f"Epoch {epoch} duration: {time.time() - start_time:.1f}s")
