"""
MUKSB direction ablation variants.

Three update-direction strategies for the retain/forget gradient merge:

  MUKSB_RawSum   (Variant A) — raw sum gr + gf, no normalisation
  MUKSB_MeanUnit (Variant B) — arithmetic mean of unit gradients

The full MUKSB method (KS bisector + harmonic-mean scaling) lives in MUKSB.py
and is used as-is for the "full" arm of the ablation.
"""

import gc
import json
import os
import time
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
import utils
from trainer import validate

from .sam import SAM


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        n     = p.numel()
        chunk = flat_vec[offset: offset + n].view_as(p)
        p.grad = chunk.clone() if p.grad is None else p.grad.copy_(chunk)
        offset += n


def l1_regularization(model):
    params_vec = [p.view(-1) for p in model.parameters()]
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


# ─────────────────────────────────────────────────────────────────────────────
# Direction strategies
# ─────────────────────────────────────────────────────────────────────────────

def _rawsum_direction(gr_flat: torch.Tensor, gf_flat: torch.Tensor, eps: float = 1e-8):
    """Variant A: raw sum of gradients, no unit-normalisation."""
    update = gr_flat + gf_flat
    cos_phi = (
        torch.dot(gr_flat, gf_flat)
        / (gr_flat.norm().clamp_min(eps) * gf_flat.norm().clamp_min(eps))
    )
    return update, cos_phi


def _meanunit_direction(gr_flat: torch.Tensor, gf_flat: torch.Tensor, eps: float = 1e-8):
    """Variant B: arithmetic mean of unit gradients."""
    g_hat_r = gr_flat / gr_flat.norm().clamp_min(eps)
    g_hat_f = gf_flat / gf_flat.norm().clamp_min(eps)
    update  = (g_hat_r + g_hat_f) * 0.5
    cos_phi = torch.dot(g_hat_r, g_hat_f)
    return update, cos_phi


# ─────────────────────────────────────────────────────────────────────────────
# Shared unlearning loop — parameterised by direction strategy
# ─────────────────────────────────────────────────────────────────────────────

def _run_variant(data_loaders, model, criterion, args, mask, direction_fn, tag):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device        = torch.device(f"cuda:{int(args.gpu)}")

    with_l1     = getattr(args, "with_l1",     False)
    alpha_l1    = getattr(args, "alpha",       1e-4)
    num_classes = getattr(args, "num_classes", 10)

    print(f"[{tag}] starting | with_l1={with_l1}")

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

    losses    = utils.AverageMeter()
    top1      = utils.AverageMeter()
    top1_u    = utils.AverageMeter()
    loader_len = max(len(forget_loader), len(retain_loader))

    cos_phi_history = []
    epoch_metrics      = []
    epoch_metrics_path = os.path.join(args.save_dir, "epoch_metrics.json")

    for epoch in range(args.unlearn_epochs):
        start_time = time.time()
        model.train()
        print(f"Epoch #{epoch}, LR: {optimizer.state_dict()['param_groups'][0]['lr']}")

        i = 0
        start = time.time()

        for data_r, data_u in zip_longest(retain_loader, forget_loader, fillvalue=None):
            i += 1
            if data_r is None and data_u is None:
                break

            # ── retain-only tail ──────────────────────────────────────────────
            elif data_u is None and data_r is not None:
                image_r, target_r = data_r
                image_r, target_r = image_r.to(device), target_r.to(device)

                optimizer.zero_grad()
                output_r = model(image_r)
                loss     = criterion(output_r, target_r)

                if with_l1:
                    current_alpha = alpha_l1 * (1 - epoch / args.unlearn_epochs)
                    loss = loss + current_alpha * l1_regularization(model)

                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.step()

                with torch.no_grad():
                    prec_r = utils.accuracy(output_r.float().data, target_r)[0]
                    losses.update(loss.item(), image_r.size(0))
                    top1.update(prec_r.item(), image_r.size(0))
                    torch.cuda.empty_cache()
                    gc.collect()

                if (i + 1) % 10 == 0:
                    print(f'Batch: {i+1:4d}, prec_r: {top1.val:.3f} ({top1.avg:.3f}), loss: {loss:.4f}')

            # ── both retain + forget batches ──────────────────────────────────
            else:
                image_r, target_r = data_r
                image_u, target_u = data_u
                image_r, target_r = image_r.to(device), target_r.to(device)
                image_u, target_u = image_u.to(device), target_u.to(device)

                target_u_rl = torch.randint(0, num_classes, target_u.shape, device=device)

                # ── retain gradient ───────────────────────────────────────────
                optimizer.zero_grad()
                output_r = model(image_r)
                loss_r   = criterion(output_r, target_r)
                grads_r  = torch.autograd.grad(loss_r, model.parameters(), retain_graph=False)
                gr_flat  = _flatten_grads(model.parameters(), grads_r)
                del grads_r

                # ── forget gradient ───────────────────────────────────────────
                optimizer.zero_grad()
                output_u = model(image_u)
                loss_u   = criterion(output_u, target_u_rl)
                grads_f  = torch.autograd.grad(loss_u, model.parameters(), retain_graph=False)
                gf_flat  = _flatten_grads(model.parameters(), grads_f)
                del grads_f

                # ── mask if supplied ──────────────────────────────────────────
                if mask:
                    mask_flat = torch.cat([
                        mask[name].view(-1)
                        for name, _ in model.named_parameters()
                        if name in mask
                    ])
                    gr_input = gr_flat[mask_flat.bool()]
                    gf_input = gf_flat[mask_flat.bool()]
                else:
                    gr_input = gr_flat
                    gf_input = gf_flat

                # ── direction strategy ────────────────────────────────────────
                update_input, cos_phi = direction_fn(gr_input, gf_input)
                cos_phi_history.append(cos_phi.item())

                # Expand back to full parameter space
                if mask:
                    update_full = torch.zeros_like(gr_flat)
                    update_full[mask_flat.bool()] = update_input
                else:
                    update_full = update_input
                del gr_flat, gf_flat, gr_input, gf_input, update_input

                optimizer.zero_grad()
                _unpack_to_grads(model.parameters(), update_full)
                del update_full

                if with_l1:
                    current_alpha = alpha_l1 * (1 - epoch / args.unlearn_epochs)
                    l1_loss  = current_alpha * l1_regularization(model)
                    l1_grads = torch.autograd.grad(l1_loss, model.parameters())
                    for p, lg in zip(model.parameters(), l1_grads):
                        if p.grad is not None and lg is not None:
                            p.grad += lg.detach()
                    del l1_grads

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                with torch.no_grad():
                    prec_r = utils.accuracy(output_r.float().data, target_r)[0]
                    prec_u = utils.accuracy(output_u.float().data, target_u)[0]
                    combined = loss_r + loss_u
                    losses.update(combined.item(), image_r.size(0) + image_u.size(0))
                    top1.update(prec_r.item(),   image_r.size(0))
                    top1_u.update(prec_u.item(), image_u.size(0))
                    torch.cuda.empty_cache()
                    gc.collect()

                if (i + 1) % 10 == 0:
                    avg_cos = float(np.mean(cos_phi_history[-10:])) if cos_phi_history else 0.0
                    print(
                        f'Batch: {i+1:4d}'
                        f'  prec_u: {top1_u.val:.3f} ({top1_u.avg:.3f})'
                        f'  loss_u: {loss_u:.4f}'
                        f'  loss_r: {loss_r:.4f}'
                        f'  cos_φ: {cos_phi.item():.4f}'
                        f'  avg_cos_φ(10): {avg_cos:.4f}'
                    )

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Time {3:.2f}'.format(
                        epoch, i, loader_len, end - start,
                        loss=losses, top1=top1,
                    )
                )
                start = time.time()

        scheduler.step()
        epoch_duration = time.time() - start_time
        print(f"[{tag}] Epoch {epoch} done | duration: {epoch_duration:.2f}s")

        # Evaluate with test transforms, then restore training transforms
        saved_transforms = {}
        for split_name, loader in data_loaders.items():
            ds = loader.dataset
            while hasattr(ds, "dataset"):
                ds = ds.dataset
            saved_transforms[split_name] = (ds, ds.transform, getattr(ds, "train", None))
            utils.dataset_convert_to_test(loader.dataset, args)

        acc_per_split = {}
        for split_name, loader in data_loaders.items():
            acc_per_split[split_name] = validate(loader, model, criterion, args)
            print(f"  Epoch {epoch} | {split_name} acc: {acc_per_split[split_name]:.3f}")

        for split_name, (ds, orig_transform, orig_train) in saved_transforms.items():
            ds.transform = orig_transform
            if orig_train is not None:
                ds.train = orig_train

        epoch_metrics.append({
            "epoch":    epoch,
            "accuracy": acc_per_split,
            "duration": epoch_duration,
        })
        with open(epoch_metrics_path, "w") as f:
            json.dump(epoch_metrics, f, indent=2)

    if cos_phi_history:
        print(
            f"[{tag}] cos_φ stats: "
            f"mean={np.mean(cos_phi_history):.4f}  "
            f"min={np.min(cos_phi_history):.4f}  "
            f"max={np.max(cos_phi_history):.4f}"
        )

    return top1.avg


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

def muksb_rawsum(data_loaders, model, criterion, args, mask=None):
    """Variant A: raw sum gr + gf, no unit normalisation."""
    return _run_variant(
        data_loaders, model, criterion, args, mask,
        direction_fn=_rawsum_direction,
        tag="MUKSB_RawSum",
    )


def muksb_meanunit(data_loaders, model, criterion, args, mask=None):
    """Variant B: arithmetic mean of unit gradients."""
    return _run_variant(
        data_loaders, model, criterion, args, mask,
        direction_fn=_meanunit_direction,
        tag="MUKSB_MeanUnit",
    )
