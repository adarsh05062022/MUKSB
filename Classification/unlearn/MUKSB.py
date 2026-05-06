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
# KS bargaining core  (identical to SD/MUKSB_cls_magnitude.py)
# ─────────────────────────────────────────────────────────────────────────────

def ks_step(
    gr_flat: torch.Tensor,
    gf_flat: torch.Tensor,
    eps: float = 1e-8,
):
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
        zero = torch.zeros_like(gr_flat)
        return (
            torch.tensor(0.0, device=gr_flat.device),
            cos_phi,
            zero,
            torch.tensor(0.0, device=gr_flat.device),
        )

    g_star          = g_sum / norm_sum
    effective_scale = 2.0 * norm_gr * norm_gf / (norm_gr + norm_gf)
    lambda_ks       = torch.dot(g_hat_r, g_star)

    return lambda_ks, cos_phi, g_star, effective_scale


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
# Main unlearning function
# ─────────────────────────────────────────────────────────────────────────────

def muksb(data_loaders, model, criterion, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device        = torch.device(f"cuda:{int(args.gpu)}")

    with_l1     = getattr(args, "with_l1",     False)
    alpha_l1    = getattr(args, "alpha",       1e-4)
    num_classes = getattr(args, "num_classes", 10)

    print(f"[MUKSB] KS bargaining | with_l1={with_l1}")

    # ── optimiser (mirrors MUNBa) ─────────────────────────────────────────────
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

    losses  = utils.AverageMeter()
    top1    = utils.AverageMeter()
    top1_u  = utils.AverageMeter()
    loader_len = max(len(forget_loader), len(retain_loader))

    # Conflict diagnostics
    skipped_steps   = 0
    total_steps     = 0
    cos_phi_history = []
    conflict_log    = []   # per-step gradient-cosine log for plot_conflict.py

    epoch_metrics      = []
    epoch_metrics_path = os.path.join(args.save_dir, "epoch_metrics.json")

    for epoch in range(args.unlearn_epochs):
        start_time = time.time()
        model.train()
        print(f"Epoch #{epoch}, Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

        i = 0
        start = time.time()

        for data_r, data_u in zip_longest(retain_loader, forget_loader, fillvalue=None):
            i += 1
            if data_r is None and data_u is None:
                break

            # ── retain-only tail (forget loader exhausted) ────────────────────
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

            # ── both retain + forget batches available: KS merge ──────────────
            else:
                image_r, target_r = data_r
                image_u, target_u = data_u
                image_r, target_r = image_r.to(device), target_r.to(device)
                image_u, target_u = image_u.to(device), target_u.to(device)

                # Random relabelling of the forget batch (same as MUNBa)
                target_u_rl = torch.randint(0, num_classes, target_u.shape, device=device)

                total_steps += 1

                # ── compute retain gradient ───────────────────────────────────
                optimizer.zero_grad()
                output_r = model(image_r)
                loss_r   = criterion(output_r, target_r)
                grads_r  = torch.autograd.grad(loss_r, model.parameters(), retain_graph=False)
                gr_flat  = _flatten_grads(model.parameters(), grads_r)
                del grads_r

                # ── compute forget gradient ───────────────────────────────────
                optimizer.zero_grad()
                output_u = model(image_u)
                loss_u   = criterion(output_u, target_u_rl)
                grads_f  = torch.autograd.grad(loss_u, model.parameters(), retain_graph=False)
                gf_flat  = _flatten_grads(model.parameters(), grads_f)
                del grads_f

                # ── apply sparse mask in gradient space ───────────────────────
                if mask:
                    # Build a boolean flat vector from the per-param mask dict
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

                # ── KS bargaining ─────────────────────────────────────────────
                lambda_ks, cos_phi, g_star, effective_scale = ks_step(
                    gr_input, gf_input,
                )
                cos_phi_history.append(cos_phi.item())

                # Gradient-conflict log: g_r vs g_f, and each vs MUKSB / MOO-sum updates
                with torch.no_grad():
                    _g_muksb = effective_scale * g_star
                    _g_naive = gr_input + gf_input
                    def _cos(a, b, _eps=1e-8):
                        n = a.norm().clamp_min(_eps) * b.norm().clamp_min(_eps)
                        return (a @ b / n).item()
                    conflict_log.append({
                        "epoch": epoch,
                        "step":  total_steps,
                        "cos_rf":      cos_phi.item(),
                        "cos_f_muksb": _cos(gf_input, _g_muksb),
                        "cos_r_muksb": _cos(gr_input, _g_muksb),
                        "cos_f_naive": _cos(gf_input, _g_naive),
                        "cos_r_naive": _cos(gr_input, _g_naive),
                    })
                    del _g_muksb, _g_naive

                # Anti-parallel: g_star is zero — skip update
                if torch.norm(g_star).item() < 1e-6:
                    skipped_steps += 1
                    del gr_flat, gf_flat, gr_input, gf_input, g_star
                    if (i + 1) % 10 == 0:
                        print(
                            f'Batch: {i+1:4d}, anti-parallel gradients '
                            f'(cos_φ={cos_phi.item():.3f}), skipping update'
                        )
                    continue

                g_star_scaled = effective_scale * g_star
                del gr_input, gf_input

                # Expand back to full parameter space
                if mask:
                    update_full = torch.zeros_like(gr_flat)
                    update_full[mask_flat.bool()] = g_star_scaled
                else:
                    update_full = g_star_scaled
                del gr_flat, gf_flat, g_star, g_star_scaled

                # Write KS update into model gradients
                optimizer.zero_grad()
                _unpack_to_grads(model.parameters(), update_full)
                del update_full

                # Optional L1 regularisation added on top of KS direction
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
                    skip_rate = skipped_steps / max(total_steps, 1)
                    avg_cos   = float(np.mean(cos_phi_history[-10:])) if cos_phi_history else 0.0
                    print(
                        f'Batch: {i+1:4d}'
                        f'  prec_u: {top1_u.val:.3f} ({top1_u.avg:.3f})'
                        f'  loss_u: {loss_u:.4f}'
                        f'  loss_r: {loss_r:.4f}'
                        f'  λ_KS: {lambda_ks.item():.4f}'
                        f'  cos_φ: {cos_phi.item():.4f}'
                        f'  avg_cos_φ(10): {avg_cos:.4f}'
                        f'  eff_scale: {effective_scale.item():.4e}'
                        f'  skip_rate: {skip_rate:.3f}'
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
        epoch_duration  = time.time() - start_time
        skip_rate_epoch = skipped_steps / max(total_steps, 1)
        print(
            f"Epoch {epoch} done | "
            f"duration: {epoch_duration:.2f}s | "
            f"cumulative skip_rate: {skip_rate_epoch:.3f}"
        )

        # Evaluate with test transforms (no augmentation), then restore training transforms
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
            "epoch": epoch,
            "accuracy": acc_per_split,
            "skip_rate": skip_rate_epoch,
            "duration": epoch_duration,
        })
        with open(epoch_metrics_path, "w") as f:
            json.dump(epoch_metrics, f, indent=2)

    # Dump gradient-conflict log for the plotting script
    with open(os.path.join(args.save_dir, "conflict_log.json"), "w") as f:
        json.dump(conflict_log, f)

    # ── final diagnostics ─────────────────────────────────────────────────────
    final_skip_rate = skipped_steps / max(total_steps, 1)
    print(
        f"[MUKSB] Anti-parallel skips: {skipped_steps}/{total_steps} "
        f"({final_skip_rate * 100:.1f}%)"
    )
    if cos_phi_history:
        print(
            f"[MUKSB] cos_φ stats: "
            f"mean={np.mean(cos_phi_history):.4f}  "
            f"min={np.min(cos_phi_history):.4f}  "
            f"max={np.max(cos_phi_history):.4f}"
        )

    return top1.avg
