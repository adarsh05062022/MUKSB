import time
import gc
import torch
import torch.nn as nn
import numpy as np
from itertools import zip_longest
from tqdm import tqdm

import utils
import clip

from .mask_utils import compute_dual_importance_mask


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def flatten_grads(parameters, grads):
    parts = []
    for p, g in zip(parameters, grads):
        if g is not None:
            parts.append(g.detach().reshape(-1))
        else:
            parts.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(parts)


def unpack_update_to_grads(parameters, flat_update, numel_list):

    offset = 0
    for p, numel in zip(parameters, numel_list):

        grad_chunk = flat_update[offset:offset + numel].view_as(p)

        if p.grad is None:
            p.grad = grad_chunk.clone()
        else:
            p.grad.copy_(grad_chunk)

        offset += numel


def l1_regularization(parameters):
    params_vec = [p.view(-1) for p in parameters]
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


# ---------------------------------------------------------
# MAIN METHOD
# ---------------------------------------------------------

def MaskedNash(texts, data_loaders, model, args, class_name):

    device = torch.device(f"cuda:{int(args.gpu)}")

    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Select parameters to train
    # -----------------------------------------------------

    parameters = []
    param_names = []

    for p in model.parameters():
        p.requires_grad = False

    if args.mode == "text":

        print("Training text encoder")

        for name, param in model.transformer.named_parameters():
            if "attn" in name:
                param.requires_grad = True
                parameters.append(param)
                param_names.append(name)

    elif args.mode == "image":

        print("Training visual encoder")

        for name, param in model.visual.transformer.named_parameters():
            if "attn" in name:
                param.requires_grad = True
                parameters.append(param)
                param_names.append(name)

    elif args.mode == "all":

        print("Training all attention layers")

        for name, param in model.named_parameters():
            if "attn" in name:
                param.requires_grad = True
                parameters.append(param)
                param_names.append(name)

    optimizer = torch.optim.SGD(
        parameters,
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(map(int, args.decreasing_lr.split(","))),
        gamma=0.1,
    )

    numel_list = [p.numel() for p in parameters]

    logit_scale = 100

    

    print("Mask ready")

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(args.unlearn_epochs):

        print("Epoch:", epoch)
        # -----------------------------------------------------
    # Compute Fisher mask
    # -----------------------------------------------------
        start_time = time.time()
        print("Computing Fisher importance mask...")
        model.eval()
        mask, mask_flat = compute_dual_importance_mask(
            model=model,
            forget_dl=forget_loader,
            remain_dl=retain_loader,
            parameters=parameters,
            param_names=param_names,
            descriptions=class_name,
            class_to_forget=0,
            beta=1.0,
            device=device,
            target_density=0.3,
            lambda_tradeoff=1.0,
        )

        model.train()

        

        loader_len = max(len(retain_loader), len(forget_loader))

        for data_r, data_f in zip_longest(
            retain_loader,
            forget_loader,
            fillvalue=None,
        ):

            if data_r is None or data_f is None:
                continue

            image_r, target_r = data_r
            image_f, target_f = data_f

            image_r = image_r.to(device)
            target_r = target_r.to(device)

            image_f = image_f.to(device)
            target_f = target_f.to(device)

            optimizer.zero_grad()

            # ---------------------------------------------
            # RETAIN LOSS
            # ---------------------------------------------

            image_features_r = model.encode_image(image_r)
            text_features = model.encode_text(texts)

            image_features_r = image_features_r / image_features_r.norm(
                dim=-1,
                keepdim=True,
            )

            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True,
            )

            logits_r = logit_scale * image_features_r @ text_features.t()

            loss_r = criterion(logits_r, target_r)

            # ---------------------------------------------
            # FORGET LOSS
            # ---------------------------------------------

            pseudo_labels = torch.randint(
                0,
                args.num_classes,
                target_f.shape,
                device=device,
            )

            image_features_f = model.encode_image(image_f)

            image_features_f = image_features_f / image_features_f.norm(
                dim=-1,
                keepdim=True,
            )

            logits_f = logit_scale * image_features_f @ text_features.t()

            loss_f = criterion(logits_f, pseudo_labels) * args.beta

            # ---------------------------------------------
            # Compute gradients
            # ---------------------------------------------

            grads_r = torch.autograd.grad(
                loss_r,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )

            grads_f = torch.autograd.grad(
                loss_f,
                parameters,
                allow_unused=True,
            )

            grads_r = [
                g.detach() if g is not None else None
                for g in grads_r
            ]

            grads_f = [
                g.detach() if g is not None else None
                for g in grads_f
            ]

            # ---------------------------------------------
            # Flatten gradients
            # ---------------------------------------------

            gr_flat = flatten_grads(parameters, grads_r)
            gf_flat = flatten_grads(parameters, grads_f)

            gr_masked = gr_flat[mask_flat]
            gf_masked = gf_flat[mask_flat]

            if gr_masked.numel() == 0:
                continue

            # ---------------------------------------------
            # Nash bargaining weights
            # ---------------------------------------------

            norm_gr = torch.clamp(torch.norm(gr_masked), min=1e-6)
            norm_gf = torch.clamp(torch.norm(gf_masked), min=1e-6)

            cos_phi = torch.clamp(
                torch.dot(gr_masked, gf_masked)
                / (norm_gr * norm_gf),
                -1.0 + 1e-6,
                1.0 - 1e-6,
            )

            sin_sq = torch.clamp(1 - cos_phi ** 2, min=0.0)

            if sin_sq < 1e-6:

                alpha_r = 0.5 / norm_gr
                alpha_f = 0.5 / norm_gf

            else:

                alpha_r = (1 / norm_gr) * torch.sqrt(
                    (1 - cos_phi) / (sin_sq + 1e-8)
                )

                alpha_f = (1 / norm_gf) * torch.sqrt(
                    sin_sq * (1 - cos_phi)
                )

            update = alpha_r * gr_masked + alpha_f * gf_masked

            # ---------------------------------------------
            # Scatter masked update
            # ---------------------------------------------

            update_full = torch.zeros_like(gr_flat)

            update_full[mask_flat] = update

            unpack_update_to_grads(
                parameters,
                update_full,
                numel_list,
            )

            nn.utils.clip_grad_norm_(parameters, 1.0)

            optimizer.step()

            del update_full, gr_flat, gf_flat

            torch.cuda.empty_cache()
            gc.collect()

        scheduler.step()

        print(
            "Epoch finished in:",
            time.time() - start_time,
        )