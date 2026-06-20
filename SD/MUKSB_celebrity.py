import argparse
import gc
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Vendored taming-transformers (needed by ldm.models.autoencoder). The other SD
# trainers rely on the caller exporting PYTHONPATH; wire it up here so this
# script is self-sufficient.
_SRC_TAMING = os.path.join(_THIS_DIR, "src", "taming-transformers")
if os.path.isdir(_SRC_TAMING) and _SRC_TAMING not in sys.path:
    sys.path.insert(0, _SRC_TAMING)

from logger.logger import setup_logger
from train_scripts.convertModels import savemodelDiffusers
from train_scripts.dataset import (
    setup_celebrity_forget_remain_data,
    setup_model,
)
from mask_variants import build_mask, MASK_VARIANT_CHOICES


EXTRA = "celeb"


# ─────────────────────────────────────────────────────────────────────────────
# KS bargaining core — magnitude-aware (identical to MUKSB_cls.py)
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

    g_star = g_sum / norm_sum

    # SCALE: harmonic mean of gradient norms (conservative, smaller-norm dominated)
    effective_scale = 2.0 * norm_gr * norm_gf / (norm_gr + norm_gf)

    # diagnostic: common proportional gain
    lambda_ks = torch.dot(g_hat_r, g_star)

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
        n = p.numel()
        chunk = flat_vec[offset: offset + n].view_as(p)
        p.grad = chunk.clone() if p.grad is None else p.grad.copy_(chunk)
        offset += n


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def l1_regularization(parameters):
    return torch.linalg.norm(
        torch.cat([p.view(-1) for p in parameters]), ord=1
    )


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def select_parameters(model, train_method):
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        keep = False
        if train_method == "full":
            keep = True
        elif train_method == "noxattn":
            keep = not (name.startswith("out.") or "attn2" in name or "time_embed" in name)
        elif train_method == "selfattn":
            keep = "attn1" in name
        elif train_method == "xattn":
            keep = "attn2" in name
        elif train_method == "notime":
            keep = not (name.startswith("out.") or "time_embed" in name)
        elif train_method == "xlayer":
            keep = "attn2" in name and (
                "output_blocks.6." in name or "output_blocks.8." in name
            )
        elif train_method == "selflayer":
            keep = "attn1" in name and (
                "input_blocks.4." in name or "input_blocks.7." in name
            )
        if keep:
            parameters.append(param)
    return parameters


def save_model(
    model, name, num,
    compvis_config_file=None, diffusers_config_file=None,
    device="cpu", save_compvis=False, save_diffusers=True,
):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    path = (
        f"{folder_path}/{name}-epoch_{num}.pt" if num is not None
        else f"{folder_path}/{name}.pt"
    )
    if save_diffusers:
        torch.save(model.state_dict(), path)
        print("Saving model in Diffusers format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file,
            device=device, num=num,
        )
    if not save_compvis and os.path.exists(path):
        os.remove(path)


def save_history(losses, name):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(v) + "\n" for v in losses])
    if len(losses) >= 1:
        w = min(3, len(losses))
        v = np.convolve(losses, np.ones(w) / w, mode="valid")
        plt.figure()
        plt.plot(v, label="loss")
        plt.legend(loc="upper left")
        plt.title("MUKSB celebrity training loss")
        plt.xlabel("Step"); plt.ylabel("Loss")
        plt.savefig(f"{folder_path}/loss.png")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main unlearning function — celebrity identity forgetting
# ─────────────────────────────────────────────────────────────────────────────

def MUKSB(
    class_to_forget,
    train_method,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_variant,
    mask_density,
    lambda_tradeoff,
    diffusers_config_path,
    device,
    image_size,
    ddim_steps,
    with_l1,
    beta,
    alpha,
    anchor_mode,
    anchor_prompt,
    logger,
):
    total_start = time.time()
    logger.info("======== MUKSB CELEBRITY (KS Bargaining) TRAINING STARTED ========")
    logger.info(f"class_to_forget={class_to_forget}  train_method={train_method}  "
                f"anchor_mode={anchor_mode}")

    # ── model + data ─────────────────────────────────────────────────────────
    model    = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()

    forget_dl, remain_dl, descriptions = setup_celebrity_forget_remain_data(
        class_to_forget, batch_size, image_size
    )
    num_classes = len(descriptions)
    num_forget  = len(forget_dl.dataset)
    logger.info(f"Celebrities: {num_classes} | forget idx {class_to_forget} "
                f"('{descriptions[class_to_forget]}') | forget samples: {num_forget} "
                f"| retain samples: {len(remain_dl.dataset)}")

    # ── parameter selection ───────────────────────────────────────────────────
    parameters = select_parameters(model, train_method)
    logger.info(f"Trainable params: {sum(p.numel() for p in parameters):,}")

    optimizer = torch.optim.Adam(parameters, lr=lr)

    # ── build importance mask (if requested) ─────────────────────────────────
    if mask_variant is not None:
        logger.info(f"Building mask: variant={mask_variant}  density={mask_density}")
        model.eval()
        mask = build_mask(
            variant         = mask_variant,
            model           = model,
            parameters      = parameters,
            forget_dl       = forget_dl,
            remain_dl       = remain_dl,
            descriptions    = descriptions,
            class_to_forget = class_to_forget,
            beta            = beta,
            device          = device,
            target_density  = mask_density,
            max_batches     = len(forget_dl),
            lambda_tradeoff = lambda_tradeoff,
            logger          = logger,
        )
        active = mask.sum().item()
        total  = mask.numel()
        logger.info(f"[Mask] active={active:,} / {total:,}  density={active/total:.4f}")
        name = (
            f"compvis-celeb_{class_to_forget}-MUKSB-{mask_variant}"
            f"-rho{int(mask_density*100)}pct"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{EXTRA}"
        )
    else:
        mask = None
        name = (
            f"compvis-celeb_{class_to_forget}-MUKSB"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{EXTRA}"
        )

    print(f"RUN_TAG={name}", flush=True)

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    losses, step = [], 0
    skipped_steps, total_steps = 0, 0
    cos_phi_history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs} started")
        remain_iter = iter(remain_dl)

        with tqdm(total=len(forget_dl)) as pbar:
            for forget_images, forget_labels in forget_dl:
                model.train()
                total_steps += 1

                try:
                    remain_images, remain_labels = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_images, remain_labels = next(remain_iter)

                remain_prompts = [descriptions[label] for label in remain_labels]
                forget_prompts = [descriptions[label] for label in forget_labels]

                # Anchor (pseudo) prompt: where the forgotten identity is steered.
                #   fixed → a neutral generic person  (ESD-style identity erasure)
                #   next  → another celebrity's caption (class-relabel, like MUKSB_cls)
                if anchor_mode == "next":
                    pseudo_prompts = [
                        descriptions[(int(label) + 1) % num_classes]
                        for label in forget_labels
                    ]
                else:  # "fixed"
                    pseudo_prompts = [anchor_prompt for _ in forget_labels]

                # ── retain loss ───────────────────────────────────────────────
                remain_batch = {
                    "jpg": remain_images.permute(0, 2, 3, 1),
                    "txt": remain_prompts,
                }
                loss_r = model.shared_step(remain_batch)[0]

                # ── forget loss ───────────────────────────────────────────────
                forget_batch = {"jpg": forget_images.permute(0, 2, 3, 1), "txt": forget_prompts}
                pseudo_batch = {"jpg": forget_images.permute(0, 2, 3, 1), "txt": pseudo_prompts}

                forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
                pseudo_input, pseudo_emb = model.get_input(pseudo_batch, model.first_stage_key)

                t     = torch.randint(0, model.num_timesteps,
                                      (forget_input.shape[0],), device=model.device).long()
                noise = torch.randn_like(forget_input, device=model.device)

                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                forget_out   = model.apply_model(forget_noisy, t, forget_emb)
                pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)
                pseudo_out   = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                loss_u = criteria(forget_out, pseudo_out) * beta

                # ── compute gradients ─────────────────────────────────────────
                grads_r = torch.autograd.grad(loss_r, parameters, retain_graph=True)
                grads_f = torch.autograd.grad(loss_u, parameters)

                gr_flat = _flatten_grads(parameters, grads_r)
                gf_flat = _flatten_grads(parameters, grads_f)
                del grads_r, grads_f

                # ── project onto masked subspace if mask provided ──────────────
                if mask is not None:
                    gr_input = gr_flat[mask]
                    gf_input = gf_flat[mask]
                else:
                    gr_input = gr_flat
                    gf_input = gf_flat

                # ── KS bargaining merge ───────────────────────────────────────
                lambda_ks, cos_phi, g_star, effective_scale = ks_step(gr_input, gf_input)
                cos_phi_history.append(cos_phi.item())

                # ── anti-parallel check ───────────────────────────────────────
                if torch.norm(g_star).item() < 1e-6:
                    skipped_steps += 1
                    logger.debug(
                        f"step={step}: anti-parallel gradients "
                        f"(cos_φ={cos_phi.item():.3f}), skipping update"
                    )
                    del gr_flat, gf_flat, gr_input, gf_input, g_star
                    pbar.update(1)
                    continue

                g_star_scaled = effective_scale * g_star
                del gr_input, gf_input

                # ── expand back to full parameter space ───────────────────────
                if mask is not None:
                    update_full = torch.zeros_like(gr_flat)
                    update_full[mask] = g_star_scaled
                else:
                    update_full = g_star_scaled

                del gr_flat, gf_flat, g_star, g_star_scaled

                # ── write update into model gradients ─────────────────────────
                optimizer.zero_grad()
                _unpack_to_grads(parameters, update_full)
                del update_full

                if with_l1:
                    current_alpha = alpha * (1 - epoch / epochs)
                    l1_loss  = current_alpha * l1_regularization(parameters)
                    l1_grads = torch.autograd.grad(l1_loss, parameters)
                    for p, lg in zip(parameters, l1_grads):
                        if p.grad is not None and lg is not None:
                            p.grad += lg.detach()

                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                combined_loss = loss_r + loss_u
                losses.append(combined_loss.item() / batch_size)
                step += 1

                if (step + 1) % 10 == 0:
                    avg_cos   = float(np.mean(cos_phi_history[-10:])) if cos_phi_history else 0.0
                    skip_rate = skipped_steps / max(total_steps, 1)
                    logger.info(
                        f"step={step}"
                        f"  λ_KS={lambda_ks.item():.4f}"
                        f"  cos_φ={cos_phi.item():.4f}"
                        f"  avg_cos_φ(10)={avg_cos:.4f}"
                        f"  eff_scale={effective_scale.item():.4e}"
                        f"  skip_rate={skip_rate:.3f}"
                        f"  loss_r={loss_r.item():.4f}"
                        f"  loss_u={loss_u.item():.4f}"
                    )
                    save_history(losses, name)

                pbar.set_description(f"Epoch {epoch+1}")
                pbar.set_postfix(
                    loss=combined_loss.item() / batch_size,
                    cos=f"{cos_phi.item():.2f}",
                    lam=f"{lambda_ks.item():.3f}",
                )
                pbar.update(1)

        epoch_time      = time.time() - epoch_start
        skip_rate_epoch = skipped_steps / max(total_steps, 1)
        logger.info(
            f"Epoch {epoch+1} done | {epoch_time:.2f}s ({epoch_time/60:.2f} min) | "
            f"cumulative skip_rate={skip_rate_epoch:.3f}"
        )

        model.eval()
        if (epoch + 1) % 5 == 0 and epoch != epochs - 1:
            save_model(
                model, name, epoch + 1,
                save_compvis=False, save_diffusers=True,
                compvis_config_file=config_path,
                diffusers_config_file=diffusers_config_path,
            )
        torch.cuda.empty_cache()
        gc.collect()

    # ── save final model ──────────────────────────────────────────────────────
    total_time      = time.time() - total_start
    final_skip_rate = skipped_steps / max(total_steps, 1)
    logger.info("======== MUKSB CELEBRITY TRAINING FINISHED ========")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    logger.info(f"Anti-parallel skips: {skipped_steps}/{total_steps} ({final_skip_rate*100:.1f}%)")
    if cos_phi_history:
        logger.info(
            f"cos_φ stats: mean={np.mean(cos_phi_history):.4f}  "
            f"min={np.min(cos_phi_history):.4f}  max={np.max(cos_phi_history):.4f}"
        )

    model.eval()
    save_model(
        model, name, epochs,
        save_compvis=False, save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )
    save_history(losses, name)
    logger.info(f"Model and loss history saved under: models/{name}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _mask_variant_type(value):
        if value is None or value == "None":
            return None
        return str(value)

    parser = argparse.ArgumentParser(
        description=(
            "MUKSB: celebrity identity unlearning for Stable Diffusion "
            "(class-wise, KS bargaining)"
        )
    )
    parser.add_argument("--class_to_forget", type=str, default="0",
                        help="Celebrity index to erase (0..N-1, per prompts/celebrity.csv)")
    parser.add_argument("--train_method",    type=str, default="full",
                        choices=["full", "noxattn", "xattn", "selfattn",
                                 "notime", "xlayer", "selflayer"])
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--ckpt_path",   type=str,
                        default="models/ldm/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--mask_variant", type=_mask_variant_type, default="None",
                        choices=list(MASK_VARIANT_CHOICES) + [None],
                        help="Parameter selection strategy for sparse update "
                             "(random / forget_fisher / salun / dual_fisher). "
                             "Omit for dense update.")
    parser.add_argument("--mask_density",    type=float, default=0.5)
    parser.add_argument("--lambda_tradeoff", type=float, default=1.0,
                        help="λ in S_diff = F̂_f − λ·F̂_r  (dual_fisher only)")
    parser.add_argument("--config_path", type=str,
                        default="configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--diffusers_config_path", type=str,
                        default="diffusers_unet_config.json")
    parser.add_argument("--device",      type=str,   default="0")
    parser.add_argument("--image_size",  type=int,   default=512)
    parser.add_argument("--ddim_steps",  type=int,   default=50)
    parser.add_argument("--with_l1",     action="store_true", default=False)
    parser.add_argument("--beta",        type=float, default=1.0)
    parser.add_argument("--alpha",       type=float, default=1e-4)
    parser.add_argument("--anchor_mode", type=str,   default="next",
                        choices=["fixed", "next"],
                        help="next → pseudo-label: steer the forgotten identity to "
                             "the next celebrity's caption descriptions[(c+1)%%N] "
                             "(matches MUKSB_cls.py); fixed → steer to --anchor_prompt.")
    parser.add_argument("--anchor_prompt", type=str, default="a photo of a person",
                        help="Neutral target caption for the forgotten identity "
                             "(used when --anchor_mode fixed).")

    args = parser.parse_args()

    logger, log_file = setup_logger(name=f"MUKSB_celeb{args.class_to_forget}_{EXTRA}")
    logger.info("======== MUKSB CELEBRITY STARTED ========")
    logger.info(f"Log: {log_file}")
    logger.info(f"Args: {vars(args)}")

    setup_seed(42)

    MUKSB(
        class_to_forget       = int(args.class_to_forget),
        train_method          = args.train_method,
        batch_size            = args.batch_size,
        epochs                = args.epochs,
        lr                    = args.lr,
        config_path           = args.config_path,
        ckpt_path             = args.ckpt_path,
        mask_variant          = args.mask_variant,
        mask_density          = args.mask_density,
        lambda_tradeoff       = args.lambda_tradeoff,
        diffusers_config_path = args.diffusers_config_path,
        device                = f"cuda:{args.device}",
        image_size            = args.image_size,
        ddim_steps            = args.ddim_steps,
        with_l1               = args.with_l1,
        beta                  = args.beta,
        alpha                 = args.alpha,
        anchor_mode           = args.anchor_mode,
        anchor_prompt         = args.anchor_prompt,
        logger                = logger,
    )
