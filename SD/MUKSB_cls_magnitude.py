"""
SD/MUKSB_cls.py
================
MUKSB — Machine Unlearning via Kalai-Smorodinsky Bargaining
Stable Diffusion component — class-concept erasure on Imagenette-10

Gradient Merge — Kalai-Smorodinsky (KS) Bargaining
----------------------------------------------------
Implements the KS closed-form solution with five targeted improvements
over the vanilla KS bisector:

  Fix 2 — Harmonic-scale step size (replaces lambda_ks * g_star)
      s_KS = 2·||g_r||·||g_f|| / (||g_r|| + ||g_f||)
      Recovers a curvature-aware scale derivable from the utopia gains.
      Replaces the unjustified lambda_ks = cos(φ/2) scaling used previously.
      This fix is always active (no flag needed).

  Fix 3 — Asymmetric priority (gamma)
      gamma=0.5  →  symmetric KS (equal proportional gain, default)
      gamma>0.5  →  retain-favoured bisector  (recommended for SD: 0.7)
      gamma<0.5  →  forget-favoured bisector
      Mathematically: g_sum = gamma·w_r·ĝ_r + (1-gamma)·w_f·ĝ_f

  Fix 4 — Asymmetric loss scales handled via gamma (see Fix 3).
      For Imagenette-10 (1 forget / 9 retain classes) use gamma ≈ 0.7.

Backward compatibility
----------------------
All new arguments default to values that recover the original pure-KS
behaviour:
    --gamma 0.5

The only non-optional change is Fix 2 (harmonic scale), which corrects
a theoretical error in the original code (lambda_ks * g_star had no
derivation grounding).

Common proportional gain (diagnostic only, no longer used for scaling):
    λ_KS = cos(φ/2)  ∈ [0, 1]

Edge case (φ → π, anti-parallel gradients):
    ||ĝ_r + ĝ_f|| → 0  →  g* = 0  (no Pareto-improving update; step skipped)

Usage
-----
  # Pure KS — identical geometry to original (Fix 2 always applies):
  python SD/MUKSB_cls.py --class_to_forget 0 --epochs 5 --device 0

  # Retain-prioritised KS (recommended for SD, 1 forget / 9 retain):
  python SD/MUKSB_cls.py --class_to_forget 0 --epochs 5 --device 0 --gamma 0.7

  # With SalUn mask:
  python SD/MUKSB_cls.py --class_to_forget 0 --epochs 5 --device 0 \\
      --mask_variant salun --mask_density 0.1 --gamma 0.7
"""

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

from logger.logger import setup_logger
from train_scripts.convertModels import savemodelDiffusers
from train_scripts.dataset import (
    setup_forget_remain_data,
    setup_model,
)
from mask_variants import build_mask, MASK_VARIANT_CHOICES

FORGET_TO_PSEUDO_PROMPT = {
    0: "a photo of a cat",         # tench        → cat
    1: "a photo of a sports car",        # springer     → car
    2: "a photo of a pizza",             # cassette     → pizza
    3: "a photo of a daisy",             # chain saw    → flower
    4: "a photo of a banana",            # church       → banana
    5: "a photo of a grand piano",       # french horn  → piano
    6: "a photo of a volcano",           # garbage truck→ volcano
    7: "a photo of a hot air balloon",   # gas pump     → balloon
    8: "a photo of a mushroom",          # golf ball    → mushroom
    9: "a photo of a coral reef",        # parachute    → coral reef
}

EXTRA = ""


# ─────────────────────────────────────────────────────────────────────────────
# KS bargaining core — improved
# ─────────────────────────────────────────────────────────────────────────────

def ks_step(
    gr_flat: torch.Tensor,
    gf_flat: torch.Tensor,
    gamma: float = 0.5,
    eps: float = 1e-8,
):
    """
    Kalai-Smorodinsky bargaining solution for gradient merging.

    Core KS derivation (unchanged from theory)
    -------------------------------------------
    Both players have utility  U_i(g̃) = -L_i(θ) + gᵢᵀg̃
    Disagreement point:        d_i     = -L_i(θ)
    Gain above disagreement:   U_i(g̃) - d_i = gᵢᵀg̃
    Utopia gain:               U_i^max - d_i = ε·||gᵢ||

    KS equal proportional-gain condition:
        ĝᵣᵀg̃* = ĝ_fᵀg̃*
    Solution: g̃* ∝ gamma·ĝᵣ + (1-gamma)·ĝ_f

    gamma (direction)
    -----------------
    g_sum  = gamma·ĝᵣ + (1-gamma)·ĝ_f          (weighted angle bisector)
    g_star = g_sum / ||g_sum||                   (unit vector)

    gamma=0.5 → symmetric KS, equal directional weight (default)
    gamma>0.5 → bisector tilts toward retain direction
    gamma=0.7 → recommended for SD (1 forget / 9 retain classes)

    Parameters
    ----------
    gr_flat : Tensor, shape (D,)
        Flattened retain gradient.
    gf_flat : Tensor, shape (D,)
        Flattened forget gradient.
    gamma : float
        Retain directional priority in [0, 1]. 0.5 = symmetric KS.
    eps : float
        Numerical stability floor.

    Returns
    -------
    lambda_ks : Tensor (scalar)
        Common proportional gain cos(φ/2). Diagnostic only.
    cos_phi : Tensor (scalar)
        Cosine of angle between the two gradients.
    g_star : Tensor, shape (D,)
        KS-merged update direction (unit vector; zero if anti-parallel).
    effective_scale : Tensor (scalar)
        Step scale = harmonic mean of gradient norms.
        Multiply g_star by this before passing to the optimiser.
    """
    norm_gr = torch.clamp(torch.norm(gr_flat), min=1e-6)
    norm_gf = torch.clamp(torch.norm(gf_flat), min=1e-6)

    # ── diagnostic: cosine angle between raw gradients ────────────────────────
    cos_phi = torch.clamp(
        torch.dot(gr_flat, gf_flat) / (norm_gr * norm_gf),
        -1.0 + eps, 1.0 - eps,
    )

    # ── unit vectors ──────────────────────────────────────────────────────────
    g_hat_r = gr_flat / norm_gr
    g_hat_f = gf_flat / norm_gf

    # ── DIRECTION: gamma-weighted bisector, tau-independent ───────────────────
    # gamma controls purely which way g_star points.
    # No magnitude terms here — tau has zero influence on direction.
    g_sum    = gamma * g_hat_r + (1.0 - gamma) * g_hat_f
    norm_sum = torch.norm(g_sum)

    # ── anti-parallel / degenerate edge case ──────────────────────────────────
    if norm_sum < 1e-6:
        zero = torch.zeros_like(gr_flat)
        return (
            torch.tensor(0.0, device=gr_flat.device),
            cos_phi,
            zero,
            torch.tensor(0.0, device=gr_flat.device),
        )

    g_star = g_sum / norm_sum

    # ── SCALE: harmonic mean of gradient norms ────────────────────────────────
    # harmonic mean: 2·a·b/(a+b) — conservative, dominated by smaller norm
    effective_scale = 2.0 * norm_gr * norm_gf / (norm_gr + norm_gf)

    # ── diagnostic: common proportional gain ──────────────────────────────────
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
# Utilities (shared with MUNBa_cls.py for fair comparison)
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
    v = np.convolve(losses, np.ones(min(3, len(losses))) / min(3, len(losses)), mode="valid")
    plt.figure()
    plt.plot(v, label="loss")
    plt.legend(loc="upper left")
    plt.title("MUKSB training loss")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.savefig(f"{folder_path}/loss.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main unlearning function
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
    # ── KS improvement parameters ─────────────────────────────────────────────
    gamma,              # Fix 3: retain priority (0.5 = symmetric KS)
    logger,
):
    """
    MUKSB unlearning for Stable Diffusion — class erasure.

    Replaces Nash bargaining (MUNBa) with improved KS bargaining.
    Data setup, parameter selection, and loss formulation are identical
    to MUNBa_cls.py for a fair comparison.

    KS improvements controlled by:
        gamma — asymmetric priority (Fix 3 / Fix 4)

    Fix 2 (harmonic scale) is always active and corrects a bug in the
    original code where lambda_ks * g_star was used without justification.

    Default value (gamma=0.5) reproduces the original pure-KS geometry
    exactly, with Fix 2 applied.
    """
    total_start = time.time()
    logger.info("======== MUKSB (KS Bargaining — Improved) TRAINING STARTED ========")
    logger.info(f"class_to_forget={class_to_forget}  train_method={train_method}")
    logger.info(f"KS config: gamma={gamma} (retain priority)")
    logger.info(
        "Pure KS mode: " +
        ("YES" if gamma == 0.5 else "NO — improvements active")
    )

    # ── model + data ─────────────────────────────────────────────────────────
    model    = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()

    forget_dl, remain_dl, descriptions = setup_forget_remain_data(
        class_to_forget, batch_size, image_size
    )
    num_forget = len(forget_dl.dataset)
    logger.info(f"Forget set: {num_forget} samples | class: {class_to_forget}")
    

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
            f"compvis-cls_{class_to_forget}-MUKSB-{mask_variant}"
            f"-rho{int(mask_density*100)}pct"
            f"-g{gamma}"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{EXTRA}"
        )
    else:
        mask = None
        name = (
            f"compvis-cls_{class_to_forget}-MUKSB"
            f"-g{gamma}"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{EXTRA}"
        )

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    losses, step = [], 0
    epoch_times  = []

    # Conflict tracking: how often gradients are near anti-parallel
    skipped_steps   = 0
    total_steps     = 0
    cos_phi_history = []
    # pseudo_prompt_text = FORGET_TO_PSEUDO_PROMPT[int(class_to_forget)]
    # logger.info(f"Pseudo class (OOD): '{pseudo_prompt_text}'  ←  forget class: {class_to_forget}")

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
                
                pseudo_prompts = [
                    descriptions[(int(class_to_forget) + 1) % 10]
                    for _ in forget_labels
                ]

                # pseudo_prompts = [pseudo_prompt_text for _ in forget_labels]

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

                # ── KS bargaining merge (improved) ────────────────────────────
                lambda_ks, cos_phi, g_star, effective_scale = ks_step(
                    gr_input, gf_input,
                    gamma=gamma,
                )

                cos_phi_history.append(cos_phi.item())

                # ── anti-parallel check ───────────────────────────────────────
                # g_star is zero only on true anti-parallel conflict.
                # Check norm directly — effective_scale can be non-zero even
                # when g_star is zero due to floating point, so don't rely on it.
                if torch.norm(g_star).item() < 1e-6:
                    skipped_steps += 1
                    logger.debug(
                        f"step={step}: anti-parallel gradients "
                        f"(cos_φ={cos_phi.item():.3f}), skipping update"
                    )
                    del gr_flat, gf_flat, gr_input, gf_input, g_star
                    pbar.update(1)
                    continue

                # ── scale: harmonic mean ──────────────────────────────────────
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

                # ── optional L1 regularisation on top of KS direction ─────────
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
                    avg_cos = float(np.mean(cos_phi_history[-10:])) if cos_phi_history else 0.0
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

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        skip_rate_epoch = skipped_steps / max(total_steps, 1)
        logger.info(
            f"Epoch {epoch+1} done | "
            f"{epoch_time:.2f}s ({epoch_time/60:.2f} min) | "
            f"cumulative skip_rate={skip_rate_epoch:.3f}"
        )

        model.eval()
        if (epoch + 1) % 1 == 0 and epoch != epochs - 1:
            save_model(
                model, name, epoch + 1,
                save_compvis=False, save_diffusers=True,
                compvis_config_file=config_path,
                diffusers_config_file=diffusers_config_path,
            )
        torch.cuda.empty_cache()
        gc.collect()

    # ── save final model ──────────────────────────────────────────────────────
    total_time = time.time() - total_start
    final_skip_rate = skipped_steps / max(total_steps, 1)
    logger.info("======== MUKSB TRAINING FINISHED ========")
    logger.info(
        f"Total time: {total_time:.2f}s "
        f"({total_time/60:.2f} min | {total_time/3600:.2f} hrs)"
    )
    logger.info(
        f"Anti-parallel skips: {skipped_steps}/{total_steps} "
        f"({final_skip_rate*100:.1f}%) — "
        f"high skip_rate indicates severe gradient conflict; consider increasing gamma"
    )
    if cos_phi_history:
        logger.info(
            f"cos_φ stats: mean={np.mean(cos_phi_history):.4f}  "
            f"min={np.min(cos_phi_history):.4f}  "
            f"max={np.max(cos_phi_history):.4f}"
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
            "MUKSB: Machine Unlearning via Kalai-Smorodinsky Bargaining "
            "(Stable Diffusion — improved)"
        )
    )

    # ── original arguments (unchanged) ───────────────────────────────────────
    parser.add_argument("--class_to_forget", type=str,   default="4",
                        help="Imagenette class index to erase (0–9)")
    parser.add_argument("--train_method",    type=str,   default="full",
                        choices=["full", "noxattn", "xattn", "selfattn",
                                 "notime", "xlayer", "selflayer"])
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--ckpt_path",   type=str,
                        default="models/ldm/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--mask_variant", type=_mask_variant_type, default="salun",
                        choices=list(MASK_VARIANT_CHOICES) + [None],
                        help=(
                            "Parameter selection strategy for sparse update.\n"
                            "  random        — uniform random top-k%%\n"
                            "  forget_fisher — forget Fisher only (F_f)\n"
                            "  salun         — gradient magnitude |∇L_f| (SalUn-style)\n"
                            "  dual_fisher   — dual Fisher score"
                        ))
    parser.add_argument("--mask_density",    type=float, default=0.5)
    parser.add_argument("--lambda_tradeoff", type=float, default=1.0,
                        help="λ in S_diff = F̂_f − λ·F̂_r  (dual_fisher only)")
    parser.add_argument("--config_path", type=str,
                        default="configs/stable-diffusion/v1-inference_nash.yaml")
    parser.add_argument("--diffusers_config_path", type=str,
                        default="diffusers_unet_config.json")
    parser.add_argument("--device",      type=str,   default="3")
    parser.add_argument("--image_size",  type=int,   default=256)
    parser.add_argument("--ddim_steps",  type=int,   default=50)
    parser.add_argument("--with_l1",     action="store_true", default=False)
    parser.add_argument("--beta",        type=float, default=1.0)
    parser.add_argument("--alpha",       type=float, default=1e-4)

    # ── new KS improvement arguments ─────────────────────────────────────────
    parser.add_argument(
        "--gamma", type=float, default=0.5,
        help=(
            "[Fix 3/5] Retain priority weight in [0, 1]. "
            "0.5 = symmetric KS (default, original behaviour). "
            "0.7 recommended for SD with 1 forget / 9 retain classes. "
            ">0.5 tilts the bisector toward the retain gradient direction."
        ),
    )
    args = parser.parse_args()

    # ── validate new arguments ────────────────────────────────────────────────
    assert 0.0 <= args.gamma <= 1.0, "--gamma must be in [0, 1]"

    logger, log_file = setup_logger(name=f"MUKSB_cls{args.class_to_forget}_{EXTRA}")
    logger.info("======== MUKSB STARTED ========")
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
        gamma                 = args.gamma,
        logger                = logger,
    )