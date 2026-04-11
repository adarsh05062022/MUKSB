"""
SD/MUKSB_nsfw.py — MUKSB
Kalai-Smorodinsky bargaining unlearning for NSFW concept removal in Stable Diffusion.

The KS merge rule:
    g* ∝ ĝ_r + ĝ_f   (unit-vector sum, satisfies Monotonicity axiom)
    λ_KS = ĝ_r · g̃*  (common proportional gain)

Steps are skipped when the retain and forget gradients are anti-parallel
(||ĝ_r + ĝ_f|| < 1e-6), i.e. no beneficial compromise exists.

If --mask_variant is given, a sparse parameter mask is built at runtime
before training using build_mask_nsfw() from mask_variants.py.  The KS
update is then restricted to the masked subspace (parameters outside the
mask receive a zero update).  If --mask_variant is omitted, all selected
parameters are updated.

Usage (run from MUKSB/SD/)
-----
  # No mask — all selected parameters updated:
  python /storage/s25017/MUKSB/SD/MUKSB_nsfw.py \\
      --train_method full --epochs 5 --device 0

  # With mask — built at runtime:
  python /storage/s25017/MUKSB/SD/MUKSB_nsfw.py \\
      --train_method full --epochs 5 --device 0 \\
      --mask_variant dual_fisher --mask_density 0.1
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

from train_scripts.convertModels import savemodelDiffusers
from train_scripts.dataset import setup_model, setup_nsfw_data
from logger.logger import setup_logger
from mask_variants import build_mask_nsfw, MASK_VARIANT_CHOICES


# ─────────────────────────────────────────────────────────────────────────────
# KS bargaining core
# ─────────────────────────────────────────────────────────────────────────────

def ks_step(gr_flat: torch.Tensor, gf_flat: torch.Tensor, eps: float = 1e-8):
    """
    Kalai-Smorodinsky gradient merge.

    Returns
    -------
    lambda_ks : scalar gain  (0.0 signals anti-parallel → skip step)
    cos_phi   : cosine between gr and gf
    g_star    : merged direction (unit vector), zero tensor if anti-parallel
    """
    norm_r = torch.clamp(torch.norm(gr_flat), min=1e-6)
    norm_f = torch.clamp(torch.norm(gf_flat), min=1e-6)
    cos_phi = torch.clamp(torch.dot(gr_flat, gf_flat) / (norm_r * norm_f),
                          -1 + eps, 1 - eps)
    g_hat_r = gr_flat / norm_r
    g_hat_f = gf_flat / norm_f
    g_sum   = g_hat_r + g_hat_f
    norm_sum = torch.norm(g_sum)
    if norm_sum < 1e-6:                          # anti-parallel → skip
        return torch.tensor(0.0, device=gr_flat.device), cos_phi, torch.zeros_like(gr_flat)
    g_star    = g_sum / norm_sum
    lambda_ks = torch.dot(g_hat_r, g_star)
    return lambda_ks, cos_phi, g_star


def _flatten_grads(params, grads):
    return torch.cat([
        g.detach().view(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
        for p, g in zip(params, grads)
    ])


def _unpack_to_grads(params, flat_vec: torch.Tensor):
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat_vec[offset: offset + n].view(p.shape).clone()
        offset += n


# ─────────────────────────────────────────────────────────────────────────────
# L1 regularisation helper
# ─────────────────────────────────────────────────────────────────────────────

def l1_regularization(parameters):
    return torch.linalg.norm(
        torch.cat([p.view(-1) for p in parameters]), ord=1
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    model, name, num,
    compvis_config_file=None, diffusers_config_file=None,
    device="cpu", save_compvis=False, save_diffusers=True,
    logger=None
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


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(v) + "\n" for v in losses])
    v = np.convolve(losses, np.ones(3) / 3, mode="valid")
    plt.figure()
    plt.plot(v, label=f"{word_print}_loss")
    plt.legend(loc="upper left")
    plt.title("Training loss (moving avg, n=3)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"{folder_path}/loss.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main MUKSB NSFW training loop
# ─────────────────────────────────────────────────────────────────────────────

def MUKSB(
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
    alpha,
    beta,
    forget_path,
    remain_path,
    logger,
):
    total_start = time.time()
    logger.info("MUKSB NSFW training started")

    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()

    forget_dl, remain_dl = setup_nsfw_data(
        batch_size,
        forget_path=forget_path,
        remain_path=remain_path,
        image_size=image_size,
    )
    num_forget = len(forget_dl.dataset)
    logger.info(f"Forget samples: {num_forget} | Remain samples: {len(remain_dl.dataset)}")

    # ── parameter selection ────────────────────────────────────────────────
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        if train_method == "noxattn":
            if not (name.startswith("out.") or "attn2" in name or "time_embed" in name):
                parameters.append(param)
        elif train_method == "selfattn":
            if "attn1" in name:
                parameters.append(param)
        elif train_method == "xattn":
            if "attn2" in name:
                parameters.append(param)
        elif train_method == "full":
            parameters.append(param)
        elif train_method == "notime":
            if not (name.startswith("out.") or "time_embed" in name):
                parameters.append(param)
        elif train_method == "xlayer":
            if "attn2" in name and ("output_blocks.6." in name or "output_blocks.8." in name):
                parameters.append(param)
        elif train_method == "selflayer":
            if "attn1" in name and ("input_blocks.4." in name or "input_blocks.7." in name):
                parameters.append(param)

    # ── build importance mask (if requested) ─────────────────────────────────
    if mask_variant is not None:
        logger.info(f"Building mask: variant={mask_variant}  density={mask_density}")
        model.eval()
        mask = build_mask_nsfw(
            variant         = mask_variant,
            model           = model,
            parameters      = parameters,
            forget_dl       = forget_dl,
            remain_dl       = remain_dl,
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
        run_tag = (
            f"compvis-nsfw-MUKSB-{mask_variant}"
            f"-rho{int(mask_density*100)}pct"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}"
        )
    else:
        mask = None
        run_tag = f"compvis-nsfw-MUKSB-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}"

    model.train()
    optimizer = torch.optim.Adam(parameters, lr=lr)
    losses    = []
    step      = 0

    word_wear = "a photo of a person wearing clothes"

    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        remain_iter = iter(remain_dl)

        with tqdm(total=len(forget_dl), desc=f"Epoch {epoch + 1}") as pbar:
            for forget_batch in forget_dl:
                model.train()
                optimizer.zero_grad()

                # ── retain batch ────────────────────────────────────────────
                try:
                    remain_batch = next(remain_iter)
                except StopIteration:
                    remain_iter  = iter(remain_dl)
                    remain_batch = next(remain_iter)

                loss_r = model.shared_step(remain_batch)[0]

                # ── forget / pseudo batch ───────────────────────────────────
                forget_input, forget_emb = model.get_input(
                    forget_batch, model.first_stage_key
                )
                pseudo_prompts = [word_wear] * forget_batch["jpg"].size(0)
                pseudo_batch   = {"jpg": forget_batch["jpg"], "txt": pseudo_prompts}
                pseudo_input, pseudo_emb = model.get_input(
                    pseudo_batch, model.first_stage_key
                )

                t     = torch.randint(0, model.num_timesteps,
                                      (forget_input.shape[0],), device=device).long()
                noise = torch.randn_like(forget_input, device=device)

                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                forget_out   = model.apply_model(forget_noisy, t, forget_emb)
                pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)
                pseudo_out   = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                loss_u = criteria(forget_out, pseudo_out) * beta

                # ── KS gradient merge ───────────────────────────────────────
                optimizer.zero_grad()
                grads_r = torch.autograd.grad(loss_r, parameters, retain_graph=True,
                                               allow_unused=True)
                grads_f = torch.autograd.grad(loss_u, parameters, retain_graph=True,
                                               allow_unused=True)

                gr_flat = _flatten_grads(parameters, grads_r)
                gf_flat = _flatten_grads(parameters, grads_f)

                # Project onto masked subspace if mask provided; otherwise use all params
                if mask is not None:
                    gr_masked = gr_flat[mask]
                    gf_masked = gf_flat[mask]
                else:
                    gr_masked = gr_flat
                    gf_masked = gf_flat

                lambda_ks, cos_phi, g_star = ks_step(gr_masked, gf_masked)

                # later try to find the gradient update with all the parameters and only updating
                #  the masked parameters and see if it is better than only finding the gradient update
                #  with the masked parameters and updating only those parameters

                

                if lambda_ks.item() == 0.0:          # anti-parallel → skip
                    del gr_flat, gf_flat, gr_masked, gf_masked, grads_r, grads_f, g_star
                    torch.cuda.empty_cache(); gc.collect()
                    pbar.update(1)
                    continue

                del gr_masked, gf_masked, grads_r, grads_f

                # Expand g_star back to the full parameter space (zeros outside mask)
                if mask is not None:
                    update_full = torch.zeros_like(gr_flat)
                    update_full[mask] = g_star
                else:
                    update_full = g_star

                del gr_flat, gf_flat, g_star

                # Write update into model gradients
                optimizer.zero_grad()
                _unpack_to_grads(parameters, update_full)
                del update_full

                if with_l1:
                    l1_loss = alpha * l1_regularization(parameters)
                    l1_grads = torch.autograd.grad(l1_loss, parameters)
                    for p, lg in zip(parameters, l1_grads):
                        if p.grad is not None and lg is not None:
                            p.grad += lg.detach()

                nn.utils.clip_grad_norm_(parameters, 1.0)
                losses.append((loss_r + loss_u).item() / batch_size)
                optimizer.step()
                step += 1
                torch.cuda.empty_cache()
                gc.collect()

                if step % 10 == 0:
                    logger.info(
                        f"step {step} | loss_r={loss_r:.4f} | loss_u={loss_u:.4f} "
                        f"| λ_KS={lambda_ks.item():.4f} | cos_φ={cos_phi.item():.4f}"
                    )
                    save_history(losses, run_tag, "nsfw")

                pbar.set_postfix(loss_r=f"{loss_r:.4f}", lam=f"{lambda_ks.item():.3f}")
                pbar.update(1)

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1} done | {epoch_time:.1f}s ({epoch_time/60:.2f} min)")

        model.eval()
        if (epoch + 1) % 1 == 0 and epoch != epochs - 1:
            save_model(model, run_tag, epoch+1,
                       compvis_config_file=config_path,
                       diffusers_config_file=diffusers_config_path,
                       save_compvis=False, save_diffusers=True, logger=logger)

    total_time = time.time() - total_start
    logger.info(
        f"======== TRAINING FINISHED ========"
        f"\nTotal: {total_time:.1f}s ({total_time/60:.2f} min)"
    )
    model.eval()
    save_model(model, run_tag, epochs,
               compvis_config_file=config_path,
               diffusers_config_file=diffusers_config_path,
               save_compvis=False, save_diffusers=True, logger=logger)
    save_history(losses, run_tag, "nsfw")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger, log_file = setup_logger(name="MUKSB_nsfw")

    parser = argparse.ArgumentParser(
        description="MUKSB: KS-Bargaining NSFW concept unlearning for Stable Diffusion"
    )
    parser.add_argument("--train_method",         type=str,   default="full")
    parser.add_argument("--batch_size",            type=int,   default=8)
    parser.add_argument("--epochs",                type=int,   default=5)
    parser.add_argument("--lr",                    type=float, default=1e-5)
    parser.add_argument("--ckpt_path",             type=str,
                        default="models/ldm/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--mask_variant",          type=str,   default="None",
                        choices=list(MASK_VARIANT_CHOICES) + [None],
                        help=(
                            "Parameter selection strategy for sparse update. "
                            "If omitted, all selected parameters are updated.\n"
                            "  random        — uniform random top-k%%\n"
                            "  forget_fisher — forget Fisher only (F_f)\n"
                            "  salun         — gradient magnitude |∇L_f| (SalUn-style)\n"
                            "  dual_fisher   — dual Fisher score (proposed)"
                        ))
    parser.add_argument("--mask_density",          type=float, default=0.1,
                        help="Fraction ρ of parameters to update when using a mask (default: 0.1)")
    parser.add_argument("--lambda_tradeoff",       type=float, default=1.0,
                        help="λ in S_diff = F̂_f − λ·F̂_r  (dual_fisher only)")
    parser.add_argument("--config_path",           type=str,
                        default="configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--diffusers_config_path", type=str,
                        default="diffusers_unet_config.json")
    parser.add_argument("--device",                type=str,   default="2")
    parser.add_argument("--image_size",            type=int,   default=256)
    parser.add_argument("--ddim_steps",            type=int,   default=50)
    parser.add_argument("--with_l1",               action="store_true", default=False)
    parser.add_argument("--alpha",                 type=float, default=1e-4,
                        help="L1 regularisation coefficient")
    parser.add_argument("--beta",                  type=float, default=100.0,
                        help="Scale factor for forget loss")
    parser.add_argument("--forget_path",           type=str,
                        default="/storage/s25017/Datasets/NSFW_removal/NSFW/nude")
    parser.add_argument("--remain_path",           type=str,
                        default="/storage/s25017/Datasets/NSFW_removal/with_dress")
    args = parser.parse_args()

    logger.info("======== MUKSB NSFW TRAINING STARTED ========")
    logger.info(f"Log file : {log_file}")
    logger.info(f"Args     : {vars(args)}")

    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    MUKSB(
        train_method         = args.train_method,
        batch_size           = args.batch_size,
        epochs               = args.epochs,
        lr                   = args.lr,
        config_path          = args.config_path,
        ckpt_path            = args.ckpt_path,
        mask_variant         = args.mask_variant,
        mask_density         = args.mask_density,
        lambda_tradeoff      = args.lambda_tradeoff,
        diffusers_config_path= args.diffusers_config_path,
        device               = f"cuda:{int(args.device)}",
        image_size           = args.image_size,
        ddim_steps           = args.ddim_steps,
        with_l1              = args.with_l1,
        alpha                = args.alpha,
        beta                 = args.beta,
        forget_path          = args.forget_path,
        remain_path          = args.remain_path,
        logger               = logger,
    )
