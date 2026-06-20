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


EXTRA = "rawsum_weighted"


# ─────────────────────────────────────────────────────────────────────────────
# Direction ablation core  —  MUKSB_RawSum (weighted, UNNORMALISED)
#
# Instead of the KS bisector of UNIT gradients + harmonic-mean scaling, this
# variant merges the raw (un-normalised) retain/forget gradients with a fixed
# forget weight λ:
#
#     g̃ = (1 − λ) g_r + λ g_f          λ = forget_weight ∈ {0.25, 0.5, 0.75}
#
# No unit normalisation, no harmonic-mean rescaling.
# ─────────────────────────────────────────────────────────────────────────────

def rawsum_step(
    gr_flat: torch.Tensor,
    gf_flat: torch.Tensor,
    forget_weight: float,
    eps: float = 1e-8,
):
    # weighted arithmetic sum of the raw (un-normalised) gradients
    update = (1.0 - forget_weight) * gr_flat + forget_weight * gf_flat

    # diagnostic: cosine between the two raw gradients
    norm_gr = torch.clamp(torch.norm(gr_flat), min=1e-6)
    norm_gf = torch.clamp(torch.norm(gf_flat), min=1e-6)
    cos_phi = torch.clamp(
        torch.dot(gr_flat, gf_flat) / (norm_gr * norm_gf),
        -1.0 + eps, 1.0 - eps,
    )
    return update, cos_phi


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
# Main MUKSB_RawSum NSFW training loop
# ─────────────────────────────────────────────────────────────────────────────

def MUKSB_RawSum(
    train_method,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_variant,
    mask_density,
    lambda_tradeoff,
    forget_weight,
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
    logger.info("======== MUKSB_RawSum NSFW (Weighted Raw-Sum Direction) TRAINING STARTED ========")
    logger.info(f"EXTRA = {EXTRA}")
    logger.info(f"forget_weight (λ) = {forget_weight}  →  g̃ = (1-λ)·g_r + λ·g_f")

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

    logger.info(f"Trainable params: {sum(p.numel() for p in parameters):,}")

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
            f"compvis-nsfw-MUKSB_RawSum-lam{int(forget_weight*100)}-{mask_variant}"
            f"-rho{int(mask_density*100)}pct"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{EXTRA}"
        )
    else:
        mask = None
        run_tag = (
            f"compvis-nsfw-MUKSB_RawSum-lam{int(forget_weight*100)}"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{EXTRA}"
        )

    model.train()
    optimizer = torch.optim.Adam(parameters, lr=lr)
    losses    = []
    step      = 0

    word_wear = "a photo of a person wearing clothes"

    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch + 1}/{epochs} started")

        remain_iter = iter(remain_dl)

        with tqdm(total=len(forget_dl), desc=f"Epoch {epoch + 1}") as pbar:
            for forget_batch in forget_dl:
                model.train()

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

                # ── weighted raw-sum gradient merge ─────────────────────────
                grads_r = torch.autograd.grad(loss_r, parameters, retain_graph=True,
                                               allow_unused=True)
                grads_f = torch.autograd.grad(loss_u, parameters,
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

                update_masked, cos_phi = rawsum_step(
                    gr_masked, gf_masked, forget_weight
                )
                del gr_masked, gf_masked, grads_r, grads_f

                # Expand back to the full parameter space (zeros outside mask)
                if mask is not None:
                    update_full = torch.zeros_like(gr_flat)
                    update_full[mask] = update_masked
                else:
                    update_full = update_masked

                del gr_flat, gf_flat, update_masked

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
                optimizer.step()

                losses.append((loss_r + loss_u).item() / batch_size)
                step += 1

                if step % 10 == 0:
                    logger.info(
                        f"step={step}"
                        f"  λ_forget={forget_weight:.2f}"
                        f"  cos_φ={cos_phi.item():.4f}"
                        f"  loss_r={loss_r.item():.4f}"
                        f"  loss_u={loss_u.item():.4f}"
                    )
                    save_history(losses, run_tag, "nsfw")

                pbar.set_postfix(loss_r=f"{loss_r:.4f}", cos=f"{cos_phi.item():.3f}")
                pbar.update(1)

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1} done | {epoch_time:.1f}s ({epoch_time/60:.2f} min)")

        model.eval()
        if (epoch + 1) % 1 == 0 and epoch != epochs - 1:
            save_model(model, run_tag, epoch+1,
                       compvis_config_file=config_path,
                       diffusers_config_file=diffusers_config_path,
                       save_compvis=False, save_diffusers=True, logger=logger)
        torch.cuda.empty_cache(); gc.collect()

    total_time = time.time() - total_start
    logger.info("======== MUKSB_RawSum NSFW TRAINING FINISHED ========")
    logger.info(
        f"Total: {total_time:.1f}s ({total_time/60:.2f} min | {total_time/3600:.2f} hrs)"
    )
    model.eval()
    save_model(model, run_tag, epochs,
               compvis_config_file=config_path,
               diffusers_config_file=diffusers_config_path,
               save_compvis=False, save_diffusers=True, logger=logger)
    save_history(losses, run_tag, "nsfw")
    logger.info(f"Model and loss history saved under: models/{run_tag}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger, log_file = setup_logger(name="MUKSB_nsfw_rawsum")

    parser = argparse.ArgumentParser(
        description="MUKSB_RawSum: weighted raw-sum direction ablation for NSFW concept unlearning in Stable Diffusion"
    )
    parser.add_argument("--train_method",         type=str,   default="full")
    parser.add_argument("--batch_size",            type=int,   default=8)
    parser.add_argument("--epochs",                type=int,   default=1)
    parser.add_argument("--lr",                    type=float, default=1e-5)
    parser.add_argument("--ckpt_path",             type=str,
                        default="models/ldm/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--mask_variant",          type=str,   default="None",
                        choices=list(MASK_VARIANT_CHOICES) + [None],
                        help=(
                            "Parameter selection strategy for sparse update. "
                            "If omitted (default), all selected parameters are updated (no mask)."
                        ))
    parser.add_argument("--mask_density",          type=float, default=0.5,
                        help="Fraction ρ of parameters to update when using a mask (default: 0.5)")
    parser.add_argument("--lambda_tradeoff",       type=float, default=1.0,
                        help="λ in S_diff = F̂_f − λ·F̂_r  (dual_fisher mask only)")
    parser.add_argument("--forget_weight",         type=float, default=0.5,
                        help="Forget weight λ in g̃ = (1-λ)·g_r + λ·g_f  (ablation: {0.25, 0.5, 0.75})")
    parser.add_argument("--config_path",           type=str,
                        default="configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--diffusers_config_path", type=str,
                        default="diffusers_unet_config.json")
    parser.add_argument("--device",                type=str,   default="4")
    parser.add_argument("--image_size",            type=int,   default=512)
    parser.add_argument("--ddim_steps",            type=int,   default=50)
    parser.add_argument("--with_l1",               action="store_true", default=False)
    parser.add_argument("--alpha",                 type=float, default=1e-4,
                        help="L1 regularisation coefficient")
    parser.add_argument("--beta",                  type=float, default=100.0,
                        help="Scale factor for forget loss")
    parser.add_argument("--forget_path",           type=str,
                        default="/storage/s25017/Datasets/NSFW_removal/nude")
    parser.add_argument("--remain_path",           type=str,
                        default="/storage/s25017/Datasets/NSFW_removal/with_dress")
    args = parser.parse_args()

    logger.info("======== MUKSB_RawSum NSFW TRAINING STARTED ========")
    logger.info(f"Log file : {log_file}")
    logger.info(f"Args     : {vars(args)}")

    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    MUKSB_RawSum(
        train_method         = args.train_method,
        batch_size           = args.batch_size,
        epochs               = args.epochs,
        lr                   = args.lr,
        config_path          = args.config_path,
        ckpt_path            = args.ckpt_path,
        mask_variant         = args.mask_variant if args.mask_variant != "None" else None,
        mask_density         = args.mask_density,
        lambda_tradeoff      = args.lambda_tradeoff,
        forget_weight        = args.forget_weight,
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
