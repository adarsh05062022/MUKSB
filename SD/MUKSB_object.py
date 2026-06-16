"""
MUKSB_object.py — OBJECT concept removal on the SD-generated objectnette2 dataset
=================================================================================
This is the EXACT same KS-bargaining class-unlearning code as MUKSB_cls.py
(Imagenette-10), pointed at the SD-generated 10-class object dataset
(/storage/s25017/Datasets/objectnette2, ImageFolder layout).

You specify which class to forget at runtime (index 0-9 or class name); the
other 9 classes are the retain set. The forget loss uses the "next class"
pseudo-label, identical to MUKSB_cls.py:

    pseudo_prompt = descriptions[(class_to_forget + 1) % num_classes]
    forget loss   = MSE( eps(forget_img | forget_prompt),
                         eps(forget_img | pseudo_prompt).detach() )

Build the dataset once with build_objectnette.py, then:

    python MUKSB_object.py --class_to_forget dog --device 4 --epochs 5
    python MUKSB_object.py --class_to_forget 6   --device 4 --epochs 5
"""



# 0 airplane  1 bicycle  2 bird  3 boat  4 car
# 5 cat       6 dog       7 horse 8 train 9 truck

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

# taming-transformers is vendored but not pip-installed in the env
_TAMING = os.path.join(_THIS_DIR, "src", "taming-transformers")
if os.path.isdir(_TAMING) and _TAMING not in sys.path:
    sys.path.insert(0, _TAMING)

from logger.logger import setup_logger
from train_scripts.convertModels import savemodelDiffusers
from train_scripts.dataset import (
    setup_objectnette_forget_remain_data,
    setup_model,
    OBJECTNETTE_ROOT,
)
from mask_variants import build_mask, MASK_VARIANT_CHOICES


EXTRA = "objnette"

MODELS_ROOT = "/storage/s25017/SD_models"


# ─────────────────────────────────────────────────────────────────────────────
# KS bargaining core — identical to MUKSB_cls.py
# ─────────────────────────────────────────────────────────────────────────────

def ks_step(gr_flat: torch.Tensor, gf_flat: torch.Tensor, eps: float = 1e-8):
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
    effective_scale = 2.0 * norm_gr * norm_gf / (norm_gr + norm_gf)
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
    folder_path = f"{MODELS_ROOT}/{name}"
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
            device=device, num=num, models_root=MODELS_ROOT,
        )
    if not save_compvis and os.path.exists(path):
        os.remove(path)


def save_history(losses, name):
    folder_path = f"{MODELS_ROOT}/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(v) + "\n" for v in losses])
    v = np.convolve(losses, np.ones(min(3, len(losses))) / min(3, len(losses)), mode="valid")
    plt.figure()
    plt.plot(v, label="loss")
    plt.legend(loc="upper left")
    plt.title("MUKSB object training loss")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.savefig(f"{folder_path}/loss.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Class-name -> index resolution (ImageFolder uses sorted folder names)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_class_index(class_arg, root):
    """Resolve a class given as an int index or a class name to its ImageFolder
    label index. Returns (index:int, classes:list)."""
    train_dir = os.path.join(root, "train")
    classes = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    s = str(class_arg)
    if s.isdigit():
        idx = int(s)
        if not (0 <= idx < len(classes)):
            raise ValueError(f"class index {idx} out of range (0..{len(classes)-1})")
        return idx, classes
    if s not in classes:
        raise ValueError(f"class '{s}' not found. Available: {classes}")
    return classes.index(s), classes


# ─────────────────────────────────────────────────────────────────────────────
# Main unlearning function — mirrors MUKSB_cls.py
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
    root,
    logger,
):
    total_start = time.time()
    logger.info("======== MUKSB OBJECT (KS Bargaining) TRAINING STARTED ========")
    logger.info(f"class_to_forget={class_to_forget}  train_method={train_method}  root={root}")

    # ── model + data ─────────────────────────────────────────────────────────
    model    = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()

    forget_dl, remain_dl, descriptions = setup_objectnette_forget_remain_data(
        class_to_forget, batch_size, image_size, root=root,
    )
    num_classes = len(descriptions)
    num_forget  = len(forget_dl.dataset)
    pseudo_idx  = (int(class_to_forget) + 1) % num_classes
    logger.info(f"Classes ({num_classes}): {descriptions}")
    logger.info(f"Forget set: {num_forget} samples | forget='{descriptions[class_to_forget]}' "
                f"| pseudo(next)='{descriptions[pseudo_idx]}'")

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
            f"compvis-obj_{class_to_forget}-MUKSB-{mask_variant}"
            f"-rho{int(mask_density*100)}pct"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{EXTRA}"
        )
    else:
        mask = None
        name = (
            f"compvis-obj_{class_to_forget}-MUKSB"
            f"-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{EXTRA}"
        )

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    losses, step = [], 0
    skipped_steps   = 0
    total_steps     = 0
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
                pseudo_prompts = [
                    descriptions[(int(class_to_forget) + 1) % num_classes]
                    for _ in forget_labels
                ]

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

                if mask is not None:
                    gr_input = gr_flat[mask]
                    gf_input = gf_flat[mask]
                else:
                    gr_input = gr_flat
                    gf_input = gf_flat

                # ── KS bargaining merge ───────────────────────────────────────
                lambda_ks, cos_phi, g_star, effective_scale = ks_step(
                    gr_input, gf_input,
                )

                cos_phi_history.append(cos_phi.item())

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

                if mask is not None:
                    update_full = torch.zeros_like(gr_flat)
                    update_full[mask] = g_star_scaled
                else:
                    update_full = g_star_scaled

                del gr_flat, gf_flat, g_star, g_star_scaled

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
    logger.info("======== MUKSB OBJECT TRAINING FINISHED ========")
    logger.info(
        f"Total time: {total_time:.2f}s "
        f"({total_time/60:.2f} min | {total_time/3600:.2f} hrs)"
    )
    logger.info(
        f"Anti-parallel skips: {skipped_steps}/{total_steps} "
        f"({final_skip_rate*100:.1f}%)"
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
    logger.info(f"Model and loss history saved under: {MODELS_ROOT}/{name}/")
    print(f"RUN_TAG={name}")
    return name


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
            "MUKSB object-concept removal (KS Bargaining) on objectnette2 — "
            "same code path as Imagenette class removal, any of the 10 classes."
        )
    )

    parser.add_argument("--class_to_forget", type=str, default="dog",
                        help="class index (0-9) or class name (e.g. dog, car) to erase")
    parser.add_argument("--root", type=str, default=OBJECTNETTE_ROOT,
                        help="objectnette2 dataset root")
    parser.add_argument("--train_method",    type=str,   default="full",
                        choices=["full", "noxattn", "xattn", "selfattn",
                                 "notime", "xlayer", "selflayer"])
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--ckpt_path",   type=str,
                        default="/storage/s25017/models/ldm/sd-v1-4-full-ema.ckpt",
                        help="SD v1.4 checkpoint. Defaults to the fast local /storage "
                             "copy (NFS /scratch is much slower to read).")
    parser.add_argument("--mask_variant", type=_mask_variant_type, default="None",
                        choices=list(MASK_VARIANT_CHOICES) + [None],
                        help="parameter selection strategy for sparse update "
                             "(random / forget_fisher / salun / dual_fisher)")
    parser.add_argument("--mask_density",    type=float, default=0.5)
    parser.add_argument("--lambda_tradeoff", type=float, default=1.0)
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

    args = parser.parse_args()

    # resolve class name/index -> ImageFolder label index
    class_idx, classes = resolve_class_index(args.class_to_forget, args.root)

    logger, log_file = setup_logger(name=f"MUKSB_obj_{classes[class_idx]}_{EXTRA}")
    logger.info("======== MUKSB OBJECT STARTED ========")
    logger.info(f"Log: {log_file}")
    logger.info(f"Resolved class_to_forget='{args.class_to_forget}' -> "
                f"index {class_idx} ('{classes[class_idx]}')  |  all classes: {classes}")
    logger.info(f"Args: {vars(args)}")

    setup_seed(42)

    MUKSB(
        class_to_forget       = class_idx,
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
        root                  = args.root,
        logger                = logger,
    )
