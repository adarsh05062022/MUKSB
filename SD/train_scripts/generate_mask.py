"""
SD/train_scripts/generate_mask.py — MUKSB
Generate gradient saliency masks for Stable Diffusion unlearning.

Two modes
---------
  --nsfw   : mask built from NSFW forget data (gradient ascent on shared_step)
  default  : mask built from Imagenette forget class (classifier-guided ascent)

Usage (run from MUNBa/SD/ so ldm resolves correctly)
-----
  python /storage/s25017/MUKSB/SD/train_scripts/generate_mask.py \\
      --classes 4 --device 0

  python /storage/s25017/MUKSB/SD/train_scripts/generate_mask.py \\
      --nsfw --device 0
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SD_DIR   = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SD_DIR not in sys.path:
    sys.path.insert(0, _SD_DIR)

from train_scripts.dataset import setup_forget_data, setup_model


# ─────────────────────────────────────────────────────────────────────────────
# Imagenette class mask
# ─────────────────────────────────────────────────────────────────────────────

def generate_mask(
    classes,
    c_guidance,
    batch_size,
    lr,
    config_path,
    ckpt_path,
    device,
    image_size=512,
):
    """Compute gradient-magnitude mask for one Imagenette class."""
    model = setup_model(config_path, ckpt_path, device)
    train_dl, descriptions = setup_forget_data(classes, batch_size, image_size)
    print("Descriptions:", descriptions)

    model.eval()
    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.diffusion_model.parameters(), lr=lr)

    gradients = {name: 0 for name, _ in model.model.diffusion_model.named_parameters()}

    with tqdm(total=len(train_dl), desc=f"Mask cls{classes}") as pbar:
        for images, labels in train_dl:
            optimizer.zero_grad()
            images = images.to(device)

            prompts      = [descriptions[label] for label in labels]
            null_prompts = [""] * len(labels)

            forget_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": prompts}
            null_batch   = {"jpg": images.permute(0, 2, 3, 1), "txt": null_prompts}

            forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
            _,            null_emb   = model.get_input(null_batch,   model.first_stage_key)

            t     = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=device).long()
            noise = torch.randn_like(forget_input, device=device)

            forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
            forget_out   = model.apply_model(forget_noisy, t, forget_emb)
            null_out     = model.apply_model(forget_noisy, t, null_emb)

            preds = (1 + c_guidance) * forget_out - c_guidance * null_out
            loss  = -criteria(noise, preds)
            loss.backward()

            with torch.no_grad():
                for name, param in model.model.diffusion_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data.cpu()
            pbar.update(1)

    _save_mask(gradients, os.path.join("mask", str(classes)),
               threshold_list=[0.2, 0.3, 0.7, 0.8])


# ─────────────────────────────────────────────────────────────────────────────
# NSFW mask
# ─────────────────────────────────────────────────────────────────────────────

def generate_nsfw_mask(
    batch_size,
    lr,
    config_path,
    ckpt_path,
    device,
    forget_path="./dataFolder/NSFW",
    remain_path="./dataFolder/NotNSFW",
    image_size=512,
):
    """Compute gradient-magnitude mask for NSFW removal."""
    from train_scripts.dataset import setup_nsfw_data
    model = setup_model(config_path, ckpt_path, device)
    forget_dl, _ = setup_nsfw_data(batch_size, forget_path, remain_path, image_size)
    print(f"NSFW forget samples: {len(forget_dl.dataset)}")

    model.eval()
    optimizer = torch.optim.Adam(model.model.diffusion_model.parameters(), lr=lr)

    gradients = {name: 0 for name, _ in model.model.diffusion_model.named_parameters()}

    with tqdm(total=len(forget_dl), desc="Mask NSFW") as pbar:
        for forget_batch in forget_dl:
            optimizer.zero_grad()
            loss = -model.shared_step(forget_batch)[0]
            loss.backward()

            with torch.no_grad():
                for name, param in model.model.diffusion_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data.cpu()
                    else:
                        gradients[name] += torch.zeros_like(param.data.cpu())
            pbar.update(1)

    os.makedirs("mask", exist_ok=True)
    _save_mask(gradients, "mask/nsfw",
               threshold_list=[0.2, 0.3, 0.7, 0.8],
               filename_prefix="with_")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: build binary masks at multiple density thresholds
# ─────────────────────────────────────────────────────────────────────────────

def _save_mask(gradients, save_dir, threshold_list, filename_prefix="with_"):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        # Negate so argsort gives highest-magnitude params first
        all_elements = -torch.cat([t.flatten() for t in gradients.values()])
        positions    = torch.argsort(all_elements)
        ranks        = torch.argsort(positions)

        for threshold in threshold_list:
            threshold_index = int(len(all_elements) * threshold)
            hard_dict = {}
            start = 0
            for key, tensor in gradients.items():
                n             = tensor.numel()
                tensor_ranks  = ranks[start: start + n]
                mask_flat     = torch.zeros_like(tensor_ranks)
                mask_flat[tensor_ranks < threshold_index] = 1
                hard_dict[key] = mask_flat.reshape(tensor.shape)
                start += n
            out_path = os.path.join(save_dir, f"{filename_prefix}{threshold}.pt")
            torch.save(hard_dict, out_path)
            print(f"Saved mask (density={threshold}) → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MUKSB: Generate saliency mask for SD unlearning"
    )
    parser.add_argument("--classes",    type=int,   default=6,
                        help="Imagenette class index (0–9); ignored if --nsfw")
    parser.add_argument("--c_guidance", type=float, default=7.5)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--ckpt_path",  type=str,
                        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--config_path", type=str,
                        default="configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--device",     type=str,   default="0")
    parser.add_argument("--image_size", type=int,   default=512)
    parser.add_argument("--nsfw",       action="store_true", default=False,
                        help="Generate mask for NSFW removal instead of class removal")
    parser.add_argument("--forget_path", type=str,  default="./dataFolder/NSFW")
    parser.add_argument("--remain_path", type=str,  default="./dataFolder/NotNSFW")
    args = parser.parse_args()

    device = f"cuda:{int(args.device)}"

    if args.nsfw:
        generate_nsfw_mask(
            batch_size=args.batch_size,
            lr=args.lr,
            config_path=args.config_path,
            ckpt_path=args.ckpt_path,
            device=device,
            forget_path=args.forget_path,
            remain_path=args.remain_path,
            image_size=args.image_size,
        )
    else:
        generate_mask(
            classes=args.classes,
            c_guidance=args.c_guidance,
            batch_size=args.batch_size,
            lr=args.lr,
            config_path=args.config_path,
            ckpt_path=args.ckpt_path,
            device=device,
            image_size=args.image_size,
        )
