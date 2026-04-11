"""
Evaluation/imagenette/generate_imagenette.py
=============================================
Generate 500 images per class for all 10 Imagenette classes using a
SSU-unlearned diffusers UNet checkpoint.

Saves to:
    <output_dir>/<model_tag>/<class_name>/   (500 PNGs per class)

Usage
-----
    python Evaluation/imagenette/generate_imagenette.py \\
        --model_path  models/SSU-cls_0-full-lr_1e-05-E5-rho10pct-both/SSU-cls_0-...-epoch_5.pt \\
        --output_dir  Evaluation/imagenette/generated \\
        --device      0 \\
        --n_per_class 500
"""

import argparse
import gc
import os

import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# ── Imagenette class metadata ─────────────────────────────────────────────────
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]
IMAGENETTE_WNIDS = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
]
PROMPTS = [f"an image of a {c}" for c in IMAGENETTE_CLASSES]

# seed per class — fixed so results are reproducible across runs
CLASS_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def load_pipeline(model_path: str, device: str):
    """Load SD v1.4 with the unlearned UNet weights."""
    base = "CompVis/stable-diffusion-v1-4"
    vae          = AutoencoderKL.from_pretrained(base, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    unet         = UNet2DConditionModel.from_pretrained(base, subfolder="unet")

    if model_path and os.path.exists(model_path):
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        # SSU saves the full model; extract UNet sub-state if needed
        unet_state = {k.replace("model.diffusion_model.", ""): v
                      for k, v in state.items()
                      if k.startswith("model.diffusion_model.")}
        if unet_state:
            missing, unexpected = unet.load_state_dict(unet_state, strict=False)
        else:
            missing, unexpected = unet.load_state_dict(state, strict=False)
        print(f"[UNet] loaded {model_path} | missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        print("[UNet] No checkpoint path provided — using vanilla SD v1.4.")

    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", num_train_timesteps=1000,
    )
    vae.to(device); text_encoder.to(device); unet.to(device)
    vae.eval(); text_encoder.eval(); unet.eval()
    return vae, tokenizer, text_encoder, unet, scheduler


@torch.no_grad()
def generate_class_images(
    vae, tokenizer, text_encoder, unet, scheduler,
    prompt: str, n_images: int, seed: int,
    device: str, guidance_scale: float, image_size: int, ddim_steps: int,
    save_dir: str,
):
    """Generate n_images for one class prompt and save as PNGs."""
    os.makedirs(save_dir, exist_ok=True)

    text_input = tokenizer(
        [prompt], padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_emb    = text_encoder(text_input.input_ids.to(device))[0]
    uncond_inp  = tokenizer(
        [""], padding="max_length",
        max_length=text_input.input_ids.shape[-1], return_tensors="pt",
    )
    uncond_emb  = text_encoder(uncond_inp.input_ids.to(device))[0]

    saved = 0

    # generate in batches of 4 for memory efficiency
    batch_size = 4
    while saved < n_images:
        this_batch = min(batch_size, n_images - saved)
        generator  = torch.manual_seed(seed + saved)

        # reset scheduler state each batch (LMSDiscreteScheduler tracks step_index)
        scheduler.set_timesteps(ddim_steps)

        latents = torch.randn(
            (this_batch, unet.config.in_channels, image_size // 8, image_size // 8),
            generator=generator,
        ).to(device)
        latents = latents * scheduler.init_noise_sigma

        cond = torch.cat([
            uncond_emb.expand(this_batch, -1, -1),
            text_emb.expand(this_batch, -1, -1),
        ])

        for t in scheduler.timesteps:
            inp        = scheduler.scale_model_input(torch.cat([latents] * 2), t)
            noise_pred = unet(inp, t, encoder_hidden_states=cond).sample
            u, c       = noise_pred.chunk(2)
            noise_pred = u + guidance_scale * (c - u)
            latents    = scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        images  = vae.decode(latents).sample
        images  = (images / 2 + 0.5).clamp(0, 1)
        images  = images.cpu().permute(0, 2, 3, 1).numpy()
        images  = (images * 255).round().astype("uint8")

        for img_arr in images:
            Image.fromarray(img_arr).save(
                os.path.join(save_dir, f"{saved:05d}.png")
            )
            saved += 1

    return saved


def generate_all_classes(
    model_path, output_dir, device,
    n_per_class=500, guidance_scale=7.5,
    image_size=512, ddim_steps=50,
):
    model_tag = (os.path.basename(model_path).replace(".pt", "")
                 if model_path else "sd14_baseline")
    out_root  = os.path.join(output_dir, model_tag)
    print(f"\n{'='*60}")
    print(f"Model   : {model_tag}")
    print(f"Classes : {len(IMAGENETTE_CLASSES)}  |  images/class : {n_per_class}")
    print(f"Output  : {out_root}")
    print(f"{'='*60}\n")

    vae, tokenizer, text_encoder, unet, scheduler = load_pipeline(model_path, device)

    for cls_idx, (cls_name, prompt, seed) in enumerate(
        zip(IMAGENETTE_CLASSES, PROMPTS, CLASS_SEEDS)
    ):
        save_dir = os.path.join(out_root, f"{cls_idx:02d}_{cls_name.replace(' ', '_')}")
        # skip if already done
        existing = len([f for f in os.listdir(save_dir) if f.endswith(".png")]) if os.path.exists(save_dir) else 0
        if existing >= n_per_class:
            print(f"[SKIP] class {cls_idx:2d} {cls_name:<20}  ({existing} images already exist)")
            continue

        print(f"[GEN]  class {cls_idx:2d} {cls_name:<20}  prompt: '{prompt}'")
        n_saved = generate_class_images(
            vae, tokenizer, text_encoder, unet, scheduler,
            prompt=prompt, n_images=n_per_class, seed=seed,
            device=device, guidance_scale=guidance_scale,
            image_size=image_size, ddim_steps=ddim_steps,
            save_dir=save_dir,
        )
        print(f"         → saved {n_saved} images to {save_dir}")

    del vae, text_encoder, unet, scheduler
    gc.collect(); torch.cuda.empty_cache()
    print(f"\nAll classes done. Root: {out_root}")
    return out_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 500 images per Imagenette class from an SSU checkpoint"
    )
    parser.add_argument("--model_path",     type=str, default="",
                        help="Path to SSU .pt checkpoint. Leave empty for SD v1.4 baseline.")
    parser.add_argument("--output_dir",     type=str,
                        default="Evaluation/imagenette/generated")
    parser.add_argument("--device",         type=str, default="0",
                        help="CUDA device index (e.g. '0')")
    parser.add_argument("--n_per_class",    type=int, default=500)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_size",     type=int, default=512)
    parser.add_argument("--ddim_steps",     type=int, default=50)
    args = parser.parse_args()

    generate_all_classes(
        model_path     = args.model_path,
        output_dir     = args.output_dir,
        device         = f"cuda:{args.device}",
        n_per_class    = args.n_per_class,
        guidance_scale = args.guidance_scale,
        image_size     = args.image_size,
        ddim_steps     = args.ddim_steps,
    )
