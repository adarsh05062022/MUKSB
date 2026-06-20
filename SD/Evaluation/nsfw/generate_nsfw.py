"""
Evaluation/nsfw/generate_nsfw.py
=================================
Generate images from the I2P dataset (unsafe-prompts4703.csv) using a
SSU-unlearned diffusers UNet checkpoint.

CSV columns used:
    case_number, prompt, evaluation_seed, evaluation_guidance

Saves to:
    <output_dir>/<model_tag>/<case_number>_<img_idx>.png

Usage
-----
    python Evaluation/nsfw/generate_nsfw.py \\
        --model_path  models/SSU-nsfw-.../SSU-nsfw-...-epoch_5.pt \\
        --output_dir  Evaluation/nsfw/generated \\
        --prompts_path /storage/s25017/Datasets/I2P_4703/unsafe-prompts4703.csv \\
        --device       0

    # Baseline (vanilla SD v1.4):
    python Evaluation/nsfw/generate_nsfw.py \\
        --output_dir  Evaluation/nsfw/generated \\
        --device       0
"""

import argparse
import gc
import os

import pandas as pd
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

I2P_CSV_DEFAULT = "/storage/s25017/Datasets/I2P_4703/unsafe-prompts4703.csv"


def load_pipeline(model_path: str, device: str):
    """Load SD v1.4 with optional unlearned UNet weights."""
    base = "CompVis/stable-diffusion-v1-4"
    vae          = AutoencoderKL.from_pretrained(base, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    unet         = UNet2DConditionModel.from_pretrained(base, subfolder="unet")

    if model_path and os.path.exists(model_path):
        if os.path.isdir(model_path):
            # HuggingFace directory with pytorch_model.bin + config.json (AdvUnlearn HF format)
            text_encoder = CLIPTextModel.from_pretrained(model_path)
            print(f"[TextEncoder] loaded HF dir {model_path}")
        else:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if "state_dict" in state:
                state = state["state_dict"]
            first_key = next(iter(state))
            if first_key.startswith("text_model."):
                # AdvUnlearn-style: text encoder checkpoint
                # Strip text_model. prefix if model uses bare keys (newer transformers)
                model_first_key = next(iter(text_encoder.state_dict()))
                if not model_first_key.startswith("text_model."):
                    state = {k.replace("text_model.", "", 1): v for k, v in state.items()}
                missing, unexpected = text_encoder.load_state_dict(state, strict=False)
                print(f"[TextEncoder] loaded {model_path} | missing={len(missing)}  unexpected={len(unexpected)}")
            else:
                # SSU/MUKSB-style: unlearned UNet checkpoint
                unet_state = {k.replace("model.diffusion_model.", ""): v
                              for k, v in state.items()
                              if k.startswith("model.diffusion_model.")}
                if unet_state:
                    missing, unexpected = unet.load_state_dict(unet_state, strict=False)
                else:
                    missing, unexpected = unet.load_state_dict(state, strict=False)
                print(f"[UNet] loaded {model_path} | missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        print("[UNet] No checkpoint — using vanilla SD v1.4.")

    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", num_train_timesteps=1000,
    )
    vae.to(device); text_encoder.to(device); unet.to(device)
    vae.eval(); text_encoder.eval(); unet.eval()
    return vae, tokenizer, text_encoder, unet, scheduler


@torch.no_grad()
def generate_one(
    vae, tokenizer, text_encoder, unet, scheduler,
    prompt: str, seed: int, guidance_scale: float,
    image_size: int, ddim_steps: int, device: str,
) -> "Image.Image":
    """Generate a single image."""
    text_input = tokenizer(
        [prompt], padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_emb = text_encoder(text_input.input_ids.to(device))[0]
    uncond_inp = tokenizer(
        [""], padding="max_length",
        max_length=text_input.input_ids.shape[-1], return_tensors="pt",
    )
    uncond_emb = text_encoder(uncond_inp.input_ids.to(device))[0]
    cond_emb   = torch.cat([uncond_emb, text_emb])

    scheduler.set_timesteps(ddim_steps)
    generator = torch.manual_seed(seed)
    latents   = torch.randn(
        (1, unet.in_channels, image_size // 8, image_size // 8),
        generator=generator,
    ).to(device)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        inp        = scheduler.scale_model_input(torch.cat([latents] * 2), t)
        noise_pred = unet(inp, t, encoder_hidden_states=cond_emb).sample
        u, c       = noise_pred.chunk(2)
        noise_pred = u + guidance_scale * (c - u)
        latents    = scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    image   = vae.decode(latents).sample
    image   = (image / 2 + 0.5).clamp(0, 1)
    image   = image.cpu().permute(0, 2, 3, 1).numpy()
    image   = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)


def generate_nsfw(
    model_path, output_dir, prompts_path,
    device, n_per_prompt=1, guidance_scale=7.5,
    image_size=512, ddim_steps=50,
    from_case=0, to_case=None,
):
    model_tag = (os.path.basename(model_path).replace(".pt", "")
                 if model_path else "sd14_baseline")
    save_dir  = os.path.join(output_dir, model_tag)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(prompts_path)
    # normalise column names (CSV has a leading unnamed index column)
    if "case_number" not in df.columns and df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "row_idx"})

    print(f"\n{'='*60}")
    print(f"Model     : {model_tag}")
    print(f"Prompts   : {prompts_path}  ({len(df)} rows)")
    print(f"Images/prompt: {n_per_prompt}")
    print(f"Output    : {save_dir}")
    print(f"{'='*60}\n")

    vae, tokenizer, text_encoder, unet, scheduler = load_pipeline(model_path, device)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating NSFW"):
        case_number = int(row["case_number"])
        if case_number < from_case:
            continue
        if to_case is not None and case_number >= to_case:
            continue

        prompt       = str(row["prompt"])
        base_seed    = int(row["evaluation_seed"])
        img_guidance = float(row.get("evaluation_guidance", guidance_scale))

        for img_idx in range(n_per_prompt):
            out_path = os.path.join(save_dir, f"{case_number:05d}_{img_idx}.png")
            if os.path.exists(out_path):
                continue  # already generated

            # offset seed per image so each sample is distinct
            seed = base_seed + img_idx

            try:
                img = generate_one(
                    vae, tokenizer, text_encoder, unet, scheduler,
                    prompt=prompt, seed=seed, guidance_scale=img_guidance,
                    image_size=image_size, ddim_steps=ddim_steps, device=device,
                )
                img.save(out_path)
            except Exception as e:
                print(f"[WARN] case {case_number} img {img_idx} failed: {e}")

    del vae, text_encoder, unet, scheduler
    gc.collect(); torch.cuda.empty_cache()

    n_saved = len([f for f in os.listdir(save_dir) if f.endswith(".png")])
    print(f"\nDone. {n_saved} images saved to: {save_dir}")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate I2P NSFW images from an SSU checkpoint"
    )
    parser.add_argument("--model_path",     type=str, default="",
                        help="SSU .pt checkpoint (empty = SD v1.4 baseline)")
    parser.add_argument("--output_dir",     type=str,
                        default="Evaluation/nsfw/coco_30k")
    parser.add_argument("--prompts_path",   type=str, default=I2P_CSV_DEFAULT)
    parser.add_argument("--device",         type=str, default="0")
    parser.add_argument("--n_per_prompt",   type=int, default=1,
                        help="Number of images to generate per prompt (default: 1)")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_size",     type=int, default=512)
    parser.add_argument("--ddim_steps",     type=int, default=50)
    parser.add_argument("--from_case",      type=int, default=0)
    parser.add_argument("--to_case",        type=int, default=None)
    args = parser.parse_args()

    generate_nsfw(
        model_path     = args.model_path,
        output_dir     = args.output_dir,
        prompts_path   = args.prompts_path,
        device         = f"cuda:{args.device}",
        n_per_prompt   = args.n_per_prompt,
        guidance_scale = args.guidance_scale,
        image_size     = args.image_size,
        ddim_steps     = args.ddim_steps,
        from_case      = args.from_case,
        to_case        = args.to_case,
    )