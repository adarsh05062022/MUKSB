"""
SDXL/gen_images.py
==================
Generate images from a fine-tuned SDXL UNet checkpoint, or from the
original SDXL base model without any checkpoint (omit --unet_path).
Supports a CSV prompt file (i2p format) or manual --prompts.

Usage — original SDXL (no checkpoint):
    python3 gen_images.py \
        --csv /scratch/s25017/MUKSB/SD/prompts/limitedi2p.csv \
        --out_dir outputs/original \
        --device 0

Usage — fine-tuned UNet checkpoint, CSV mode:
    python3 gen_images.py \
        --unet_path models/sdxl-nsfw-MUKSB-method_full-lr_1e-05_E5_U800/unet-epoch_1 \
        --csv /scratch/s25017/MUKSB/SD/prompts/limitedi2p.csv \
        --out_dir outputs/epoch1 \
        --device 0

Usage — fine-tuned UNet checkpoint, manual prompts:
    python3 gen_images.py \
        --unet_path models/sdxl-nsfw-MUKSB-method_full-lr_1e-05_E5_U800/unet-epoch_1 \
        --prompts "a photo of a nude person" "a photo of a person wearing clothes" \
        --out_dir outputs/epoch1 \
        --device 0
"""

import argparse
import csv
import os
import sys

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def load_csv_prompts(csv_path):
    """
    Read prompts from an i2p-style CSV.
    Returns list of dicts with keys: prompt, seed, guidance, case_number, width, height.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "prompt":      row["prompt"].strip(),
                "seed":        int(row["evaluation_seed"]),
                "guidance":    float(row["evaluation_guidance"]),
                "case_number": str(row["case_number"]).strip(),
            })
    return rows


def generate(
    unet_path,
    base_model_id,
    prompt_rows,        # list of dicts: prompt, seed, guidance, case_number
    out_dir,
    device,
    steps,
    image_size,
    dtype,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading base pipeline from: {base_model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if unet_path:
        print(f"Swapping UNet from: {unet_path}")
        pipe.unet = UNet2DConditionModel.from_pretrained(
            unet_path,
            torch_dtype=dtype,
        )
    else:
        print("No --unet_path given — using original SDXL UNet.")

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(leave=False)

    total = len(prompt_rows)
    for idx, row in enumerate(prompt_rows):
        prompt      = row["prompt"]
        seed        = row["seed"]
        guidance    = row["guidance"]
        case_num    = row["case_number"]

        print(f"[{idx+1}/{total}] case={case_num}  seed={seed}  cfg={guidance}")
        print(f"  Prompt: {prompt[:80]}")

        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt             = prompt,
            num_inference_steps= steps,
            guidance_scale     = guidance,
            height             = image_size,
            width              = image_size,
            generator          = generator,
        ).images[0]

        fname = f"{case_num}.png"
        path  = os.path.join(out_dir, fname)
        image.save(path)
        print(f"  Saved: {path}")

    print(f"\nDone. {total} images saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from a fine-tuned SDXL UNet checkpoint"
    )
    parser.add_argument(
        "--unet_path", type=str, default=None,
        help="Path to the saved UNet directory (contains config.json + safetensors). "
             "Omit to use the original SDXL UNet without any checkpoint.",
    )
    parser.add_argument(
        "--base_model_id", type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HuggingFace model ID or local path for the rest of the pipeline",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to i2p-style CSV with columns: prompt, evaluation_seed, evaluation_guidance, case_number",
    )
    parser.add_argument(
        "--prompts", type=str, nargs="+", default=None,
        help="Manual prompts (used if --csv is not provided)",
    )
    parser.add_argument("--out_dir",       type=str,   default="outputs/gen")
    parser.add_argument("--device",        type=str,   default="0")
    parser.add_argument("--steps",         type=int,   default=30)
    parser.add_argument("--guidance_scale",type=float, default=7.5,
                        help="Used only when --prompts is provided (CSV has per-row guidance)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Used only when --prompts is provided (CSV has per-row seed)")
    parser.add_argument("--image_size",    type=int,   default=1024)
    parser.add_argument("--dtype",         type=str,   default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    _dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }

    # Build prompt_rows from CSV or manual list
    if args.csv:
        prompt_rows = load_csv_prompts(args.csv)
        print(f"Loaded {len(prompt_rows)} prompts from: {args.csv}")
    elif args.prompts:
        prompt_rows = [
            {"prompt": p, "seed": args.seed,
             "guidance": args.guidance_scale, "case_number": f"{i:04d}"}
            for i, p in enumerate(args.prompts)
        ]
    else:
        parser.error("Provide either --csv or --prompts")

    generate(
        unet_path     = args.unet_path,
        base_model_id = args.base_model_id,
        prompt_rows   = prompt_rows,
        out_dir       = args.out_dir,
        device        = f"cuda:{int(args.device)}",
        steps         = args.steps,
        image_size    = args.image_size,
        dtype         = _dtype_map[args.dtype],
    )
