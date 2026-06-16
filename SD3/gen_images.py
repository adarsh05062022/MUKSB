"""
SD3/gen_images.py
=================
Generate images from a fine-tuned SD3 transformer checkpoint.
Supports CSV prompt files (i2p format) or manual --prompts.

Usage — CSV mode:
    python3 gen_images.py \
        --transformer_path models/sd3-nsfw-MUKSB-method_attn-.../transformer-epoch_1 \
        --csv /scratch/s25017/MUKSB/SD/prompts/limitedi2p.csv \
        --out_dir outputs/epoch1 \
        --device 0

Usage — manual prompts:
    python3 gen_images.py \
        --transformer_path models/sd3-nsfw-MUKSB-.../transformer-epoch_1 \
        --prompts "a photo of a nude person" "a photo of a person wearing clothes" \
        --out_dir outputs/epoch1 \
        --device 0
"""

import argparse
import csv
import os
import sys

import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def load_csv_prompts(csv_path):
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
    transformer_path, base_model_id, prompt_rows,
    out_dir, device, steps, image_size, dtype, skip_t5,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading base SD3 pipeline from: {base_model_id}")
    kwargs = dict(torch_dtype=dtype, use_safetensors=True)
    if skip_t5:
        kwargs["text_encoder_3"] = None
        kwargs["tokenizer_3"]    = None

    pipe = StableDiffusion3Pipeline.from_pretrained(base_model_id, **kwargs)

    if transformer_path:
        print(f"Swapping transformer from: {transformer_path}")
        pipe.transformer = SD3Transformer2DModel.from_pretrained(
            transformer_path, torch_dtype=dtype
        )
    else:
        print("No --transformer_path given — using original SD3 transformer.")

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(leave=False)

    total = len(prompt_rows)
    for idx, row in enumerate(prompt_rows):
        prompt   = row["prompt"]
        seed     = row["seed"]
        guidance = row["guidance"]
        case_num = row["case_number"]

        print(f"[{idx+1}/{total}] case={case_num}  seed={seed}  cfg={guidance}")
        print(f"  Prompt: {prompt[:80]}")

        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt              = prompt,
            num_inference_steps = steps,
            guidance_scale      = guidance,
            height              = image_size,
            width               = image_size,
            generator           = generator,
        ).images[0]

        image.save(os.path.join(out_dir, f"{case_num}.png"))
        print(f"  Saved: {case_num}.png")

    print(f"\nDone. {total} images saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from a fine-tuned SD3 transformer checkpoint"
    )
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Path to fine-tuned transformer checkpoint. Omit to use the base SD3 model.")
    parser.add_argument("--base_model_id", type=str,
                        default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--csv",     type=str, default=None)
    parser.add_argument("--prompts", type=str, nargs="+", default=None)
    parser.add_argument("--out_dir",        type=str,   default="outputs/gen")
    parser.add_argument("--device",         type=str,   default="0")
    parser.add_argument("--steps",          type=int,   default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--image_size",     type=int,   default=1024)
    parser.add_argument("--dtype",          type=str,   default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--skip_t5",        action="store_true", default=False)
    args = parser.parse_args()

    _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

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
        transformer_path = args.transformer_path,
        base_model_id    = args.base_model_id,
        prompt_rows      = prompt_rows,
        out_dir          = args.out_dir,
        device           = f"cuda:{int(args.device)}",
        steps            = args.steps,
        image_size       = args.image_size,
        dtype            = _dtype_map[args.dtype],
        skip_t5          = args.skip_t5,
    )
