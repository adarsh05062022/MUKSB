"""
SD3/Evaluation/nsfw/generate_nsfw.py
======================================
Generate NSFW evaluation images from a fine-tuned SD3 transformer checkpoint
(or the original SD3 base model) using I2P prompts.

Requires: conda activate munba3_sd3   (diffusers >= 0.29)

Usage — fine-tuned checkpoint:
    python Evaluation/nsfw/generate_nsfw.py \\
        --transformer_path models/sd3-nsfw-MUKSB-method_full-lr_1e-05_E5_U800/transformer-epoch_1 \\
        --output_dir       Evaluation/nsfw/generated \\
        --device           0

Usage — SD3 baseline (no checkpoint):
    python Evaluation/nsfw/generate_nsfw.py \\
        --output_dir Evaluation/nsfw/generated \\
        --device     0
"""

import argparse
import csv
import os

import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel

torch.backends.cudnn.benchmark = False

I2P_CSV_DEFAULT = "/scratch/s25017/MUKSB/SD/prompts/limitedi2p.csv"
BASE_MODEL_ID   = "stabilityai/stable-diffusion-3-medium-diffusers"


def load_csv_prompts(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "prompt":      row["prompt"].strip(),
                "seed":        int(row["evaluation_seed"]),
                "guidance":    float(row["evaluation_guidance"]) if "evaluation_guidance" in row else None,
                "case_number": int(row["case_number"]),
            })
    return rows


def generate_nsfw(
    transformer_path,
    output_dir,
    prompts_path,
    device,
    n_per_prompt       = 1,
    guidance_scale     = 7.0,
    image_size         = 1024,
    steps              = 28,
    base_model_id      = BASE_MODEL_ID,
    dtype              = torch.bfloat16,
    skip_t5            = False,
    max_sequence_length= 512,
):
    model_tag = (
        os.path.basename(transformer_path.rstrip("/"))
        if transformer_path else "sd3_baseline"
    )
    gen_dir = os.path.join(output_dir, model_tag)
    os.makedirs(gen_dir, exist_ok=True)

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

    prompt_rows = load_csv_prompts(prompts_path)
    print(f"Loaded {len(prompt_rows)} prompts from: {prompts_path}")
    total = len(prompt_rows) * n_per_prompt

    count = 0
    for row in prompt_rows:
        prompt   = row["prompt"]
        seed     = row["seed"]
        guidance = row["guidance"] if row["guidance"] is not None else guidance_scale
        case_num = row["case_number"]

        for i in range(n_per_prompt):
            count += 1
            cur_seed = seed + i
            print(f"[{count}/{total}] case={case_num:05d}  img={i}  seed={cur_seed}  cfg={guidance}")
            print(f"  Prompt: {prompt[:80]}")

            torch.cuda.empty_cache()
            generator = torch.Generator(device=device).manual_seed(cur_seed)
            image = pipe(
                prompt              = prompt,
                num_inference_steps = steps,
                guidance_scale      = guidance,
                height              = image_size,
                width               = image_size,
                generator           = generator,
                max_sequence_length = max_sequence_length,
            ).images[0]

            fname = f"{case_num:05d}_{i}.png"
            image.save(os.path.join(gen_dir, fname))
            print(f"  Saved: {fname}")

    print(f"\nDone. {total} images saved to: {gen_dir}")
    return gen_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate NSFW evaluation images from a fine-tuned SD3 transformer checkpoint"
    )
    parser.add_argument("--transformer_path", type=str,   default=None,
                        help="Path to fine-tuned transformer directory. Omit for SD3 baseline.")
    parser.add_argument("--base_model_id",    type=str,   default=BASE_MODEL_ID)
    parser.add_argument("--output_dir",       type=str,   default="Evaluation/nsfw/generated")
    parser.add_argument("--prompts_path",     type=str,   default=I2P_CSV_DEFAULT)
    parser.add_argument("--device",           type=str,   default="0")
    parser.add_argument("--n_per_prompt",     type=int,   default=1)
    parser.add_argument("--guidance_scale",   type=float, default=7.0)
    parser.add_argument("--image_size",       type=int,   default=1024)
    parser.add_argument("--steps",            type=int,   default=28)
    parser.add_argument("--dtype",            type=str,   default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--skip_t5",          action="store_true", default=False)
    args = parser.parse_args()

    _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    generate_nsfw(
        transformer_path = args.transformer_path,
        output_dir       = args.output_dir,
        prompts_path     = args.prompts_path,
        device           = f"cuda:{int(args.device)}",
        n_per_prompt     = args.n_per_prompt,
        guidance_scale   = args.guidance_scale,
        image_size       = args.image_size,
        steps            = args.steps,
        base_model_id    = args.base_model_id,
        dtype            = _dtype_map[args.dtype],
        skip_t5          = args.skip_t5,
    )
