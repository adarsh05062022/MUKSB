"""
SDXL/Evaluation/nsfw/generate_nsfw.py
======================================
Generate NSFW evaluation images from a fine-tuned SDXL UNet checkpoint
(or the original SDXL base model) using I2P prompts.

Usage — fine-tuned checkpoint:
    python Evaluation/nsfw/generate_nsfw.py \\
        --unet_path  models/sdxl-nsfw-MUKSB-method_full-lr_1e-05_E5_U800/unet-epoch_1 \\
        --output_dir Evaluation/nsfw/generated \\
        --device     0

Usage — SDXL baseline (no checkpoint):
    python Evaluation/nsfw/generate_nsfw.py \\
        --output_dir Evaluation/nsfw/generated \\
        --device     0
"""

import argparse
import csv
import os

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel


def _encode_prompt_sdxl(pipe, prompt, device, dtype):
    """SDXL chunked prompt encoding — bypasses the 77-token CLIP limit.

    SDXL has two text encoders (CLIP-L + OpenCLIP-G).  Each is encoded with
    75-token chunks; the penultimate hidden states are averaged across chunks
    and concatenated across encoders → [1, 77, 2048].  The pooled embedding
    from OpenCLIP-G is also averaged across chunks → [1, 1280].
    """

    def chunk_encode(tokenizer, text_encoder, text, get_pooled=False):
        max_len    = tokenizer.model_max_length          # 77
        chunk_size = max_len - 2                         # 75 content tokens
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

        raw_ids = tokenizer(text, truncation=False, padding=False).input_ids
        core    = [t for t in raw_ids if t not in (bos, eos, pad)]
        chunks  = [core[i:i + chunk_size] for i in range(0, max(len(core), 1), chunk_size)]

        seq_embs, pool_embs = [], []
        for chunk in chunks:
            padded = [bos] + chunk + [eos] + [pad] * (chunk_size - len(chunk))
            tensor = torch.tensor([padded], dtype=torch.long).to(device)
            with torch.no_grad():
                out = text_encoder(tensor, output_hidden_states=True)
            seq_embs.append(out.hidden_states[-2])       # [1, 77, dim]
            if get_pooled:
                pool_embs.append(out[0])                 # [1, dim]

        avg_seq  = torch.stack(seq_embs).mean(dim=0)    # [1, 77, dim]
        avg_pool = torch.stack(pool_embs).mean(dim=0) if pool_embs else None
        return avg_seq, avg_pool

    hs1, _      = chunk_encode(pipe.tokenizer,   pipe.text_encoder,   prompt)
    hs2, pooled = chunk_encode(pipe.tokenizer_2, pipe.text_encoder_2, prompt, get_pooled=True)

    prompt_embeds = torch.cat([hs1, hs2], dim=-1).to(dtype=dtype)  # [1, 77, 2048]
    pooled_embeds = pooled.to(dtype=dtype)                          # [1, 1280]
    return prompt_embeds, pooled_embeds

I2P_CSV_DEFAULT = "/scratch/s25017/MUKSB/SD/prompts/limitedi2p.csv"
BASE_MODEL_ID   = "stabilityai/stable-diffusion-xl-base-1.0"


def load_csv_prompts(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "prompt":      row["prompt"].strip(),
                "seed":        int(row["evaluation_seed"]),
                "guidance":    float(row["evaluation_guidance"]) if "evaluation_guidance" in row else 7.5,
                "case_number": int(row["case_number"]),
            })
    return rows


def generate_nsfw(
    unet_path,
    output_dir,
    prompts_path,
    device,
    n_per_prompt  = 1,
    guidance_scale= 7.5,
    image_size    = 1024,
    steps         = 30,
    base_model_id = BASE_MODEL_ID,
    dtype         = torch.bfloat16,
):
    model_tag = (
        os.path.basename(unet_path.rstrip("/")) if unet_path else "sdxl_baseline"
    )
    gen_dir = os.path.join(output_dir, model_tag)
    os.makedirs(gen_dir, exist_ok=True)

    print(f"Loading base SDXL pipeline from: {base_model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id, torch_dtype=dtype, use_safetensors=True
    )

    if unet_path:
        print(f"Swapping UNet from: {unet_path}")
        pipe.unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
    else:
        print("No --unet_path given — using original SDXL UNet.")

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(leave=False)

    prompt_rows = load_csv_prompts(prompts_path)
    print(f"Loaded {len(prompt_rows)} prompts from: {prompts_path}")
    total = len(prompt_rows) * n_per_prompt

    count = 0
    for row in prompt_rows:
        prompt   = row["prompt"]
        seed     = row["seed"]
        guidance = row["guidance"]
        case_num = row["case_number"]

        for i in range(n_per_prompt):
            count += 1
            cur_seed = seed + i
            print(f"[{count}/{total}] case={case_num:05d}  img={i}  seed={cur_seed}  cfg={guidance}")
            print(f"  Prompt: {prompt}")

            generator = torch.Generator(device=device).manual_seed(cur_seed)

            pos_embeds, pos_pooled = _encode_prompt_sdxl(pipe, prompt, device, pipe.unet.dtype)
            neg_embeds, neg_pooled = _encode_prompt_sdxl(pipe, "",     device, pipe.unet.dtype)

            image = pipe(
                prompt_embeds                  = pos_embeds,
                pooled_prompt_embeds           = pos_pooled,
                negative_prompt_embeds         = neg_embeds,
                negative_pooled_prompt_embeds  = neg_pooled,
                num_inference_steps            = steps,
                guidance_scale                 = guidance,
                height                         = image_size,
                width                          = image_size,
                generator                      = generator,
            ).images[0]

            fname = f"{case_num:05d}_{i}.png"
            image.save(os.path.join(gen_dir, fname))
            print(f"  Saved: {fname}")

    print(f"\nDone. {total} images saved to: {gen_dir}")
    return gen_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate NSFW evaluation images from a fine-tuned SDXL UNet checkpoint"
    )
    parser.add_argument("--unet_path",      type=str,   default=None,
                        help="Path to fine-tuned UNet directory. Omit for SDXL baseline.")
    parser.add_argument("--base_model_id",  type=str,   default=BASE_MODEL_ID)
    parser.add_argument("--output_dir",     type=str,   default="Evaluation/nsfw/generated")
    parser.add_argument("--prompts_path",   type=str,   default=I2P_CSV_DEFAULT)
    parser.add_argument("--device",         type=str,   default="0")
    parser.add_argument("--n_per_prompt",   type=int,   default=1)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_size",     type=int,   default=1024)
    parser.add_argument("--steps",          type=int,   default=30)
    parser.add_argument("--dtype",          type=str,   default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    generate_nsfw(
        unet_path      = args.unet_path,
        output_dir     = args.output_dir,
        prompts_path   = args.prompts_path,
        device         = f"cuda:{int(args.device)}",
        n_per_prompt   = args.n_per_prompt,
        guidance_scale = args.guidance_scale,
        image_size     = args.image_size,
        steps          = args.steps,
        base_model_id  = args.base_model_id,
        dtype          = _dtype_map[args.dtype],
    )
