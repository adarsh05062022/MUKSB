"""
Evaluation/nsfw/compute_clip_nsfw.py
======================================
Compute CLIP image-text cosine similarity for generated I2P images.

Each image is paired with its original I2P prompt from the CSV.
The CLIP score measures how well the unlearned model still follows
the prompt (should stay high — prompt following is preserved, but nudity removed).

Outputs (saved in gen_dir):
    clip_nsfw_results.json
    clip_nsfw_histogram.png

Usage
-----
    python Evaluation/nsfw/compute_clip_nsfw.py \\
        --gen_dir      Evaluation/nsfw/generated/my-model \\
        --prompts_path /storage/s25017/Datasets/I2P_4703/unsafe-prompts4703.csv \\
        --device       0
"""

import argparse
import json
import os

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

try:
    import clip
except ImportError:
    raise SystemExit("[ERROR] openai-clip not found.\nInstall:  pip install openai-clip")

I2P_CSV_DEFAULT = "/storage/s25017/Datasets/I2P_4703/unsafe-prompts4703.csv"


def tokenize_truncated(prompt: str, device: str):
    """Tokenize a prompt, silently truncating if it exceeds CLIP's 77-token limit."""
    return clip.tokenize([prompt], truncate=True).to(device)


def compute_clip_nsfw(gen_dir: str, prompts_path: str, device: str):
    # ── load CLIP ─────────────────────────────────────────────────────────────
    print("[CLIP] Loading ViT-B/32 …")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # ── load prompts CSV ──────────────────────────────────────────────────────
    df = pd.read_csv(prompts_path)
    if "case_number" not in df.columns and df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "row_idx"})
    case_to_prompt = {int(row["case_number"]): str(row["prompt"])
                      for _, row in df.iterrows()}

    # ── collect generated images ──────────────────────────────────────────────
    img_files = sorted([
        f for f in os.listdir(gen_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if not img_files:
        print(f"[CLIP] No images found in: {gen_dir}")
        return {}

    scores = []
    missing_prompt = 0
    truncated = 0

    for fname in tqdm(img_files, desc="CLIP (NSFW)"):
        try:
            case_number = int(fname.split("_")[0])
        except ValueError:
            continue

        prompt = case_to_prompt.get(case_number)
        if prompt is None:
            missing_prompt += 1
            continue

        img_path = os.path.join(gen_dir, fname)
        image    = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            img_tensor = preprocess(image).unsqueeze(0).to(device)

            # check token length before encoding; count truncations for logging
            raw_tokens = clip.tokenize([prompt], truncate=True)
            if raw_tokens[0, -1].item() != 0:   # last token non-zero → was truncated
                truncated += 1
            text_token = raw_tokens.to(device)

            img_emb    = model.encode_image(img_tensor).float()
            text_emb   = model.encode_text(text_token).float()
            img_emb    = img_emb  / img_emb.norm(dim=-1, keepdim=True)
            text_emb   = text_emb / text_emb.norm(dim=-1, keepdim=True)
            cos_sim    = (img_emb * text_emb).sum(dim=-1).item()
        scores.append({"case_number": case_number, "clip_score": cos_sim})

    if missing_prompt:
        print(f"[CLIP] {missing_prompt} images had no matching prompt in CSV.")
    if truncated:
        print(f"[CLIP] {truncated} prompts were truncated to fit CLIP's 77-token limit.")

    avg_score = sum(s["clip_score"] for s in scores) / max(len(scores), 1)
    model_tag = os.path.basename(gen_dir.rstrip("/"))

    # ── histogram ─────────────────────────────────────────────────────────────
    vals = [s["clip_score"] for s in scores]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(avg_score, color="crimson", linewidth=1.5,
               label=f"mean = {avg_score:.4f}")
    ax.set_xlabel("CLIP cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"CLIP Score Distribution — {model_tag}")
    ax.legend()
    plt.tight_layout()
    hist_path = os.path.join(gen_dir, "clip_nsfw_histogram.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "model_tag":      model_tag,
        "n_images":       len(scores),
        "avg_clip_score": round(avg_score, 4),
        "n_truncated":    truncated,
        "per_image":      scores,
    }
    out_json = os.path.join(gen_dir, "clip_nsfw_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Model          : {model_tag}")
    print(f"  Images scored  : {len(scores)}")
    print(f"  Avg CLIP score : {avg_score:.4f}")
    print(f"  Truncated      : {truncated}")
    print(f"{'='*50}")
    print(f"  Saved: {out_json}")
    print(f"         {hist_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIP score for generated I2P NSFW images"
    )
    parser.add_argument("--gen_dir",      type=str, required=True,
                        help="Folder containing generated PNG images")
    parser.add_argument("--prompts_path", type=str, default=I2P_CSV_DEFAULT)
    parser.add_argument("--device",       type=str, default="0")
    args = parser.parse_args()

    compute_clip_nsfw(
        gen_dir      = args.gen_dir,
        prompts_path = args.prompts_path,
        device       = f"cuda:{args.device}",
    )