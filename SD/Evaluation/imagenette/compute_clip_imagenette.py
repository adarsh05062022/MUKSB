"""
Evaluation/imagenette/compute_clip_imagenette.py
=================================================
Compute CLIP image-text cosine similarity for generated Imagenette images.
Uses openai-clip (clip.load) — same loader as MUNBa/eval_scripts/CLIP/clip_sim.py.

For each class:
    CLIP score = mean cosine_sim(image_embed, text_embed)
    averaged over all generated images in that class folder.

Reports:
  - Per-class CLIP score (all 10 classes)
  - Retain-class average
  - Forget-class score (should drop after unlearning)

Usage
-----
    python Evaluation/imagenette/compute_clip_imagenette.py \\
        --gen_root        generated/my-model \\
        --class_to_forget 0 \\
        --device          0
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import clip
except ImportError:
    raise SystemExit(
        "[ERROR] openai-clip not found.\n"
        "Install:  pip install openai-clip"
    )

IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]
PROMPTS = [f"an image of a {c}" for c in IMAGENETTE_CLASSES]


@torch.no_grad()
def _encode_images(paths: list, model, preprocess, device: str, batch_size: int = 64):
    """Return per-image L2-normalised embeddings, shape (N, 512)."""
    all_feats = []
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i: i + batch_size]:
            try:
                batch.append(preprocess(Image.open(p).convert("RGB")))
            except Exception as e:
                print(f"  [WARN] {p}: {e}")
        if not batch:
            continue
        tensor = torch.stack(batch).to(device)
        feats  = model.encode_image(tensor).float()
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 512))


@torch.no_grad()
def _encode_text(prompts: list, model, device: str):
    """Return L2-normalised text embeddings, shape (N, 512)."""
    tokens = clip.tokenize(prompts).to(device)
    feats  = model.encode_text(tokens).float()
    feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def compute_clip_scores(gen_root: str, class_to_forget: int, device: str):
    print(f"[CLIP] Loading ViT-B/32 on cuda:{device.split(':')[-1]} …")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # encode all 10 class prompts at once
    text_embs = _encode_text(PROMPTS, model, device)   # (10, 512)

    per_class = {}

    for cls_idx, cls_name in enumerate(IMAGENETTE_CLASSES):
        safe_name = cls_name.replace(" ", "_")
        cls_dir   = os.path.join(gen_root, f"{cls_idx:02d}_{safe_name}")

        if not os.path.isdir(cls_dir):
            for entry in sorted(os.listdir(gen_root)):
                if entry.startswith(f"{cls_idx:02d}_"):
                    cls_dir = os.path.join(gen_root, entry)
                    break

        if not os.path.isdir(cls_dir):
            print(f"  [CLIP] class {cls_idx} ({cls_name}): dir not found, skipping.")
            continue

        img_paths = sorted([
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not img_paths:
            continue

        img_embs  = _encode_images(img_paths, model, preprocess, device)  # (N, 512)
        text_emb  = text_embs[cls_idx]                                     # (512,)

        # cosine similarity: image_embs already L2-normalised, text_emb too
        cos_sims  = img_embs @ text_emb                                    # (N,)
        avg_score = float(cos_sims.mean())

        per_class[cls_name] = {
            "clip_score": round(avg_score, 4),
            "n_images":   len(img_paths),
            "is_forget":  cls_idx == class_to_forget,
        }

    # ── summary ───────────────────────────────────────────────────────────────
    retain_scores = [v["clip_score"] for v in per_class.values() if not v["is_forget"]]
    forget_scores = [v["clip_score"] for v in per_class.values() if v["is_forget"]]

    retain_avg = round(float(np.mean(retain_scores)), 4) if retain_scores else None
    forget_avg = round(float(np.mean(forget_scores)), 4) if forget_scores else None

    print(f"\n{'='*55}")
    print(f"  Forget class    : {IMAGENETTE_CLASSES[class_to_forget]} (idx={class_to_forget})")
    print(f"  Forget CLIP     : {forget_avg}")
    print(f"  Retain CLIP avg : {retain_avg}")
    print(f"{'='*55}")
    print("\n  Per-class CLIP scores:")
    for cls_name_key, info in sorted(per_class.items(),
                                     key=lambda x: IMAGENETTE_CLASSES.index(x[0])):
        tag = " ← FORGET" if info["is_forget"] else ""
        print(f"    {cls_name_key:<22}  {info['clip_score']:.4f}  (n={info['n_images']}){tag}")

    results = {
        "class_to_forget": class_to_forget,
        "forget_class":    IMAGENETTE_CLASSES[class_to_forget],
        "forget_clip":     forget_avg,
        "retain_clip_avg": retain_avg,
        "per_class":       per_class,
    }

    out_json = os.path.join(gen_root, "clip_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[CLIP] Results saved to {out_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIP score for generated Imagenette images (openai-clip)"
    )
    parser.add_argument("--gen_root",        type=str, required=True)
    parser.add_argument("--class_to_forget", type=int, default=0)
    parser.add_argument("--device",          type=str, default="0")
    args = parser.parse_args()

    compute_clip_scores(
        gen_root        = args.gen_root,
        class_to_forget = args.class_to_forget,
        device          = f"cuda:{args.device}",
    )
