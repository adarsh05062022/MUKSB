"""
SD/eval_scripts/evaluate_models.py — MUKSB
Evaluate unlearned SD models using:
  - CLIP similarity between generated images and text prompts
  - FID score (generated vs real Imagenette images)
  - Classifier accuracy on generated images

Usage
-----
  python evaluate_models.py \\
      --gen_dir eval_out/cls0 \\
      --class_idx 0 \\
      --real_dir /storage/s25017/Datasets/imagenette2-320/train \\
      --device 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SD_DIR   = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SD_DIR not in sys.path:
    sys.path.insert(0, _SD_DIR)

IMAGENETTE_DESCRIPTIONS = [
    "an image of a tench",
    "an image of a English springer",
    "an image of a cassette player",
    "an image of a chain saw",
    "an image of a church",
    "an image of a French horn",
    "an image of a garbage truck",
    "an image of a gas pump",
    "an image of a golf ball",
    "an image of a parachute",
]

IMAGENETTE_WNIDS = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLIP similarity
# ─────────────────────────────────────────────────────────────────────────────

def compute_clip_similarity(gen_dir, prompt, device):
    """Mean CLIP cosine similarity between generated images and text prompt."""
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
    except ImportError:
        print("CLIP not available, skipping CLIP similarity evaluation")
        return None

    img_files = sorted([os.path.join(gen_dir, f) for f in os.listdir(gen_dir)
                        if f.lower().endswith(".png")])
    if not img_files:
        print(f"No PNG files found in {gen_dir}")
        return None

    text_tokens = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    sims = []
    for f in tqdm(img_files, desc="CLIP sim"):
        img = preprocess(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims.append((img_feat @ text_feat.T).item())

    return float(np.mean(sims))


# ─────────────────────────────────────────────────────────────────────────────
# FID score
# ─────────────────────────────────────────────────────────────────────────────

def compute_fid(gen_dir, real_dir, device):
    """Compute FID between generated and real images using pytorch-fid."""
    try:
        from pytorch_fid import fid_score
        fid = fid_score.calculate_fid_given_paths(
            [gen_dir, real_dir],
            batch_size=50,
            device=device,
            dims=2048,
        )
        return float(fid)
    except ImportError:
        print("pytorch-fid not installed (pip install pytorch-fid), skipping FID")
        return None
    except Exception as e:
        print(f"FID computation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Classifier accuracy
# ─────────────────────────────────────────────────────────────────────────────

def compute_classifier_accuracy(gen_dir, class_idx, device):
    """
    Classify generated images with a pre-trained ResNet50 (ImageNet weights).
    Returns fraction of images correctly predicted as the target Imagenette class.
    """
    try:
        import torchvision.models as models
        import torchvision.transforms as T
        model = models.resnet50(weights="IMAGENET1K_V1").to(device)
        model.eval()
    except Exception as e:
        print(f"Could not load classifier: {e}")
        return None

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Map Imagenette class → ImageNet class index
    import json as _json
    wnid = IMAGENETTE_WNIDS[class_idx]
    # Use a fixed mapping (tench=0, springer=217, cassette=482, chainsaw=491,
    # church=497, horn=566, garbage truck=569, pump=571, golf=574, parachute=701)
    imagenet_cls = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701][class_idx]

    img_files = sorted([os.path.join(gen_dir, f) for f in os.listdir(gen_dir)
                        if f.lower().endswith(".png")])
    correct = 0
    for f in tqdm(img_files, desc="Classifier"):
        img = transform(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img).argmax(1).item()
        if pred == imagenet_cls:
            correct += 1

    return correct / max(len(img_files), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MUKSB: Evaluate unlearned SD model (CLIP sim, FID, classifier accuracy)"
    )
    parser.add_argument("--gen_dir",    type=str, required=True,
                        help="Directory with generated PNG images")
    parser.add_argument("--class_idx",  type=int, default=0,
                        help="Imagenette class index (0–9)")
    parser.add_argument("--real_dir",   type=str, default=None,
                        help="Real Imagenette class dir for FID (optional)")
    parser.add_argument("--out_json",   type=str, default=None,
                        help="JSON file to save results (default: <gen_dir>/eval.json)")
    parser.add_argument("--device",     type=str, default="0")
    args = parser.parse_args()

    device  = f"cuda:{args.device}"
    prompt  = IMAGENETTE_DESCRIPTIONS[args.class_idx]
    out_json = args.out_json or os.path.join(args.gen_dir, "eval.json")

    print(f"=== MUKSB Evaluation — class {args.class_idx}: {prompt} ===")
    results = {"class_idx": args.class_idx, "prompt": prompt, "gen_dir": args.gen_dir}

    clip_sim = compute_clip_similarity(args.gen_dir, prompt, device)
    print(f"CLIP similarity : {clip_sim}")
    results["clip_similarity"] = clip_sim

    if args.real_dir:
        fid = compute_fid(args.gen_dir, args.real_dir, device)
        print(f"FID             : {fid}")
        results["fid"] = fid

    acc = compute_classifier_accuracy(args.gen_dir, args.class_idx, device)
    print(f"Classifier acc  : {acc}")
    results["classifier_accuracy"] = acc

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_json}")


if __name__ == "__main__":
    main()
