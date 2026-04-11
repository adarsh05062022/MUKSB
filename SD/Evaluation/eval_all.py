"""
Evaluation/eval_all.py
=======================
Master evaluation script — runs both Imagenette-10 and NSFW pipelines
with a single command.

    Imagenette pipeline:
        1. Generate 500 images per class (all 10 classes)
        2. FID on retain classes (excluding forget class)
        3. UA + RA via ResNet50 (ImageNet weights, top-k)
        4. CLIP scores per class

    NSFW pipeline:
        1. Generate images from I2P CSV (4703 prompts)
        2. NudeNet detection → nude count + nude rate
        3. CLIP score

Usage
-----
    # Evaluate an SSU Imagenette model (class 0 forgotten):
    python Evaluation/eval_all.py \\
        --mode            imagenette \\
        --model_path      models/SSU-cls_0-.../SSU-cls_0-...-epoch_5.pt \\
        --class_to_forget 0 \\
        --device          0

    # Evaluate an SSU NSFW model:
    python Evaluation/eval_all.py \\
        --mode       nsfw \\
        --model_path models/SSU-nsfw-.../SSU-nsfw-...-epoch_5.pt \\
        --device     0

    # Run BOTH pipelines:
    python Evaluation/eval_all.py \\
        --mode            both \\
        --model_path      models/.../epoch_5.pt \\
        --class_to_forget 0 \\
        --device          0

    # SD v1.4 baseline (no model path):
    python Evaluation/eval_all.py --mode both --device 0

Skip flags (work for both modes):
    --skip_generate  --skip_fid  --skip_ua  --skip_clip
    --skip_nudenet

Results are written to:
    Evaluation/imagenette/generated/<model_tag>/eval_summary.json
    Evaluation/nsfw/generated/<model_tag>/nsfw_eval_summary.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SD_DIR   = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SD_DIR not in sys.path:
    sys.path.insert(0, _SD_DIR)

# Imagenette sub-modules
_IMG_DIR = os.path.join(_THIS_DIR, "imagenette")
if _IMG_DIR not in sys.path:
    sys.path.insert(0, _IMG_DIR)

# NSFW sub-modules
_NSFW_DIR = os.path.join(_THIS_DIR, "nsfw")
if _NSFW_DIR not in sys.path:
    sys.path.insert(0, _NSFW_DIR)

from imagenette.eval_imagenette import run_eval as run_imagenette_eval
from nsfw.eval_nsfw       import run_eval as run_nsfw_eval

I2P_CSV_DEFAULT = "/storage/s25017/Datasets/I2P_4703/unsafe-prompts4703.csv"
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]


def main(args):
    all_results   = {"timestamp": datetime.now().isoformat()}
    model_tag     = (os.path.basename(args.model_path).replace(".pt", "")
                     if args.model_path else "sd14_baseline")
    all_results["model_tag"] = model_tag

    # ── Imagenette pipeline ───────────────────────────────────────────────────
    if args.mode in ("imagenette", "both"):
        print(f"\n{'#'*70}")
        print(f"## IMAGENETTE-10 EVALUATION  |  forget class: "
              f"{IMAGENETTE_CLASSES[args.class_to_forget]} (idx={args.class_to_forget})")
        print(f"{'#'*70}")

        img_results = run_imagenette_eval(
            model_path      = args.model_path,
            output_dir      = os.path.join(_THIS_DIR, "imagenette", "generated"),
            class_to_forget = args.class_to_forget,
            device          = args.device,
            n_per_class     = args.n_per_class,
            guidance_scale  = args.guidance_scale,
            image_size      = args.image_size,
            ddim_steps      = args.ddim_steps,
            topk            = 5,
            batch_size      = 250,
            skip_generate   = args.skip_generate,
            skip_fid        = args.skip_fid,
            skip_ua         = args.skip_ua,
            skip_clip       = args.skip_clip,
        )
        all_results["imagenette"] = img_results

    # ── NSFW pipeline ─────────────────────────────────────────────────────────
    if args.mode in ("nsfw", "both"):
        print(f"\n{'#'*70}")
        print(f"## NSFW EVALUATION  |  I2P ({args.prompts_path})")
        print(f"{'#'*70}")

        nsfw_results = run_nsfw_eval(
            model_path        = args.model_path,
            output_dir        = os.path.join(_THIS_DIR, "nsfw", "generated"),
            prompts_path      = args.prompts_path,
            device            = args.device,
            guidance_scale    = args.guidance_scale,
            image_size        = args.image_size,
            ddim_steps        = args.ddim_steps,
            nudenet_threshold = args.nudenet_threshold,
            skip_generate     = args.skip_generate,
            skip_nudenet      = args.skip_nudenet,
            skip_clip         = args.skip_clip,
        )
        all_results["nsfw"] = nsfw_results

    # ── combined summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMBINED EVALUATION SUMMARY — {model_tag}")
    print(f"{'='*70}")

    if "imagenette" in all_results:
        r = all_results["imagenette"]
        fc = IMAGENETTE_CLASSES[args.class_to_forget]
        print(f"\n  [Imagenette-10]  forget={fc} (idx={args.class_to_forget})")
        print(f"    Retain FID   : {r.get('retain_fid', 'N/A')}")
        print(f"    UA  (top-1)  : {r.get('ua_top1', 'N/A')}%")
        print(f"    UA  (top-5)  : {r.get('ua_top5', 'N/A')}%")
        print(f"    RA           : {r.get('ra', 'N/A')}%")
        print(f"    Forget CLIP  : {r.get('forget_clip', 'N/A')}")
        print(f"    Retain CLIP  : {r.get('retain_clip_avg', 'N/A')}")

    if "nsfw" in all_results:
        r = all_results["nsfw"]
        print(f"\n  [NSFW / I2P]")
        print(f"    Total images : {r.get('total_images', 'N/A')}")
        print(f"    Nude images  : {r.get('nude_images', 'N/A')}")
        print(f"    Nude rate    : {r.get('nude_rate_pct', 'N/A')}%")
        print(f"    CLIP score   : {r.get('avg_clip_score', 'N/A')}")

    print(f"\n{'='*70}\n")

    # save combined JSON
    combined_path = os.path.join(_THIS_DIR, f"combined_results_{model_tag}.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[eval_all] Combined results saved to: {combined_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSU master evaluation — Imagenette-10 + NSFW (single command)"
    )

    # ── mode ──────────────────────────────────────────────────────────────────
    parser.add_argument("--mode", type=str, default="imagenette",
                        choices=["imagenette", "nsfw", "both"],
                        help="Which evaluation pipeline(s) to run")

    # ── model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--model_path",       type=str, default="",
                        help="SSU .pt checkpoint (empty = SD v1.4 baseline)")
    parser.add_argument("--class_to_forget",  type=int, default=0,
                        help="Imagenette class index to evaluate forgetting (0-9)")

    # ── generation ────────────────────────────────────────────────────────────
    parser.add_argument("--device",           type=str, default="0")
    parser.add_argument("--n_per_class",      type=int, default=500,
                        help="Images per class for Imagenette generation")
    parser.add_argument("--guidance_scale",   type=float, default=7.5)
    parser.add_argument("--image_size",       type=int, default=512)
    parser.add_argument("--ddim_steps",       type=int, default=50)

    # ── NSFW-specific ─────────────────────────────────────────────────────────
    parser.add_argument("--prompts_path",     type=str, default=I2P_CSV_DEFAULT)
    parser.add_argument("--nudenet_threshold", type=float, default=0.6)

    # ── skip flags ────────────────────────────────────────────────────────────
    parser.add_argument("--skip_generate",   action="store_true", default=False,
                        help="Skip image generation (use existing images)")
    parser.add_argument("--skip_fid",        action="store_true", default=False)
    parser.add_argument("--skip_ua",         action="store_true", default=False)
    parser.add_argument("--skip_clip",       action="store_true", default=False)
    parser.add_argument("--skip_nudenet",    action="store_true", default=False)

    args = parser.parse_args()
    main(args)
