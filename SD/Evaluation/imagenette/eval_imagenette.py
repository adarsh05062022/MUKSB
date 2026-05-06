"""
Evaluation/imagenette/eval_imagenette.py
=========================================
Single-command orchestrator for the full Imagenette-10 evaluation pipeline:

    Step 1 — Generate 500 images per class  (generate_imagenette.py)
    Step 2 — Compute FID on retain classes  (compute_fid_imagenette.py)
    Step 3 — Compute UA + RA               (compute_ua_imagenette.py)
    Step 4 — Compute CLIP scores           (compute_clip_imagenette.py)
    Step 5 — Aggregate results → JSON + table

UA/RA uses ResNet50 (ImageNet weights) — no external classifier needed.
Any individual step can be skipped with --skip_generate / --skip_fid /
--skip_ua / --skip_clip.

Usage
-----
    # Full pipeline (one command):
    python Evaluation/imagenette/eval_imagenette.py \\
        --model_path      models/SSU-cls_0-.../SSU-cls_0-...-epoch_5.pt \\
        --class_to_forget 0 \\
        --device          0

    # Only metrics on already-generated images:
    python Evaluation/imagenette/eval_imagenette.py \\
        --model_path      models/... \\
        --class_to_forget 0 \\
        --device          0 \\
        --skip_generate
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SD_DIR   = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _SD_DIR not in sys.path:
    sys.path.insert(0, _SD_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from generate_imagenette         import generate_all_classes
from generate_imagenette_multigpu import launch_multigpu as generate_all_classes_multigpu
from compute_fid_imagenette      import compute_fid
from compute_ua_imagenette       import compute_ua_ra
from compute_clip_imagenette     import compute_clip_scores


IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]


def run_eval(
    model_path, output_dir, class_to_forget, devices,
    n_per_class, guidance_scale, image_size, ddim_steps,
    skip_generate, skip_fid, skip_ua, skip_clip,
    topk=5, batch_size=250,
):
    # Primary device for FID / UA / CLIP (always single GPU)
    device_str = f"cuda:{devices[0]}"

    # ── derive model tag and gen_root ─────────────────────────────────────────
    model_tag = (os.path.basename(model_path).replace(".pt", "")
                 if model_path else "sd14_baseline")
    gen_root  = os.path.join(output_dir, model_tag)
    results   = {
        "model_tag":       model_tag,
        "model_path":      model_path,
        "class_to_forget": class_to_forget,
        "forget_class":    IMAGENETTE_CLASSES[class_to_forget],
        "gen_root":        gen_root,
    }

    # ── Step 1: Generate ─────────────────────────────────────────────────────
    if not skip_generate:
        print(f"\n{'#'*60}")
        if len(devices) > 1:
            print(f"# Step 1/4 — Generate images ({n_per_class}/class) on {len(devices)} GPUs: {devices}")
        else:
            print(f"# Step 1/4 — Generate images ({n_per_class} per class) on GPU {devices[0]}")
        print(f"{'#'*60}")
        if len(devices) > 1:
            generate_all_classes_multigpu(
                model_path     = model_path,
                output_dir     = output_dir,
                n_per_class    = n_per_class,
                gpu_ids        = devices,
                guidance_scale = guidance_scale,
                image_size     = image_size,
                ddim_steps     = ddim_steps,
            )
        else:
            generate_all_classes(
                model_path     = model_path,
                output_dir     = output_dir,
                device         = device_str,
                n_per_class    = n_per_class,
                guidance_scale = guidance_scale,
                image_size     = image_size,
                ddim_steps     = ddim_steps,
            )
    else:
        print(f"\n[SKIP] Image generation (--skip_generate). Using: {gen_root}")

    # ── Step 2: FID ───────────────────────────────────────────────────────────
    if not skip_fid:
        print(f"\n{'#'*60}")
        print(f"# Step 2/4 — FID on retain classes")
        print(f"{'#'*60}")
        fid_res = compute_fid(
            gen_root        = gen_root,
            class_to_forget = class_to_forget,
            device          = device_str,
        )
        results.update({k: fid_res.get(k) for k in ("retain_fid",)})
    else:
        print("\n[SKIP] FID computation (--skip_fid).")

    # ── Step 3: UA / RA ───────────────────────────────────────────────────────
    if not skip_ua:
        print(f"\n{'#'*60}")
        print(f"# Step 3/4 — UA + RA  (ResNet50 / ImageNet weights)")
        print(f"{'#'*60}")
        ua_res = compute_ua_ra(
            gen_root        = gen_root,
            class_to_forget = class_to_forget,
            device          = device_str,
            topk            = topk,
            batch_size      = batch_size,
        )
        results.update({k: ua_res.get(k) for k in ("ua_top1", f"ua_top{topk}", "ra")})
    else:
        print("\n[SKIP] UA/RA computation (--skip_ua).")

    # ── Step 4: CLIP ──────────────────────────────────────────────────────────
    if not skip_clip:
        print(f"\n{'#'*60}")
        print(f"# Step 4/4 — CLIP scores")
        print(f"{'#'*60}")
        clip_res = compute_clip_scores(
            gen_root        = gen_root,
            class_to_forget = class_to_forget,
            device          = device_str,
        )
        results.update({
            "forget_clip":     clip_res.get("forget_clip"),
            "retain_clip_avg": clip_res.get("retain_clip_avg"),
        })
    else:
        print("\n[SKIP] CLIP computation (--skip_clip).")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  IMAGENETTE-10 EVALUATION SUMMARY")
    print(f"  Model          : {model_tag}")
    print(f"  Forget class   : {IMAGENETTE_CLASSES[class_to_forget]} (idx={class_to_forget})")
    print(f"  Retain FID     : {results.get('retain_fid', 'N/A')}")
    print(f"  UA  (top-1)    : {results.get('ua_top1', 'N/A')}%")
    print(f"  UA  (top-{topk})    : {results.get(f'ua_top{topk}', 'N/A')}%")
    print(f"  RA             : {results.get('ra', 'N/A')}%")
    print(f"  Forget CLIP    : {results.get('forget_clip', 'N/A')}")
    print(f"  Retain CLIP    : {results.get('retain_clip_avg', 'N/A')}")
    print(f"{'='*60}\n")

    summary_path = os.path.join(gen_root, "eval_summary.json")
    os.makedirs(gen_root, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Summary] Saved to {summary_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full Imagenette-10 evaluation pipeline (generate → FID → UA → CLIP)"
    )
    # required-ish
    parser.add_argument("--model_path",      type=str, default="/storage/s25017/MUKSB/SD/models/compvis-cls_0-MUKSB-g0.5-method_full-lr_5e-06_E10_U963_/diffusers-cls_0-MUKSB-g0.5-method_full-lr_5e-06_E10_U963_-epoch_2.pt")
    parser.add_argument("--class_to_forget", type=int, default=0)
    # output
    parser.add_argument("--output_dir",      type=str,
                        default="/storage/s25017/MUKSB/SD/Evaluation/imagenette/different_l2")
    parser.add_argument("--device",          type=int, nargs="+", default=[4,5,6],
                        help="GPU id(s) to use. Single value → one GPU. "
                             "Multiple values → multi-GPU generation, e.g. --device 0 1 2 3")
    # generation
    parser.add_argument("--n_per_class",     type=int, default=10)
    parser.add_argument("--guidance_scale",  type=float, default=7.5)
    parser.add_argument("--image_size",      type=int, default=512)
    parser.add_argument("--ddim_steps",      type=int, default=50)
    parser.add_argument("--topk",            type=int, default=5,
                        help="Top-k for UA classification (default: 5)")
    parser.add_argument("--batch_size",      type=int, default=250,
                        help="Batch size for ResNet50 classification")
    # skip flags
    parser.add_argument("--skip_generate",   action="store_true", default=False)
    parser.add_argument("--skip_fid",        action="store_true", default=False)
    parser.add_argument("--skip_ua",         action="store_true", default=False)
    parser.add_argument("--skip_clip",       action="store_true", default=True)

    args = parser.parse_args()

    run_eval(
        model_path      = args.model_path,
        output_dir      = args.output_dir,
        class_to_forget = args.class_to_forget,
        devices         = args.device,
        n_per_class     = args.n_per_class,
        guidance_scale  = args.guidance_scale,
        image_size      = args.image_size,
        ddim_steps      = args.ddim_steps,
        topk            = args.topk,
        batch_size      = args.batch_size,
        skip_generate   = args.skip_generate,
        skip_fid        = args.skip_fid,
        skip_ua         = args.skip_ua,
        skip_clip       = args.skip_clip,
    )
