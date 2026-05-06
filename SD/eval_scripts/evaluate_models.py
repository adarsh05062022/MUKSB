"""
evaluate_models.py — Generate images and compute FID for a list of checkpoints.
Supports single or multi-class forgetting via --class_to_forget.
"""

import argparse
import gc
import json
import os
from datetime import datetime
from typing import List, Union

import torch

from compute_fid import compute_fid
from generate_images import generate_images

ClassSpec = Union[int, List[int]]


def evaluate_checkpoint(
    checkpoint_path: str,
    prompts_path: str,
    save_root: str,
    class_to_forget: ClassSpec,
    image_size: int,
    num_samples: int,
    device: str,
    guidance_scale: float,
    ddim_steps: int,
) -> dict:
    """
    Generate images from one checkpoint and compute FID.

    Returns a result dict with model name, FID, and metadata.
    """
    model_name = os.path.basename(checkpoint_path)
    print(f"\n===== Evaluating: {model_name} =====")

    # ── 1. Generate images ──
    generate_images(
        model_name=checkpoint_path,
        prompts_path=prompts_path,
        save_path=save_root,
        device=device,
        guidance_scale=guidance_scale,
        image_size=image_size,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        from_case=0,
    )

    generated_folder = os.path.join(save_root, model_name)

    # ── 2. Compute FID ──
    print("Computing FID...")
    fid_value = compute_fid(
        class_to_forget=class_to_forget,
        path=generated_folder,
        image_size=image_size,
    )

    result = {
        "model_name":      model_name,
        "class_to_forget": class_to_forget if isinstance(class_to_forget, list) else [class_to_forget],
        "num_samples":     num_samples,
        "fid":             float(fid_value),
        "last_execution":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    del fid_value
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(args):
    # Normalise class_to_forget: keep as list if multi, unwrap if single
    class_to_forget = (
        args.class_to_forget
        if len(args.class_to_forget) > 1
        else args.class_to_forget[0]
    )

    with open(args.checkpoints_txt, "r") as f:
        checkpoints = [line.strip() for line in f if line.strip()]

    for ckpt in checkpoints:
        result = evaluate_checkpoint(
            checkpoint_path=ckpt,
            prompts_path=args.prompts_path,
            save_root=args.save_path,
            class_to_forget=class_to_forget,
            image_size=args.image_size,
            num_samples=args.num_samples,
            device=args.device,
            guidance_scale=args.guidance_scale,
            ddim_steps=args.ddim_steps,
        )

        # Load, append, and save results incrementally
        existing = []
        if os.path.exists(args.output_json):
            with open(args.output_json, "r") as f:
                existing = json.load(f)

        existing.append(result)

        with open(args.output_json, "w") as f:
            json.dump(existing, f, indent=4)

        print(f"Saved result for {result['model_name']} → FID: {result['fid']:.4f}")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print(f"\nAll results saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate unlearned checkpoints via FID")

    parser.add_argument("--checkpoints_txt", type=str, required=False,
                        default="/storage/s25017/MUKSB/SD/eval_scripts/CLASS/checkpoints_to_evaluate.txt")
    parser.add_argument("--prompts_path", type=str, required=False,
                        default="/storage/s25017/MUKSB/SD/prompts/imagenette.csv")
    parser.add_argument("--save_path", type=str, required=False,
                        default="/storage/s25017/MUKSB/SD/eval_scripts/CLASS/generated")
    parser.add_argument("--output_json", type=str, default="CLASS/results.json")

    # Multi-class: --class_to_forget 0 7   OR   --class_to_forget 0
    parser.add_argument("--class_to_forget", type=int, nargs="+", default=[7],
                        help="One or more class indices to forget")

    parser.add_argument("--image_size",     type=int,   default=512)
    parser.add_argument("--num_samples",    type=int,   default=10)
    parser.add_argument("--device",         type=str,   default="cuda:2")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--ddim_steps",     type=int,   default=50)

    args = parser.parse_args()
    main(args)