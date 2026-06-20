#!/usr/bin/env python
"""
IP2P/run_forget_sweep.py
========================
Strengthen NSFW forgetting: train_method=full (whole UNet), no mask,
at several epoch counts. After each run it auto-generates a viz contact
sheet for the FINAL epoch so you can eyeball forget + retain together.

Each --epochs value is an INDEPENDENT run trained from the base model
(the 10-epoch run is NOT a continuation of the 5-epoch run).

Run it with the env that has `diffusers` (munba3); sys.executable is reused
for the child processes, so just call it with that interpreter:

    python run_forget_sweep.py --device 3
    python run_forget_sweep.py --device 3 --epochs 5 10 --beta 5
    python run_forget_sweep.py --device 3 --batch_size 2          # if OOM
    python run_forget_sweep.py --device 3 --train_method xattn --beta 10
    python run_forget_sweep.py --device 3 --epochs 10 --skip_viz
"""

import argparse
import glob
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def train(py, args, epochs):
    cmd = [
        py, "MUKSB_nsfw_i2i.py",
        "--mask_variant", "none",
        "--train_method", args.train_method,
        "--beta",         str(args.beta),
        "--epochs",       str(epochs),
        "--lr",           args.lr,
        "--batch_size",   str(args.batch_size),
        "--ckpt_path",    args.ckpt_path,
        "--forget_path",  args.forget_path,
        "--remain_path",  args.remain_path,
        "--device",       str(args.device),
    ]
    print("=" * 60)
    print(f" TRAIN  method={args.train_method}  epochs={epochs}  "
          f"beta={args.beta}  device={args.device}")
    print("=" * 60, flush=True)
    subprocess.run(cmd, cwd=THIS_DIR, check=True)


def find_final_ckpt(train_method, epochs):
    pattern = os.path.join(
        THIS_DIR, "models",
        f"i2p-nsfw-MUKSB-i2i-method_{train_method}-*E{epochs}_U*",
        f"epoch_{epochs}",
    )
    hits = sorted(glob.glob(pattern))
    for h in hits:
        if os.path.isfile(os.path.join(h, "model_index.json")):
            return h
    return None


def viz(py, args, epochs):
    ckpt = find_final_ckpt(args.train_method, epochs)
    if ckpt is None:
        print(f"WARN: final checkpoint for E{epochs} not found "
              f"(looked for epoch_{epochs}); skipping viz", flush=True)
        return
    out_dir = f"smoke_results/viz_{args.train_method}_E{epochs}"
    cmd = [
        py, "viz_check.py",
        "--model_path", ckpt,
        "--src_dir",    args.viz_src,
        "--device",     str(args.device),
        "--n_sources",  str(args.n_sources),
        "--out_dir",    out_dir,
    ]
    print("=" * 60)
    print(f" VIZ    {ckpt}")
    print("=" * 60, flush=True)
    subprocess.run(cmd, cwd=THIS_DIR, check=True)
    print(f" contact sheet -> {out_dir}/contact_sheet.png", flush=True)


def main():
    ap = argparse.ArgumentParser(description="MUKSB I2I forget-strength sweep")
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, nargs="+", default=[5, 10],
                    help="One independent run per value (default: 5 10).")
    ap.add_argument("--train_method", default="full",
                    choices=["full", "noxattn", "xattn", "selfattn",
                             "notime", "xlayer", "selflayer"])
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", default="1e-5")
    ap.add_argument("--ckpt_path", default="timbrooks/instruct-pix2pix")
    ap.add_argument("--forget_path",
                    default="/storage/s25017/Datasets/NSFW_removal/nude")
    ap.add_argument("--remain_path",
                    default="/storage/s25017/Datasets/NSFW_removal/with_dress")
    ap.add_argument("--viz_src",
                    default="/storage/s25017/Datasets/NSFW_removal/with_dress")
    ap.add_argument("--n_sources", type=int, default=4)
    ap.add_argument("--skip_viz", action="store_true")
    args = ap.parse_args()

    py = sys.executable  # reuse the interpreter running this script

    for ep in args.epochs:
        train(py, args, ep)
        if not args.skip_viz:
            viz(py, args, ep)

    print("Sweep complete.")


if __name__ == "__main__":
    main()
