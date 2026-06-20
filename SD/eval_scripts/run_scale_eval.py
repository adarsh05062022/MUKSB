"""
run_scale_eval.py — End-to-end evaluation pipeline for the scale-ablation
checkpoints (ImageNette-10, SD v1.4).

Given a list of trained DIFFUSERS checkpoints (the `diffusers-cls_<C>-...-epoch_N.pt`
files), it runs, IN ORDER, for each checkpoint:

  1. generate_images.py        → 50 images / class (10 classes) into a clean
                                  per-checkpoint folder
  2. compute_fid.py            → retain-set FID (forget class excluded)
  3. imageclassify.py          → ResNet-50 top-k classification CSV
  4. compute_ua_imagenette_csv → UA (forget) + RA (retain avg) from the CSV

All metrics (FID, UA top-1, UA top-5, RA, per-class) are written to a single
consolidated JSON, plus a per-checkpoint JSON.

The forget class and the scale variant are parsed from the checkpoint filename
(`cls_<C>` and `MUKSB_scale_<variant>`); override the class with `path:CLASS`.

Usage (run from MUKSB/SD/, env = munba3)
-----
  python eval_scripts/run_scale_eval.py \
      --device cuda:2 \
      --checkpoints \
        models/compvis-cls_0-MUKSB_scale_arithmetic-.../diffusers-...-epoch_5.pt \
        models/compvis-cls_0-MUKSB_scale_min-.../diffusers-...-epoch_5.pt \
        models/compvis-cls_0-MUKSB_scale_fixed-.../diffusers-...-epoch_5.pt \
        models/compvis-cls_2-MUKSB_scale_arithmetic-.../diffusers-...-epoch_5.pt \
        models/compvis-cls_2-MUKSB_scale_min-.../diffusers-...-epoch_5.pt \
        models/compvis-cls_2-MUKSB_scale_fixed-.../diffusers-...-epoch_5.pt

  # force the forget class for a checkpoint with `path:CLASS`
  --checkpoints ".../diffusers-...epoch_5.pt:0"
"""
import argparse
import gc
import json
import os
import re
import sys
import time
import traceback

import pandas as pd
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from generate_images import generate_images
from compute_fid import compute_fid
from imageclassify import build_classifier, classify_folder
from compute_ua_imagenette_csv import (
    IMAGENETTE_CLASSES,
    compute_ua_ra_from_csvs,
)

_CLS_RE     = re.compile(r"cls_(\d+)")
_VARIANT_RE = re.compile(r"MUKSB_scale_([A-Za-z0-9]+)")


def parse_forget_class(ckpt_path):
    m = _CLS_RE.search(os.path.basename(ckpt_path))
    return int(m.group(1)) if m else None


def parse_variant(ckpt_path):
    m = _VARIANT_RE.search(os.path.basename(ckpt_path))
    return m.group(1) if m else "unknown"


def split_per_class_csvs(merged_csv, out_dir):
    """Split the merged classification CSV into one `<class>_classification.csv`
    per ImageNette class, as expected by compute_ua_ra_from_csvs."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(merged_csv)
    if "class" not in df.columns:
        raise KeyError(
            f"'class' column missing in {merged_csv}; "
            "cannot split per-class for UA/RA."
        )
    written = []
    for cls_name, sub in df.groupby("class"):
        safe = str(cls_name).strip().replace(" ", "_")
        out = os.path.join(out_dir, f"{safe}_classification.csv")
        sub.to_csv(out, index=False)
        written.append(out)
    return written


def evaluate_checkpoint(
    ckpt_path,
    forget_class,
    args,
    classifier,
    weights,
):
    variant   = parse_variant(ckpt_path)
    base       = os.path.basename(ckpt_path)
    gen_folder = os.path.join(args.gen_root, base)

    record = {
        "checkpoint": ckpt_path,
        "scale_variant": variant,
        "forget_class_idx": forget_class,
        "forget_class": IMAGENETTE_CLASSES[forget_class],
        "fid": None,
        "ua_top1": None,
        f"ua_top{args.topk}": None,
        "ra": None,
        "classification_csv": None,
        "ua_ra_json": None,
        "per_class": None,
        "status": "ok",
        "errors": [],
    }

    print("\n" + "=" * 72)
    print(f"  CKPT     : {ckpt_path}")
    print(f"  Variant  : {variant}   |   Forget class : {forget_class} "
          f"({IMAGENETTE_CLASSES[forget_class]})")
    print("=" * 72)

    # ── checkpoint must exist before anything else ───────────────────────────
    if not os.path.exists(ckpt_path):
        msg = f"checkpoint not found: {ckpt_path}"
        print(f"  [SKIP] {msg}")
        record["status"] = "failed"
        record["errors"].append(msg)
        _save_record(record, base, args)
        return record

    # ── 1. generate 50 imgs / class (mandatory — without images nothing else
    #       can run, so a failure here aborts only THIS checkpoint) ───────────
    try:
        expected = 10 * args.num_samples  # imgs per class (case)
        already = 0
        if os.path.isdir(gen_folder):
            already = len([f for f in os.listdir(gen_folder) if f.endswith(".png")])
        if args.skip_existing and already >= expected * 10:
            print(f"[1/4] generate — SKIP ({already} pngs already in {gen_folder})")
        else:
            print(f"[1/4] generate → {gen_folder}")
            generate_images(
                model_name=ckpt_path,
                prompts_path=args.prompts_path,
                save_path=args.gen_root,
                device=args.device,
                guidance_scale=args.guidance_scale,
                image_size=args.image_size,
                ddim_steps=args.ddim_steps,
                num_samples=args.num_samples,
                from_case=0,
            )
    except Exception as e:
        traceback.print_exc()
        print(f"  [FAIL] generation failed: {e}")
        record["status"] = "failed"
        record["errors"].append(f"generate: {e}")
        _save_record(record, base, args)
        gc.collect(); torch.cuda.empty_cache()
        return record

    # ── 2. retain-set FID (best effort) ──────────────────────────────────────
    try:
        print(f"[2/4] FID (retain, forget class {forget_class} excluded)")
        record["fid"] = round(float(compute_fid([forget_class], gen_folder, args.image_size)), 4)
    except Exception as e:
        traceback.print_exc()
        print(f"  [WARN] FID failed: {e}")
        record["status"] = "partial"
        record["errors"].append(f"fid: {e}")

    # ── 3. classification CSV (needed for UA/RA) ─────────────────────────────
    merged_csv = None
    try:
        print(f"[3/4] classify → {args.csv_root}")
        merged_csv = classify_folder(
            folder=gen_folder,
            prompts_path=args.prompts_path,
            save_path=args.csv_root,
            device=args.device,
            topk=args.topk,
            batch_size=args.batch_size,
            model=classifier,
            weights=weights,
        )
        record["classification_csv"] = merged_csv
    except Exception as e:
        traceback.print_exc()
        print(f"  [WARN] classification failed: {e}")
        record["status"] = "partial"
        record["errors"].append(f"classify: {e}")

    # ── 4. UA + RA (best effort; needs the classification CSV) ───────────────
    if merged_csv is not None:
        try:
            print(f"[4/4] UA / RA")
            per_class_dir = os.path.join(args.uacsv_root, base)
            split_per_class_csvs(merged_csv, per_class_dir)
            ua_json = os.path.join(args.results_dir, f"{base}_ua_ra.json")
            ua_res = compute_ua_ra_from_csvs(
                csv_dir=per_class_dir,
                class_to_forget=forget_class,
                topk=args.topk,
                output_json=ua_json,
            )
            record["ua_top1"]            = ua_res.get("ua_top1")
            record[f"ua_top{args.topk}"] = ua_res.get(f"ua_top{args.topk}")
            record["ra"]                 = ua_res.get("ra")
            record["ua_ra_json"]         = ua_json
            record["per_class"]          = ua_res.get("per_class")
        except Exception as e:
            traceback.print_exc()
            print(f"  [WARN] UA/RA failed: {e}")
            record["status"] = "partial"
            record["errors"].append(f"ua_ra: {e}")
    else:
        record["status"] = "partial"
        record["errors"].append("ua_ra: skipped (no classification CSV)")

    _save_record(record, base, args)
    gc.collect()
    torch.cuda.empty_cache()
    return record


def _save_record(record, base, args):
    """Write the per-checkpoint consolidated JSON (best effort)."""
    try:
        per_ckpt_json = os.path.join(args.results_dir, f"{base}_summary.json")
        with open(per_ckpt_json, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  → {per_ckpt_json}  [status={record['status']}]")
    except Exception as e:
        print(f"  [WARN] could not write per-ckpt json: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end UA/RA/FID eval for scale-ablation checkpoints"
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="DIFFUSERS .pt checkpoints; append ':CLASS' to override forget class",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--prompts_path", type=str,
        default="/scratch/s25017/MUKSB/SD/prompts/imagenette.csv",
    )
    parser.add_argument(
        "--eval_root", type=str,
        default="/scratch/s25017/MUKSB/SD/eval_scripts/scale_ablation",
        help="root for generated images, csvs and result jsons",
    )
    parser.add_argument("--num_samples", type=int, default=5,
                        help="samples/prompt; final imgs/class = 10*num_samples")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip generation if the per-ckpt folder is full")
    parser.add_argument("--summary_json", type=str, default=None,
                        help="path for the combined summary (default eval_root/...)")
    args = parser.parse_args()

    # ── derived output dirs ──────────────────────────────────────────────────
    args.gen_root    = os.path.join(args.eval_root, "generated")
    args.csv_root    = os.path.join(args.eval_root, "classification")
    args.uacsv_root  = os.path.join(args.eval_root, "ua_csvs")
    args.results_dir = os.path.join(args.eval_root, "results")
    for d in (args.gen_root, args.csv_root, args.uacsv_root, args.results_dir):
        os.makedirs(d, exist_ok=True)
    if args.summary_json is None:
        args.summary_json = os.path.join(args.results_dir, "scale_eval_summary.json")

    # ── parse checkpoints (+ optional :CLASS override) ───────────────────────
    # Parse errors do NOT abort the run — they are recorded as failed entries
    # and we move on to the next checkpoint.
    parsed = []          # (path, forget) ready to evaluate
    parse_failures = []  # records for entries we could not even parse
    for entry in args.checkpoints:
        try:
            if ":" in entry and entry.rsplit(":", 1)[1].isdigit():
                path, cls = entry.rsplit(":", 1)
                forget = int(cls)
            else:
                path = entry
                forget = parse_forget_class(path)
            if forget is None:
                raise ValueError(
                    f"could not parse forget class from '{path}'; "
                    f"append ':CLASS' (e.g. '{path}:0')"
                )
            parsed.append((path, forget))
        except Exception as e:
            print(f"  [SKIP] {entry}: {e}")
            parse_failures.append({
                "checkpoint": entry,
                "status": "failed",
                "errors": [f"parse: {e}"],
            })

    print(f"Checkpoints to evaluate: {len(parsed)} "
          f"(+{len(parse_failures)} unparseable)")
    for p, c in parsed:
        print(f"  [{parse_variant(p):>10} | cls {c}] {p}")

    classifier, weights = build_classifier(args.device)

    summary = list(parse_failures)
    t0 = time.time()
    for path, forget in parsed:
        try:
            rec = evaluate_checkpoint(path, forget, args, classifier, weights)
        except Exception as e:
            # final safety net — one checkpoint must never kill the rest
            traceback.print_exc()
            print(f"  [FAIL] unhandled error on {path}: {e}")
            rec = {
                "checkpoint": path,
                "scale_variant": parse_variant(path),
                "forget_class_idx": forget,
                "status": "failed",
                "errors": [f"unhandled: {e}"],
            }
        summary.append(rec)
        # write/refresh combined summary after every checkpoint (crash-safe)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)

    dt = time.time() - t0
    n_ok      = sum(1 for r in summary if r.get("status") == "ok")
    n_partial = sum(1 for r in summary if r.get("status") == "partial")
    n_failed  = sum(1 for r in summary if r.get("status") == "failed")
    print("\n" + "=" * 72)
    print(f"DONE — {len(summary)} checkpoint(s) in {dt/60:.1f} min  "
          f"(ok={n_ok}  partial={n_partial}  failed={n_failed})")
    print(f"Combined summary: {args.summary_json}")
    print("=" * 72)

    def _fmt(v, w, prec=None):
        if v is None:
            return f"{'—':>{w}}"
        if prec is not None:
            try:
                return f"{float(v):>{w}.{prec}f}"
            except (TypeError, ValueError):
                pass
        return f"{str(v):>{w}}"

    print(f"{'variant':>12} {'cls':>4} {'FID':>8} {'UA1':>7} {'UA5':>7} "
          f"{'RA':>7}  status")
    for r in summary:
        print(f"{_fmt(r.get('scale_variant'),12)} "
              f"{_fmt(r.get('forget_class_idx'),4)} "
              f"{_fmt(r.get('fid'),8,3)} "
              f"{_fmt(r.get('ua_top1'),7)} "
              f"{_fmt(r.get('ua_top'+str(args.topk)),7)} "
              f"{_fmt(r.get('ra'),7)}  {r.get('status','?')}")
    if n_failed or n_partial:
        print("\nIssues:")
        for r in summary:
            if r.get("status") in ("failed", "partial"):
                print(f"  [{r.get('status')}] "
                      f"{os.path.basename(str(r.get('checkpoint')))}: "
                      f"{'; '.join(r.get('errors', []))}")


if __name__ == "__main__":
    main()
