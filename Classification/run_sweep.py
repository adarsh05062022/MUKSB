"""
run_sweep.py — Replicate Table 1 (CIFAR-10) from the MUNBa / MUKSB paper.

Runs every method under identical shared hyperparameters, collects:
    Acc_Df (↓)   Acc_Dr (↑)   Acc_Dt (↑)   MIA (↑)   Avg. Gap

Avg. Gap = mean |method_metric - retrain_metric| across 4 metrics
(same definition as the paper).

Paper setup (CIFAR-10):
    10% randomly selected samples forgotten  →  --num_indexes_to_replace 5000
    ResNet-18, seed=2, unlearn_epochs=10, unlearn_lr=0.01

Run command
-----------
    python run_sweep.py \\
        --mask /storage/s25017/MUKSB/Classification/checkpoints/resnet18_cifar10/0checkpoint.pth.tar \\
        --dataset cifar10 --arch resnet18 \\
        --num_indexes_to_replace 5000 \\
        --unlearn_epochs 10 --unlearn_lr 0.01 \\
        --gpu 0 \\
        --out_csv ./results/table1_cifar10.csv

Options
-------
    --methods  GA FT MUNBa MUKSB   # run only these methods
    --skip_done                    # resume: skip if eval_result.pth.tar exists
    --dry_run                      # print commands without running
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import torch


# ── Method registry ───────────────────────────────────────────────────────────
# Columns:  (sweep_name, --unlearn key, extra_flags, paper_label)
#
# Order matches Table 1: Retrain, FT, GA, IU, BE, BS, ℓ1-sparse, SalUn, SHs, MUNBa, MUKSB
METHODS = [
    ("retrain",   "retrain",            [],                                        "Retrain"),
    ("FT",        "FT",                 [],                                        "FT"),
    ("GA",        "GA",                 [],                                        "GA"),
    ("IU",        "IU",                 ["--iu_damping", "1e-3",
                                          "--iu_scale",   "1.0"],                  "IU"),
    ("BE",        "boundary_expanding", [],                                        "BE"),
    ("BS",        "boundary_shrink",    [],                                        "BS"),
    ("l1_sparse", "GA_l1",              ["--with_l1"],                             "ℓ1-sparse"),
    ("SalUn",     "SalUn",              ["--salun_density", "0.5"],                "SalUn"),
    ("SHs",       "SHs",                ["--sparsity", "0.9", "--lam", "0.1"],     "SHs"),
    ("MUNBa",     "MUNBa",              ["--beta", "1.0"],                         "MUNBa"),
    ("MUKSB",     "MUKSB",              ["--beta", "1.0"],                         "MUKSB"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def checkpoint_path(save_root, sweep_name):
    return os.path.join(save_root, sweep_name, "eval_result.pth.tar")


def load_eval_result(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [warn] Could not load {path}: {e}")
        return None


def parse_metrics(eval_result):
    """
    Flatten eval_result into scalar metrics dict.
    Keys: acc_forget, acc_retain, acc_val, acc_test, MIA,
          SVC_MIA_forget_efficacy, SVC_MIA_training_privacy
    """
    if eval_result is None:
        return {}
    metrics = {}
    acc = eval_result.get("accuracy") or eval_result.get("new_accuracy") or {}
    if isinstance(acc, dict):
        for split, val in acc.items():
            if val is not None:
                metrics[f"acc_{split}"] = round(float(val), 4)
    for key in ("MIA", "SVC_MIA_forget_efficacy", "SVC_MIA_training_privacy"):
        v = eval_result.get(key)
        if v is not None:
            metrics[key] = round(float(v), 4)
    return metrics


def compute_avg_gap(row, retrain_row):
    """
    Avg. Gap = mean( |method_metric - retrain_metric| )
    computed over: acc_forget, acc_retain, acc_test, MIA
    Matches Table 1 definition.
    """
    keys = ["acc_forget", "acc_retain", "acc_test", "MIA"]
    diffs = []
    for k in keys:
        if k in row and k in retrain_row:
            diffs.append(abs(float(row[k]) - float(retrain_row[k])))
    return round(sum(diffs) / len(diffs), 2) if diffs else ""


def run_method(sweep_name, unlearn_key, extra_flags, base_flags, save_root, dry_run=False):
    save_dir = os.path.join(save_root, sweep_name)
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable, "main_random.py",
        "--unlearn",  unlearn_key,
        "--save_dir", save_dir,
    ] + base_flags + extra_flags

    print(f"\n{'='*66}")
    print(f"  [{sweep_name}]  →  --unlearn {unlearn_key}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*66}")

    if dry_run:
        return True, 0.0

    t0     = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0
    success = result.returncode == 0
    print(f"  → {'OK' if success else 'FAILED (code ' + str(result.returncode) + ')'}  "
          f"({elapsed:.1f}s)")
    return success, elapsed


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Table-1 replication sweep — MUKSB / MUNBa (CIFAR-10)"
    )

    # ── required ──────────────────────────────────────────────────────────────
    p.add_argument("--mask", type=str, required=True,
                   help="Pretrained model checkpoint (.pth.tar)")

    # ── dataset / model ───────────────────────────────────────────────────────
    p.add_argument("--dataset",     type=str, default="cifar10")
    p.add_argument("--arch",        type=str, default="resnet18")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--data",        type=str,
                   default="/storage/s25017/Datasets/CIFAR10",
                   help="Path to CIFAR-10 data directory")

    # ── forget setup (paper: 10% random = 5000 samples) ──────────────────────
    p.add_argument("--class_to_replace",       type=int, default=-1,
                   help="-1 = random samples across all classes")
    p.add_argument("--num_indexes_to_replace", type=int, default=5000,
                   help="Number of samples to forget (CIFAR-10 10%% = 5000)")

    # ── hardware ──────────────────────────────────────────────────────────────
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--seed",        type=int, default=2)
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)

    # ── shared unlearn hyperparameters (FIXED across ALL methods) ─────────────
    p.add_argument("--unlearn_epochs",  type=int,   default=10)
    p.add_argument("--unlearn_lr",      type=float, default=0.01)
    p.add_argument("--momentum",        type=float, default=0.9)
    p.add_argument("--weight_decay",    type=float, default=5e-4)
    p.add_argument("--decreasing_lr",   type=str,   default="91,136")
    p.add_argument("--alpha",           type=float, default=0.2,
                   help="L1 scale for GA_l1 / FT_l1")

    # ── sweep control ─────────────────────────────────────────────────────────
    p.add_argument("--methods",    nargs="+", default=None,
                   help="Subset of sweep_names to run, e.g. --methods GA FT MUNBa MUKSB")
    p.add_argument("--skip_done",  action="store_true",
                   help="Skip methods whose eval_result.pth.tar already exists")
    p.add_argument("--dry_run",    action="store_true",
                   help="Print commands without running them")

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument("--save_root",  type=str, default="./sweep_checkpoints",
                   help="Root dir — each method gets a sub-folder")
    p.add_argument("--out_csv",    type=str, default="./results/table1_cifar10.csv")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── shared base flags passed to every main_forget.py call ─────────────────
    base_flags = [
        "--dataset",                args.dataset,
        "--arch",                   args.arch,
        "--num_classes",            str(args.num_classes),
        "--class_to_replace",       str(args.class_to_replace),
        "--num_indexes_to_replace", str(args.num_indexes_to_replace),
        "--gpu",                    str(args.gpu),
        "--seed",                   str(args.seed),
        "--batch_size",             str(args.batch_size),
        "--num_workers",            str(args.num_workers),
        "--data",                   args.data,
        "--unlearn_epochs",         str(args.unlearn_epochs),
        "--unlearn_lr",             str(args.unlearn_lr),
        "--momentum",               str(args.momentum),
        "--weight_decay",           str(args.weight_decay),
        "--decreasing_lr",          args.decreasing_lr,
        "--alpha",                  str(args.alpha),
        "--mask",                   args.mask,
        "--print_freq",             "50",
    ]

    # ── filter to requested subset ────────────────────────────────────────────
    run_list = METHODS
    if args.methods:
        allowed  = set(args.methods)
        run_list = [(sn, uk, ef, lbl) for sn, uk, ef, lbl in METHODS if sn in allowed]
        missing  = allowed - {sn for sn, *_ in run_list}
        if missing:
            valid = [sn for sn, *_ in METHODS]
            print(f"[warn] Unknown method names: {missing}")
            print(f"       Valid names: {valid}")

    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    # ── run each method ───────────────────────────────────────────────────────
    rows   = []
    failed = []

    for sweep_name, unlearn_key, extra_flags, label in run_list:
        ckpt = checkpoint_path(args.save_root, sweep_name)

        if args.skip_done and os.path.exists(ckpt):
            print(f"\n[skip] {sweep_name} — checkpoint already exists")
        else:
            success, _ = run_method(
                sweep_name, unlearn_key, extra_flags,
                base_flags, args.save_root, dry_run=args.dry_run,
            )
            if not success:
                failed.append(sweep_name)

        metrics = parse_metrics(load_eval_result(ckpt))
        rows.append({
            "sweep_name":             sweep_name,
            "label":                  label,
            "unlearn_epochs":         args.unlearn_epochs,
            "unlearn_lr":             args.unlearn_lr,
            "seed":                   args.seed,
            "num_indexes_to_replace": args.num_indexes_to_replace,
            "status":                 "failed" if sweep_name in failed else "ok",
            **metrics,
        })
        print(f"  Metrics: { {k: v for k, v in metrics.items()} }")

    # ── compute Avg. Gap (retrain = oracle) ───────────────────────────────────
    retrain_row = next((r for r in rows if r["sweep_name"] == "retrain"), None)
    for row in rows:
        if row["sweep_name"] == "retrain":
            row["Avg_Gap"] = "-"
        else:
            row["Avg_Gap"] = compute_avg_gap(row, retrain_row) if retrain_row else ""

    # ── write CSV ─────────────────────────────────────────────────────────────
    all_keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})
    print(f"\nCSV saved → {args.out_csv}")

    # ── print Table-1 style summary ───────────────────────────────────────────
    COL_MAP = {
        "acc_forget": "Acc_Df(↓)",
        "acc_retain": "Acc_Dr(↑)",
        "acc_test":   "Acc_Dt(↑)",
        "MIA":        "MIA(↑)",
        "Avg_Gap":    "Avg.Gap",
    }
    present = [k for k in COL_MAP if any(k in r for r in rows)]
    SEP = "-" * (22 + 14 * len(present))
    HDR = f"{'Method':<22}" + "".join(f"{COL_MAP[k]:>14}" for k in present)

    print(f"\n{'='*len(SEP)}")
    print(f"  Table 1 — {args.dataset.upper()}  "
          f"(forget {args.num_indexes_to_replace} samples, "
          f"{args.unlearn_epochs} epochs, lr={args.unlearn_lr}, seed={args.seed})")
    print(SEP)
    print(HDR)
    print(SEP)
    for row in rows:
        line = f"{row['label']:<22}" + "".join(
            f"{str(row.get(k, 'N/A')):>14}" for k in present
        )
        print(line)
    print(SEP)

    if failed:
        print(f"\n[!] Failed methods: {failed}")


if __name__ == "__main__":
    main()
