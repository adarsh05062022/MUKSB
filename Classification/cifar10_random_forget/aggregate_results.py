#!/usr/bin/env python
"""
aggregate_results.py — collect per-seed eval results into a Table-1 summary.

Walks   <root>/<method>/seed<N>/*eval_result.pth.tar
and reports, per method, mean ± std across seeds for:

    Acc_Df (↓)   Acc_Dr (↑)   Acc_Dt (↑)   MIA (↑)

plus  Avg.Gap = mean over those 4 metrics of |method_mean − retrain_mean|
(same definition as the MUNBa / MUKSB paper Table 1; retrain is the oracle).

Usage
-----
    python aggregate_results.py \
        --root    cifar10_random_forget/checkpoints \
        --out     cifar10_random_forget/results
"""
import argparse
import csv
import glob
import json
import math
import os

import torch

# Display order + pretty labels (keys = the method folder names used by the driver)
METHOD_ORDER = ["retrain", "FT", "GA", "IU", "BE", "l1sparse", "SalUn", "MUNBa", "MUKSB"]
LABELS = {
    "retrain": "Retrain", "FT": "FT", "GA": "GA", "IU": "IU", "BE": "BE",
    "l1sparse": "l1-sparse", "SalUn": "SalUn", "MUNBa": "MUNBa", "MUKSB": "MUKSB",
}
# metric key -> (column header, gap-metric?)
METRICS = [
    ("acc_forget", "Acc_Df(v)"),
    ("acc_retain", "Acc_Dr(^)"),
    ("acc_test",   "Acc_Dt(^)"),
    ("MIA",        "MIA(^)"),
]
GAP_KEYS = ["acc_forget", "acc_retain", "acc_test", "MIA"]


def load_eval(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not load {path}: {e}")
        return None


def extract(ev):
    """Flatten one evaluation_result dict into scalar metrics."""
    if ev is None:
        return {}
    m = {}
    acc = ev.get("accuracy") or ev.get("new_accuracy") or {}
    if isinstance(acc, dict):
        for split in ("forget", "retain", "val", "test"):
            v = acc.get(split)
            if v is not None:
                m[f"acc_{split}"] = float(v)
    mia = ev.get("MIA")
    if mia is not None:
        m["MIA"] = float(mia) * 100.0   # report as a percentage, like the accuracies
    return m


def load_epoch_metrics(seed_dir):
    """Per-epoch status list (MUKSB-style epoch_metrics.json), or [] if absent."""
    path = os.path.join(seed_dir, "epoch_metrics.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not read {path}: {e}")
        return []


def build_run_json(method, seed, seed_dir):
    """Combined per-run record: final metrics + every epoch's status."""
    files = glob.glob(os.path.join(seed_dir, "*eval_result.pth.tar"))
    final = extract(load_eval(files[0])) if files else {}
    return {
        "method": method,
        "label": LABELS.get(method, method),
        "seed": seed,
        "final_metrics": final,
        "epochs": load_epoch_metrics(seed_dir),
    }


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
    return mu, sd


def collect(root):
    """method -> {seed -> metrics dict}"""
    out = {}
    for method_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(method_dir):
            continue
        method = os.path.basename(method_dir)
        per_seed = {}
        for seed_dir in sorted(glob.glob(os.path.join(method_dir, "seed*"))):
            files = glob.glob(os.path.join(seed_dir, "*eval_result.pth.tar"))
            if not files:
                continue
            seed = os.path.basename(seed_dir).replace("seed", "")
            per_seed[seed] = extract(load_eval(files[0]))
        if per_seed:
            out[method] = per_seed
    return out


def fmt(mu, sd):
    return "N/A" if mu is None else f"{mu:.2f}+/-{sd:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="checkpoints root (<method>/seed<N>/)")
    ap.add_argument("--out", default=None, help="output dir for CSVs (default: <root>/..)")
    args = ap.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.root))
    os.makedirs(out_dir, exist_ok=True)

    data = collect(args.root)
    if not data:
        print(f"[error] no *eval_result.pth.tar found under {args.root}")
        return

    # ── per-method aggregate ─────────────────────────────────────────────────
    agg = {}   # method -> {metric -> (mu, sd)}, plus "_n"
    for method, per_seed in data.items():
        seeds = sorted(per_seed.keys(), key=lambda s: int(s) if s.isdigit() else s)
        stats = {"_n": len(seeds), "_seeds": ",".join(seeds)}
        for key, _ in METRICS:
            stats[key] = mean_std([per_seed[s].get(key) for s in seeds])
        agg[method] = stats

    # ── Avg.Gap vs retrain mean ──────────────────────────────────────────────
    retrain = agg.get("retrain")
    for method, stats in agg.items():
        if method == "retrain" or retrain is None:
            stats["Avg_Gap"] = None
            continue
        diffs = []
        for k in GAP_KEYS:
            mu = stats.get(k, (None,))[0]
            rmu = retrain.get(k, (None,))[0]
            if mu is not None and rmu is not None:
                diffs.append(abs(mu - rmu))
        stats["Avg_Gap"] = (sum(diffs) / len(diffs)) if diffs else None

    ordered = [m for m in METHOD_ORDER if m in agg] + \
              [m for m in agg if m not in METHOD_ORDER]

    # ── write per-seed CSV ───────────────────────────────────────────────────
    per_seed_csv = os.path.join(out_dir, "per_seed_metrics.csv")
    with open(per_seed_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed"] + [k for k, _ in METRICS])
        for method in ordered:
            for seed, m in sorted(data[method].items(),
                                  key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
                w.writerow([method, seed] + [m.get(k, "") for k, _ in METRICS])

    # ── write summary CSV (mean / std / n) ───────────────────────────────────
    summary_csv = os.path.join(out_dir, "summary_mean_std.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["method", "label", "n_seeds", "seeds"]
        for k, _ in METRICS:
            header += [f"{k}_mean", f"{k}_std"]
        header += ["Avg_Gap"]
        w.writerow(header)
        for method in ordered:
            s = agg[method]
            row = [method, LABELS.get(method, method), s["_n"], s["_seeds"]]
            for k, _ in METRICS:
                mu, sd = s[k]
                row += ["" if mu is None else round(mu, 4),
                        "" if sd is None else round(sd, 4)]
            row += ["" if s["Avg_Gap"] is None else round(s["Avg_Gap"], 3)]
            w.writerow(row)

    # ── per-run combined JSON (final metrics + epoch status) ─────────────────
    per_run_dir = os.path.join(out_dir, "per_run")
    os.makedirs(per_run_dir, exist_ok=True)
    all_runs = []
    for method in ordered:
        for seed in sorted(data[method].keys(),
                           key=lambda s: int(s) if s.isdigit() else s):
            seed_dir = os.path.join(args.root, method, f"seed{seed}")
            rj = build_run_json(method, seed, seed_dir)
            with open(os.path.join(per_run_dir, f"{method}_seed{seed}.json"), "w") as f:
                json.dump(rj, f, indent=2)
            all_runs.append(rj)
    with open(os.path.join(out_dir, "all_runs.json"), "w") as f:
        json.dump(all_runs, f, indent=2)

    # ── pretty print ─────────────────────────────────────────────────────────
    cols = [h for _, h in METRICS] + ["Avg.Gap"]
    sep = "-" * (12 + 16 * len(cols))
    print("\n" + "=" * len(sep))
    print("  CIFAR-10 / ResNet-18 — 10% random forgetting  (mean +/- std over seeds)")
    print(sep)
    print(f"{'Method':<12}" + "".join(f"{c:>16}" for c in cols))
    print(sep)
    for method in ordered:
        s = agg[method]
        cells = [fmt(*s[k]) for k, _ in METRICS]
        gap = "-" if s["Avg_Gap"] is None else f"{s['Avg_Gap']:.3f}"
        cells.append(gap)
        print(f"{LABELS.get(method, method):<12}" + "".join(f"{c:>16}" for c in cells))
    print(sep)
    print(f"  n_seeds per method: " +
          ", ".join(f"{LABELS.get(m, m)}={agg[m]['_n']}" for m in ordered))
    print(f"\n  per-seed CSV : {per_seed_csv}")
    print(f"  summary CSV  : {summary_csv}")
    print(f"  per-run JSON : {per_run_dir}/<method>_seed<N>.json  (final + epoch status)")
    print(f"  all runs JSON: {os.path.join(out_dir, 'all_runs.json')}")
    print("  Note: MIA reported as a percentage; Avg.Gap is vs the Retrain oracle.")


if __name__ == "__main__":
    main()
