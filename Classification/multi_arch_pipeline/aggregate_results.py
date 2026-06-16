"""aggregate_results.py — collect MUKSB/retrain results across architectures.

Walks the results tree produced by run_unlearn.sh and writes a single CSV
with one row per (arch, method, seed) and mean ± std summary rows per
(arch, method).  Designed for inclusion in the paper supplementary.

Layout expected:
    <results_root>/<dataset>/<arch>/<method>/<forget_tag>/seed<seed>/
        eval_result.pth.tar
        epoch_metrics.json

Usage:
    python aggregate_results.py \
        --results_root /scratch/.../Classification/results_multi_arch \
        --out_csv      /scratch/.../multi_arch_pipeline/multi_arch_summary.csv
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean, stdev

import torch


def _flatten_eval(eval_dict):
    """Pull the headline numbers out of the eval_result dict."""
    out = {
        "acc_retain": None,
        "acc_forget": None,
        "acc_val":    None,
        "acc_test":   None,
        "mia":        None,
        "svc_mia_forget_efficacy_confidence": None,
        "svc_mia_forget_efficacy_entropy":    None,
    }
    if not isinstance(eval_dict, dict):
        return out

    acc = eval_dict.get("accuracy", {}) or {}
    out["acc_retain"] = acc.get("retain")
    out["acc_forget"] = acc.get("forget")
    out["acc_val"]    = acc.get("val")
    out["acc_test"]   = acc.get("test")
    out["mia"]        = eval_dict.get("MIA")

    svc = eval_dict.get("SVC_MIA_forget_efficacy") or {}
    if isinstance(svc, dict):
        out["svc_mia_forget_efficacy_confidence"] = svc.get("confidence")
        out["svc_mia_forget_efficacy_entropy"]    = svc.get("entropy")
    return out


def _load_eval(seed_dir):
    f = os.path.join(seed_dir, "eval_resulteval_result.pth.tar")  # never matches
    candidates = [
        os.path.join(seed_dir, "eval_result.pth.tar"),
        os.path.join(seed_dir, "MUKSBeval_result.pth.tar"),
        os.path.join(seed_dir, "retraineval_result.pth.tar"),
    ]
    # The actual filename is "<method>eval_result.pth.tar" because save_checkpoint
    # prepends args.unlearn to the filename.  Try a glob as final fallback.
    for c in candidates:
        if os.path.isfile(c):
            try:
                obj = torch.load(c, map_location="cpu", weights_only=False)
                # save_unlearn_checkpoint stored evaluation_result directly here.
                return obj if isinstance(obj, dict) else {}
            except Exception as e:
                print(f"  [warn] failed to load {c}: {e}")

    for c in glob.glob(os.path.join(seed_dir, "*eval_result.pth.tar")):
        try:
            obj = torch.load(c, map_location="cpu", weights_only=False)
            return obj if isinstance(obj, dict) else {}
        except Exception as e:
            print(f"  [warn] failed to load {c}: {e}")
    return {}


def _final_epoch_metrics(seed_dir):
    f = os.path.join(seed_dir, "epoch_metrics.json")
    if not os.path.isfile(f):
        return None
    try:
        with open(f) as fh:
            data = json.load(fh)
        if not data:
            return None
        return data[-1]
    except Exception as e:
        print(f"  [warn] failed to parse {f}: {e}")
        return None


def _agg(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return (None, None, 0)
    if len(vals) == 1:
        return (vals[0], 0.0, 1)
    return (mean(vals), stdev(vals), len(vals))


def collect(results_root):
    """Yield dicts: one per (arch, method, seed) discovered on disk."""
    pattern = os.path.join(
        results_root, "*", "*", "*", "*", "seed*")
    for seed_dir in sorted(glob.glob(pattern)):
        parts = seed_dir.split(os.sep)
        try:
            seed       = int(parts[-1].replace("seed", ""))
            forget_tag = parts[-2]
            method     = parts[-3]
            arch       = parts[-4]
            dataset    = parts[-5]
        except (IndexError, ValueError):
            continue

        eval_dict = _load_eval(seed_dir)
        row = {
            "dataset":    dataset,
            "arch":       arch,
            "method":     method,
            "forget_tag": forget_tag,
            "seed":       seed,
            "path":       seed_dir,
        }
        row.update(_flatten_eval(eval_dict))

        # Use epoch_metrics.json as a fallback / sanity check for accuracy.
        if any(row.get(k) is None for k in ("acc_retain", "acc_forget", "acc_test")):
            last = _final_epoch_metrics(seed_dir)
            if last and isinstance(last.get("accuracy"), dict):
                acc = last["accuracy"]
                for k_src, k_dst in [
                    ("retain", "acc_retain"),
                    ("forget", "acc_forget"),
                    ("val",    "acc_val"),
                    ("test",   "acc_test"),
                ]:
                    if row.get(k_dst) is None and k_src in acc:
                        row[k_dst] = acc[k_src]
        yield row


METRIC_COLS = [
    "acc_retain",
    "acc_forget",
    "acc_val",
    "acc_test",
    "mia",
    "svc_mia_forget_efficacy_confidence",
    "svc_mia_forget_efficacy_entropy",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out_csv",      required=True)
    args = ap.parse_args()

    rows = list(collect(args.results_root))
    if not rows:
        print(f"No results found under {args.results_root}")
        return

    print(f"Collected {len(rows)} (arch, method, seed) runs from "
          f"{args.results_root}")

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["arch"], r["method"], r["forget_tag"])].append(r)

    summary_rows = []
    for key, group in sorted(grouped.items()):
        ds, arch, method, tag = key
        agg_row = {
            "dataset": ds, "arch": arch, "method": method,
            "forget_tag": tag, "seed": "MEAN±STD",
            "n_seeds": len(group),
            "path": "",
        }
        for col in METRIC_COLS:
            m, s, _ = _agg([r.get(col) for r in group])
            if m is None:
                agg_row[col] = ""
            else:
                agg_row[col] = f"{m:.4f}±{s:.4f}"
        summary_rows.append(agg_row)

    fieldnames = (
        ["dataset", "arch", "method", "forget_tag", "seed", "n_seeds"]
        + METRIC_COLS
        + ["path"]
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (
                x["dataset"], x["arch"], x["method"], x["forget_tag"], x["seed"])):
            r = dict(r)
            r["n_seeds"] = 1
            for col in METRIC_COLS:
                v = r.get(col)
                if isinstance(v, float):
                    r[col] = f"{v:.4f}"
            w.writerow(r)
        w.writerow({})  # blank separator
        for r in summary_rows:
            w.writerow(r)

    print(f"Wrote {args.out_csv}")
    print()
    print("=== Summary (mean ± std across seeds) ===")
    for r in summary_rows:
        print(f"  {r['arch']:>12s} | {r['method']:>8s} | n={r['n_seeds']} | "
              f"retain={r.get('acc_retain','')} forget={r.get('acc_forget','')} "
              f"test={r.get('acc_test','')} MIA={r.get('mia','')}")


if __name__ == "__main__":
    main()
