#!/usr/bin/env python3
"""Display all sweep checkpoint eval results in a formatted table."""

import os
import torch

SWEEP_DIR = os.path.join(os.path.dirname(__file__), "sweep_checkpoints")


def load_eval_result(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def collect_results():
    results = []
    for method_dir in sorted(os.listdir(SWEEP_DIR)):
        dir_path = os.path.join(SWEEP_DIR, method_dir)
        if not os.path.isdir(dir_path):
            continue
        eval_files = [f for f in os.listdir(dir_path) if "eval_result" in f]
        if not eval_files:
            results.append({"method": method_dir, "error": "no eval_result file"})
            continue
        eval_path = os.path.join(dir_path, eval_files[0])
        try:
            data = load_eval_result(eval_path)
            results.append({"method": method_dir, "data": data})
        except Exception as e:
            results.append({"method": method_dir, "error": str(e)})
    return results


def fmt(val, decimals=4):
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def print_results(results):
    # --- Accuracy table ---
    acc_keys = ["retain", "forget", "val", "test"]
    svc_forget_keys = ["correctness", "confidence", "entropy", "m_entropy", "prob"]
    svc_train_keys = svc_forget_keys

    sep = "-" * 120

    print("\n" + "=" * 120)
    print("SWEEP CHECKPOINT RESULTS")
    print("=" * 120)

    # Accuracy
    print(f"\n{'ACCURACY':}")
    header = f"{'Method':<18}" + "".join(f"{k:>12}" for k in acc_keys)
    print(header)
    print(sep[:len(header)])
    for r in results:
        if "error" in r:
            print(f"{r['method']:<18}  ERROR: {r['error']}")
            continue
        acc = r["data"].get("accuracy", {})
        row = f"{r['method']:<18}" + "".join(f"{fmt(acc.get(k, float('nan'))):>12}" for k in acc_keys)
        print(row)

    # MIA
    print(f"\n{'MIA (Membership Inference Attack)'}")
    header = f"{'Method':<18}{'MIA':>12}"
    print(header)
    print(sep[:len(header)])
    for r in results:
        if "error" in r:
            continue
        mia = r["data"].get("MIA", float("nan"))
        print(f"{r['method']:<18}{fmt(mia):>12}")

    # SVC MIA Forget Efficacy
    print(f"\n{'SVC_MIA Forget Efficacy'}")
    header = f"{'Method':<18}" + "".join(f"{k:>14}" for k in svc_forget_keys)
    print(header)
    print(sep[:len(header)])
    for r in results:
        if "error" in r:
            continue
        d = r["data"].get("SVC_MIA_forget_efficacy", {})
        row = f"{r['method']:<18}" + "".join(f"{fmt(d.get(k, float('nan'))):>14}" for k in svc_forget_keys)
        print(row)

    # SVC MIA Training Privacy
    print(f"\n{'SVC_MIA Training Privacy'}")
    header = f"{'Method':<18}" + "".join(f"{k:>14}" for k in svc_train_keys)
    print(header)
    print(sep[:len(header)])
    for r in results:
        if "error" in r:
            continue
        d = r["data"].get("SVC_MIA_training_privacy", {})
        row = f"{r['method']:<18}" + "".join(f"{fmt(d.get(k, float('nan'))):>14}" for k in svc_train_keys)
        print(row)

    # Errors summary
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n{'MISSING / ERRORS'}")
        for r in errors:
            print(f"  {r['method']}: {r['error']}")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    results = collect_results()
    print_results(results)
