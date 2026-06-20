"""
compute_clip_attacks.py
=======================
Compute CLIP image-text cosine similarity (CLIP score) for already-generated
attack-benchmark images across all methods and prompt CSVs.

Loads ViT-B/32 CLIP once, then for each (backbone, method, csv_stem) cell
matches every generated PNG back to its original prompt and computes the
cosine similarity. Higher = the model still follows the prompt.

Scans the same attack_eval layout produced by eval_nsfw_attacks.py:
    <attack_eval_root>/<method>/<csv_stem>/<case_number:05d>_<img_idx>.png

Handles the schema differences across all four attack CSVs automatically
(mma-diffusion uses adv_prompt + no case_number, the rest use prompt +
case_number).

Usage
-----
    # Both SDXL and SD3:
    conda activate safe
    cd /scratch/s25017/MUKSB
    python compute_clip_attacks.py --device 0

    # One backbone only:
    python compute_clip_attacks.py --device 0 --backbones SDXL
    python compute_clip_attacks.py --device 0 --backbones SD3
"""

import argparse
import glob
import json
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
SDXL_DEFAULT = "/scratch/s25017/MUKSB/SDXL/Evaluation/nsfw/attack_eval"
SD3_DEFAULT  = "/scratch/s25017/MUKSB/SD3/Evaluation/nsfw/attack_eval"

PROMPT_CSVS = {
    "nudity-diffUnlearnAttk":           "/scratch/s25017/MUKSB/SD/prompts/nudity-diffUnlearnAttk.csv",
    "nudity-ring-a-bell":               "/scratch/s25017/MUKSB/SD/prompts/nudity-ring-a-bell.csv",
    "p4dn_16_prompt":                   "/scratch/s25017/MUKSB/SD/prompts/p4dn_16_prompt.csv",
    "mma-diffusion-nsfw-adv-prompts":   "/scratch/s25017/MUKSB/SD/prompts/mma-diffusion-nsfw-adv-prompts.csv",
}

CSV_SHORT = {
    "p4dn_16_prompt":                   "P4D",
    "nudity-ring-a-bell":               "Ring-A-Bell",
    "nudity-diffUnlearnAttk":           "UnlearnDiffAttk",
    "mma-diffusion-nsfw-adv-prompts":   "MMA-Diffusion",
}

METHOD_ORDER = ["MUKSB", "MUNBa", "baseline"]


# ── Schema-agnostic prompt loader ─────────────────────────────────────────────
def load_case_to_prompt(csv_path: str) -> dict:
    """Return {case_number (int): prompt (str)}.

    Handles all four CSV schemas:
      - prompt column: "prompt" or "adv_prompt"
      - case_number  : "case_number" column, else row index
    """
    df = pd.read_csv(csv_path)
    cols = set(df.columns)
    pcol = "prompt" if "prompt" in cols else "adv_prompt"
    has_case = "case_number" in cols

    mapping = {}
    for idx, row in df.iterrows():
        p = row[pcol]
        if not isinstance(p, str) or not p.strip():
            continue
        case = int(row["case_number"]) if has_case and not pd.isna(row["case_number"]) else idx
        mapping[case] = p.strip()
    return mapping


# ── CLIP score for one folder ─────────────────────────────────────────────────
@torch.no_grad()
def clip_score_folder(model, preprocess, device: str,
                      gen_dir: str, case_to_prompt: dict) -> dict:
    img_paths = sorted(
        glob.glob(os.path.join(gen_dir, "*.png")) +
        glob.glob(os.path.join(gen_dir, "*.jpg"))
    )
    if not img_paths:
        return {}

    import clip as _clip

    scores = []
    missing = 0
    for img_path in tqdm(img_paths, desc=f"  CLIP {os.path.basename(gen_dir)}", leave=False):
        fname = os.path.basename(img_path)
        try:
            case_num = int(fname.split("_")[0])
        except ValueError:
            continue

        prompt = case_to_prompt.get(case_num)
        if prompt is None:
            missing += 1
            continue

        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        tok = _clip.tokenize([prompt], truncate=True).to(device)

        img_emb  = model.encode_image(img).float()
        text_emb = model.encode_text(tok).float()
        img_emb  = img_emb  / img_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        cos_sim  = (img_emb * text_emb).sum(dim=-1).item()
        scores.append({"case_number": case_num, "clip_score": cos_sim})

    if missing:
        print(f"    [WARN] {missing} images had no matching prompt")

    avg = sum(s["clip_score"] for s in scores) / max(len(scores), 1)
    return {
        "n_images":       len(scores),
        "avg_clip_score": round(avg, 4),
        "per_image":      scores,
    }


# ── Backbone evaluation ───────────────────────────────────────────────────────
def eval_backbone(model, preprocess, device: str,
                  attack_eval_root: str, backbone_tag: str,
                  prompt_maps: dict) -> dict:
    results = {}
    if not os.path.isdir(attack_eval_root):
        print(f"[WARN] {backbone_tag} dir not found: {attack_eval_root}")
        return results

    for method in METHOD_ORDER:
        method_dir = os.path.join(attack_eval_root, method)
        if not os.path.isdir(method_dir):
            continue
        results[method] = {}
        for stem, case_map in prompt_maps.items():
            gen_dir = os.path.join(method_dir, stem)
            if not os.path.isdir(gen_dir):
                continue
            print(f"\n  [{backbone_tag}] {method}/{stem}")
            summary = clip_score_folder(model, preprocess, device, gen_dir, case_map)
            if summary:
                results[method][stem] = summary
                print(f"    -> n={summary['n_images']}  "
                      f"avg_clip={summary['avg_clip_score']:.4f}")
                # save alongside nudenet results
                out = os.path.join(gen_dir, "clip_score.json")
                with open(out, "w") as f:
                    json.dump(summary, f, indent=2)
    return results


# ── Summary table ─────────────────────────────────────────────────────────────
def print_table(all_results: dict):
    stems   = list(PROMPT_CSVS.keys())
    bb_list = list(all_results.keys())
    col_w   = 12

    print(f"\n{'='*78}")
    print("  CLIP Score (higher = better prompt following)")
    print(f"{'='*78}")
    header = f"  {'Method':<10}"
    for bb in bb_list:
        for s in stems:
            header += f"  {(bb+'/'+CSV_SHORT.get(s,s)):>{col_w}}"
    print(header)
    print("  " + "-" * (10 + len(bb_list) * len(stems) * (col_w + 2)))

    for method in METHOD_ORDER:
        row = f"  {method:<10}"
        for bb in bb_list:
            for s in stems:
                v = all_results.get(bb, {}).get(method, {}).get(s, {})
                row += f"  {v.get('avg_clip_score', 'N/A'):>{col_w}}"
        print(row)
    print(f"{'='*78}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CLIP score for SDXL/SD3 attack-eval generated images."
    )
    parser.add_argument("--device",    type=int, default=0)
    parser.add_argument("--sdxl_dir",  type=str, default=SDXL_DEFAULT)
    parser.add_argument("--sd3_dir",   type=str, default=SD3_DEFAULT)
    parser.add_argument("--backbones", type=str, nargs="*",
                        default=["SDXL", "SD3"], choices=["SDXL", "SD3"])
    parser.add_argument("--output",    type=str, default=None)
    args = parser.parse_args()

    device_str = f"cuda:{args.device}"

    import clip
    print("[CLIP] Loading ViT-B/32...")
    model, preprocess = clip.load("ViT-B/32", device=device_str)
    model.eval()
    print("[CLIP] Ready.")

    # Load all prompt maps once
    print("\n[Prompts] Loading CSVs...")
    prompt_maps = {}
    for stem, csv_path in PROMPT_CSVS.items():
        prompt_maps[stem] = load_case_to_prompt(csv_path)
        print(f"  {stem:<42} {len(prompt_maps[stem])} entries")

    backbone_dirs = {"SDXL": args.sdxl_dir, "SD3": args.sd3_dir}

    all_results = {}
    for bb in args.backbones:
        print(f"\n{'#'*60}\n# BACKBONE: {bb}\n{'#'*60}")
        all_results[bb] = eval_backbone(
            model, preprocess, device_str,
            backbone_dirs[bb], bb, prompt_maps
        )

    print_table(all_results)

    out_path = args.output or os.path.join(
        backbone_dirs[args.backbones[0]], "..", "clip_score_summary.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Done] Summary saved: {out_path}")


if __name__ == "__main__":
    main()
