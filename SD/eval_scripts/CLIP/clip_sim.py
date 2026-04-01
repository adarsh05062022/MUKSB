"""
CLIP Cosine Similarity Matrix — Before / After Unlearning
==========================================================
Directory layout expected:
    before/          ← images generated BEFORE unlearning  (reference)
        0_001.png
        0_002.png
        1_001.png
        ...
    after/           ← images generated AFTER unlearning   (query)
        0_001.png
        ...
    prompts.csv      ← case_number, prompt, seed

HOW THE MATRIX IS COMPUTED:
    before_mat[i,j] = sim( before_embed[i],  before_embed[j] )   self vs self
    after_mat[i,j]  = sim( after_embed[i],   before_embed[j] )   after  vs before-reference

    For the forget class row/col the after_mat diagonal value should DROP
    because the model can no longer reproduce that concept.

Usage:
    python clip_similarity.py \
        --before_dir  ./before \
        --after_dir   ./after \
        --prompts_csv ./prompts.csv \
        --forget_class 0 \
        --output_dir  ./results
"""

import os
import re
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image

try:
    import clip
    import torch
except ImportError:
    raise SystemExit(
        "[ERROR] Missing dependencies.\n"
        "Install:  pip install openai-clip torch torchvision"
    )

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_prompts(csv_path: str) -> dict:
    """Return {case_number: prompt}."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # support both 'case_number' and 'classidx' column names
    id_col = "case_number" if "case_number" in df.columns else "classidx"

    mapping = {}
    for _, row in df.iterrows():
        idx = int(row[id_col])
        if idx not in mapping:
            mapping[idx] = str(row["prompt"]).strip()
    return mapping


def parse_classidx_from_filename(fname: str):
    """
    0_001.png       -> 0
    3_seed42.png    -> 3
    class5_img2.jpg -> 5
    """
    stem = Path(fname).stem
    m = re.match(r"(?:class)?(\d+)[_\-]", stem)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)$", stem)
    if m:
        return int(m.group(1))
    return None


def load_images_by_class(image_dir: str) -> dict:
    """Return {classidx: [path, ...]} scanning a flat directory."""
    groups = defaultdict(list)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for p in sorted(Path(image_dir).iterdir()):
        if p.suffix.lower() not in exts:
            continue
        idx = parse_classidx_from_filename(p.name)
        if idx is None:
            print(f"  [WARN] Cannot parse classidx from '{p.name}' — skipping")
            continue
        groups[idx].append(p)
    return dict(groups)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CLIP ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def build_clip(device: str):
    print(f"[INFO] Loading CLIP ViT-B/32 on {device} ...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


@torch.no_grad()
def encode_images(paths, model, preprocess, device, batch_size=64):
    """Returns single L2-normalised mean embedding for a list of images."""
    embeddings = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        imgs = []
        for p in batch:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except Exception as e:
                print(f"  [WARN] Failed to load {p}: {e}")
        if not imgs:
            continue
        tensor = torch.stack(imgs).to(device)
        feats = model.encode_image(tensor).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())

    if not embeddings:
        return np.zeros(512)
    all_feats = np.concatenate(embeddings, axis=0)   # (N, 512)
    mean_feat = all_feats.mean(axis=0)
    mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-8)
    return mean_feat                                  # (512,)


def get_embeddings_for_dir(image_dir, class_order, model, preprocess, device):
    """Encode every class in image_dir; return {classidx: embedding}."""
    groups = load_images_by_class(image_dir)
    embeddings = {}
    for idx in class_order:
        paths = groups.get(idx, [])
        if not paths:
            print(f"  [WARN] No images for classidx={idx} in '{image_dir}'")
            embeddings[idx] = np.zeros(512)
        else:
            print(f"  Encoding class {idx}  ({len(paths)} images) ...")
            embeddings[idx] = encode_images(paths, model, preprocess, device)
    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MATRIX COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_cross_similarity_matrix(query_embeds, ref_embeds, class_order):
    """
    mat[i, j] = cosine_sim( query_embed[class_i],  ref_embed[class_j] )

    before_mat : query=before, ref=before  -> pure self-similarity (diagonal ~ 1)
    after_mat  : query=after,  ref=before  -> how well after reproduces before
                                              forget class diagonal drops toward 0
    """
    N = len(class_order)
    mat = np.zeros((N, N))
    for i, ci in enumerate(class_order):
        for j, cj in enumerate(class_order):
            mat[i, j] = float(np.dot(query_embeds[ci], ref_embeds[cj]))
    return mat


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

CMAP = "viridis"


def _plot_single_matrix(
    ax, mat, labels, title, vmin, vmax,
    forget_idx=None, thumbnail_paths=None, thumb_size=0.15,
):
    N = len(labels)
    ax.imshow(mat, cmap=CMAP, vmin=vmin, vmax=vmax, aspect="equal")

    # cell text
    thresh = vmin + (vmax - vmin) * 0.6
    for i in range(N):
        for j in range(N):
            val = mat[i, j]
            colour = "white" if val < thresh else "black"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=6, color=colour, fontweight="bold")

    # red border around forget class row & column
    if forget_idx is not None:
        fi = forget_idx
        lw = 2.5
        # highlight row
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.5, fi - 0.5), N, 1,
            boxstyle="square,pad=0", linewidth=lw,
            edgecolor="red", facecolor="none", zorder=5
        ))
        # highlight column
        ax.add_patch(mpatches.FancyBboxPatch(
            (fi - 0.5, -0.5), 1, N,
            boxstyle="square,pad=0", linewidth=lw,
            edgecolor="red", facecolor="none", zorder=5
        ))

    # axes labels
    ax.set_xticks([])
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.tick_params(left=False, bottom=False)
    ax.set_title(title, fontsize=10, fontweight="bold",
                 pad=32 if thumbnail_paths else 8)

    # thumbnails above columns
    if thumbnail_paths:
        fig    = ax.get_figure()
        ax_pos = ax.get_position()
        cell_w = ax_pos.width / N
        for j, p in enumerate(thumbnail_paths):
            if p is None:
                continue
            try:
                img   = Image.open(p).convert("RGB")
                left  = ax_pos.x0 + j * cell_w + cell_w * 0.05
                bot   = ax_pos.y1 + 0.004
                w     = cell_w * 0.88
                inset = fig.add_axes([left, bot, w, thumb_size])
                inset.imshow(img)
                inset.axis("off")
                if forget_idx is not None and j == forget_idx:
                    for spine in inset.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor("red")
                        spine.set_linewidth(2)
            except Exception as e:
                print(f"  [WARN] thumbnail {p}: {e}")


def save_comparison_figure(
    before_mat, after_mat, labels, output_path,
    forget_idx=None, thumbnail_paths=None,
):
    vmin = min(before_mat.min(), after_mat.min())
    vmax = max(before_mat.max(), after_mat.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.subplots_adjust(left=0.13, right=0.92, top=0.78, bottom=0.05, wspace=0.38)

    _plot_single_matrix(axes[0], before_mat, labels,
                        "",
                        vmin, vmax, forget_idx, thumbnail_paths)

    _plot_single_matrix(axes[1], after_mat, labels,
                        "",
                        vmin, vmax, forget_idx, thumbnail_paths)

    # # shared colorbar
    # cbar_ax = fig.add_axes([0.93, 0.10, 0.014, 0.62])
    # norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # sm   = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    # sm.set_array([])
    # fig.colorbar(sm, cax=cbar_ax)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved comparison figure -> {output_path}")


def save_individual_figures(
    before_mat, after_mat, labels, output_dir,
    forget_idx=None, thumbnail_paths=None
):
    for mat, tag, title in [
        (before_mat, "before", "(a) Original cosine similarity matrix"),
        (after_mat,  "after",  "(b) Cosine similarity matrix after unlearning"),
    ]:
        vmin, vmax = mat.min(), mat.max()
        fig, ax = plt.subplots(figsize=(5.5, 5))
        fig.subplots_adjust(left=0.20, right=0.97, top=0.78, bottom=0.05)
        _plot_single_matrix(ax, mat, labels, title,
                            vmin, vmax, forget_idx, thumbnail_paths)
        path = os.path.join(output_dir, f"clip_similarity_{tag}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved individual figure -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_matrix_csv(mat, labels, path):
    df = pd.DataFrame(mat, index=labels, columns=labels)
    df.to_csv(path, float_format="%.4f")
    print(f"[INFO] Saved matrix CSV -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  THUMBNAILS
# ══════════════════════════════════════════════════════════════════════════════

def pick_thumbnails(image_dir, class_order):
    groups = load_images_by_class(image_dir)
    thumbs = [groups.get(idx, [None])[0] for idx in class_order]
    return None if all(t is None for t in thumbs) else thumbs


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TERMINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_diagonal_summary(before_mat, after_mat, labels, forget_idx):
    print("\n[INFO] Diagonal: before_mat[i,i] vs after_mat[i,i]")
    print("       after_mat[i,i] = sim(after_class_i, before_class_i)")
    print("       Forget class diagonal should DROP significantly.\n")
    print(f"  {'Label':<35}  {'Before':>8}  {'After':>8}  {'Delta':>8}  Note")
    print("  " + "-" * 78)
    for i, lbl in enumerate(labels):
        b, a = before_mat[i, i], after_mat[i, i]
        note = "  <- FORGET CLASS" if i == forget_idx else (
               "  <- UNLEARNED?"   if (b - a) > 0.10  else "")
        print(f"  {lbl:<35}  {b:>8.4f}  {a:>8.4f}  {a-b:>+8.4f}{note}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLIP cosine similarity matrices before/after unlearning."
    )
    parser.add_argument("--before_dir",  required=False,
                        default="/storage/s25017/MUKSB/SD/eval_scripts_food/CLASS/generated_images/orginal food",
                        help="Flat dir with BEFORE images  (classidx_xxx.png)")
    parser.add_argument("--after_dir",   required=False,
                        default="/storage/s25017/MUKSB/SD/eval_scripts_food/CLASS/generated_images/diffusers-cls_1-MUNBa-method_full-lr_1e-05_E5_U750_FOOD101_masked_nash_topk10_frequent_mask.pt",
                        help="Flat dir with AFTER images   (classidx_xxx.png)")
    parser.add_argument("--prompts_csv", required=False,
                        default="/storage/s25017/MUKSB/SD/eval_scripts/CLIP/prompts.csv",
                        help="CSV with columns: case_number (or classidx), prompt, seed")
    parser.add_argument("--output_dir",  default="./results")
    parser.add_argument("--forget_class", type=int, default=1,
                        help="classidx that was unlearned — draws red border on that row/col")
    parser.add_argument("--class_order", nargs="*", type=int, default=None,
                        help="Explicit class indices in display order")
    parser.add_argument("--label_type",  choices=["prompt", "classidx"], default="prompt")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_thumbnails", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # prompts / labels
    print("[INFO] Loading prompts ...")
    prompt_map  = load_prompts(args.prompts_csv)
    class_order = args.class_order if args.class_order else sorted(prompt_map.keys())
    print(f"[INFO] Class order: {class_order}")

    if args.label_type == "prompt":
        labels = [
            prompt_map.get(idx, str(idx))[:32] + ("..." if len(prompt_map.get(idx, "")) > 32 else "")
            for idx in class_order
        ]
    else:
        labels = [str(idx) for idx in class_order]

    # find position of forget class
    forget_idx = None
    if args.forget_class is not None:
        if args.forget_class in class_order:
            forget_idx = class_order.index(args.forget_class)
            print(f"[INFO] Forget class = {args.forget_class}  (position {forget_idx})")
        else:
            print(f"[WARN] --forget_class {args.forget_class} not found in class_order")

    # CLIP model
    model, preprocess = build_clip(args.device)

    # encode both directories
    print("\n[INFO] Encoding BEFORE images ...")
    before_embeds = get_embeddings_for_dir(
        args.before_dir, class_order, model, preprocess, args.device)

    print("\n[INFO] Encoding AFTER images ...")
    after_embeds = get_embeddings_for_dir(
        args.after_dir, class_order, model, preprocess, args.device)

    # compute matrices
    # before_mat[i,j] = sim(before_i, before_j)  — pure self-similarity
    # after_mat[i,j]  = sim(after_i,  before_j)  — how well after reproduces before
    print("\n[INFO] Computing similarity matrices ...")
    before_mat = compute_cross_similarity_matrix(before_embeds, before_embeds, class_order)
    after_mat  = compute_cross_similarity_matrix(after_embeds,  before_embeds, class_order)

    # thumbnails
    thumbnail_paths = None
    if not args.no_thumbnails:
        thumbnail_paths = pick_thumbnails(args.before_dir, class_order)

    # save CSVs
    save_matrix_csv(before_mat, labels,
                    os.path.join(args.output_dir, "similarity_before.csv"))
    save_matrix_csv(after_mat,  labels,
                    os.path.join(args.output_dir, "similarity_after.csv"))

    # plots
    save_comparison_figure(
        before_mat, after_mat, labels,
        os.path.join(args.output_dir, "clip_similarity_comparison.png"),
        forget_idx=forget_idx,
        thumbnail_paths=thumbnail_paths,
    )
    save_individual_figures(
        before_mat, after_mat, labels, args.output_dir,
        forget_idx=forget_idx,
        thumbnail_paths=thumbnail_paths,
    )

    # terminal summary
    print_diagonal_summary(before_mat, after_mat, labels, forget_idx)
    print("[DONE] All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()