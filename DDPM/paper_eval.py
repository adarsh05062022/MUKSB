"""
paper_eval.py — standalone visual evaluation for unlearning experiments.

Generates three paper-ready figures from already-sampled images:
  1. Image grid  — base model vs unlearned model for the forgotten class
  2. Bar chart   — per-class softmax probability distribution (forgotten class samples)
  3. Entropy histogram — per-image entropy distribution (base vs unlearned)

All figures are saved to --output_dir and can be regenerated at any time
from the saved sample images without re-running unlearning or sampling.

Usage
-----
# Minimal (base model has no class_samples yet — only grid is skipped):
python paper_eval.py \
    --forget_samples  results/cifar10/forget/rl/0.001_no_mask/<timestamp>/class_samples \
    --label_to_forget 0 \
    --clf_ckpt        cifar10_resnet34.pth

# Full (with base model samples for before/after grid):
python paper_eval.py \
    --forget_samples  results/cifar10/forget/rl/0.001_no_mask/<timestamp>/class_samples \
    --base_samples    results/cifar10/2026_05_13_232008/class_samples \
    --label_to_forget 0 \
    --clf_ckpt        cifar10_resnet34.pth \
    --output_dir      paper_figs/forget_class_0

# Batch: loop over all 10 forget experiments
for L in {0..9}; do
    FOLDER=$(ls -td results/cifar10/forget/rl/0.001_no_mask/*/ | sed -n "$((L+1))p" | sed 's|/$||')
    python paper_eval.py \
        --forget_samples "${FOLDER}/class_samples" \
        --base_samples   results/cifar10/2026_05_13_232008/class_samples \
        --label_to_forget ${L} \
        --clf_ckpt       cifar10_resnet34.pth \
        --output_dir     paper_figs/forget_class_${L}
done
"""

import argparse
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── CIFAR-10 class names ──────────────────────────────────────────────────────
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "png", "ppm", "tif", "tiff", "webp"}


# ── Dataset helpers ───────────────────────────────────────────────────────────

class ImageFolderFlat(Dataset):
    """Loads all images from a flat folder (no sub-folders)."""
    def __init__(self, folder, transform=None, limit=None):
        p = pathlib.Path(folder)
        self.files = sorted(
            f for ext in IMAGE_EXTENSIONS for f in p.glob(f"*.{ext}")
        )
        if limit:
            self.files = self.files[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def clf_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def grid_transform(img_size=32):
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])


def load_classifier(ckpt_path, n_classes=10):
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probs_and_entropy(model, folder, device, batch_size=128, limit=None):
    """Returns (probs [N,10], entropy [N]) for all images in folder."""
    ds = ImageFolderFlat(folder, transform=clf_transform(), limit=limit)
    if len(ds) == 0:
        return None, None
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    all_probs = []
    for imgs in tqdm(loader, desc=f"  clf {os.path.basename(folder)}", leave=False):
        logits = model(imgs.to(device))
        probs = torch.softmax(logits, dim=-1).cpu()
        all_probs.append(probs)
    probs = torch.cat(all_probs)                          # [N, 10]
    entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=1)  # [N]
    return probs.numpy(), entropy.numpy()


# ── Figure 1: image grid ──────────────────────────────────────────────────────

def make_image_grid(folder, n_cols=10, n_rows=5, img_size=32):
    """Returns a (H, W, 3) uint8 numpy array arranged in a grid."""
    ds = ImageFolderFlat(folder, transform=grid_transform(img_size),
                         limit=n_cols * n_rows)
    n = min(len(ds), n_cols * n_rows)
    imgs = [np.array(ds[i].permute(1, 2, 0).clamp(0, 1) * 255, dtype=np.uint8)
            for i in range(n)]
    # pad if needed
    blank = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    while len(imgs) < n_cols * n_rows:
        imgs.append(blank)
    rows = [np.concatenate(imgs[r * n_cols:(r + 1) * n_cols], axis=1)
            for r in range(n_rows)]
    return np.concatenate(rows, axis=0)


def plot_image_grids(base_folder, forget_folder, label, out_path):
    """Side-by-side grid: base model (left) vs unlearned (right)."""
    has_base = base_folder and os.path.isdir(base_folder) and \
               len(list(pathlib.Path(base_folder).glob("*.png"))) > 0

    if has_base:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(make_image_grid(base_folder))
        axes[0].set_title(f"Base model — class {label} ({CLASS_NAMES[label]})",
                          fontsize=13, fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(make_image_grid(forget_folder))
        axes[1].set_title(f"After unlearning class {label} ({CLASS_NAMES[label]})",
                          fontsize=13, fontweight="bold")
        axes[1].axis("off")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.imshow(make_image_grid(forget_folder))
        ax.set_title(f"After unlearning class {label} ({CLASS_NAMES[label]})",
                     fontsize=13, fontweight="bold")
        ax.axis("off")

    plt.suptitle(f"Generated samples — forgotten class: {CLASS_NAMES[label]}",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [grid]      saved → {out_path}")


# ── Figure 2: softmax bar chart ───────────────────────────────────────────────

def plot_bar_chart(base_probs, forget_probs, label, out_path):
    """
    Mean softmax probability per class for images generated as the forgotten class.
    Shows base model vs unlearned model side by side.
    After good unlearning the forgotten class bar should collapse to ~0.1 (uniform).
    """
    x = np.arange(10)
    width = 0.35

    forget_means = forget_probs.mean(axis=0)
    forget_stds  = forget_probs.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))

    if base_probs is not None:
        base_means = base_probs.mean(axis=0)
        base_stds  = base_probs.std(axis=0)
        ax.bar(x - width / 2, base_means, width, yerr=base_stds,
               label="Base model", color="#4C72B0", alpha=0.85,
               capsize=3, error_kw={"linewidth": 0.8})
        ax.bar(x + width / 2, forget_means, width, yerr=forget_stds,
               label="Unlearned model", color="#DD8452", alpha=0.85,
               capsize=3, error_kw={"linewidth": 0.8})
    else:
        ax.bar(x, forget_means, width * 1.5, yerr=forget_stds,
               label="Unlearned model", color="#DD8452", alpha=0.85,
               capsize=3, error_kw={"linewidth": 0.8})

    # uniform reference line
    ax.axhline(1 / 10, color="gray", linestyle="--", linewidth=1,
               label="Uniform (1/10)")

    # highlight forgotten class
    ax.axvspan(label - 0.5, label + 0.5, alpha=0.12, color="red",
               label=f"Forgotten class ({CLASS_NAMES[label]})")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean softmax probability", fontsize=11)
    ax.set_title(
        f"Classifier output on forgotten-class samples — forget: {CLASS_NAMES[label]}",
        fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [bar chart] saved → {out_path}")


# ── Figure 3: entropy histogram ───────────────────────────────────────────────

def plot_entropy_histogram(base_entropy, forget_entropy, label, out_path):
    """
    Per-image entropy distribution for forgotten-class samples.
    High entropy after unlearning = model is uncertain = good unlearning.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    bins = np.linspace(0, np.log(10) + 0.1, 40)

    if base_entropy is not None:
        ax.hist(base_entropy, bins=bins, alpha=0.65, label="Base model",
                color="#4C72B0", density=True)
    ax.hist(forget_entropy, bins=bins, alpha=0.65, label="Unlearned model",
            color="#DD8452", density=True)

    ax.axvline(np.log(10), color="gray", linestyle="--", linewidth=1,
               label=f"Max entropy (uniform) = {np.log(10):.2f}")

    ax.set_xlabel("Entropy (nats)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"Entropy of classifier on forgotten-class samples — forget: {CLASS_NAMES[label]}",
        fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [entropy]   saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate paper figures from unlearning experiment samples."
    )
    p.add_argument("--forget_samples", type=str, required=True,
                   help="Path to class_samples/ folder from the UNLEARNED model "
                        "(contains sub-folders 0/, 1/, … 9/)")
    p.add_argument("--base_samples", type=str, default=None,
                   help="Path to class_samples/ folder from the BASE model "
                        "(optional — enables before/after comparison)")
    p.add_argument("--label_to_forget", type=int, required=True,
                   help="Class label that was forgotten (0-9)")
    p.add_argument("--clf_ckpt", type=str, default="cifar10_resnet34.pth",
                   help="Path to trained ResNet34 classifier checkpoint")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to save figures. Defaults to <forget_samples>/../paper_figs/")
    p.add_argument("--n_clf_samples", type=int, default=500,
                   help="Max images to pass through classifier (per model)")
    p.add_argument("--grid_rows", type=int, default=5)
    p.add_argument("--grid_cols", type=int, default=10)
    p.add_argument("--no_grid", action="store_true",
                   help="Skip image grid (useful if base_samples not available)")
    return p.parse_args()


def main():
    args = parse_args()
    label = args.label_to_forget

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.forget_samples.rstrip("/")), "paper_figs"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    forget_class_dir = os.path.join(args.forget_samples, str(label))
    base_class_dir   = (os.path.join(args.base_samples, str(label))
                        if args.base_samples else None)

    if not os.path.isdir(forget_class_dir):
        print(f"ERROR: forget_samples folder not found: {forget_class_dir}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Classifier probs / entropy ────────────────────────────────────────────
    clf_available = os.path.isfile(args.clf_ckpt)
    if not clf_available:
        print(f"  WARNING: classifier checkpoint not found at '{args.clf_ckpt}'.")
        print("  Train it first:  python train_classifier.py --dataset cifar10")
        print("  Bar chart and entropy histogram will be skipped.\n")

    forget_probs = forget_entropy = None
    base_probs   = base_entropy   = None

    if clf_available:
        print(f"\nLoading classifier from {args.clf_ckpt} ...")
        clf = load_classifier(args.clf_ckpt).to(device)

        print(f"Running classifier on unlearned-model samples (class {label}) ...")
        forget_probs, forget_entropy = get_probs_and_entropy(
            clf, forget_class_dir, device, limit=args.n_clf_samples)

        if base_class_dir and os.path.isdir(base_class_dir):
            print(f"Running classifier on base-model samples (class {label}) ...")
            base_probs, base_entropy = get_probs_and_entropy(
                clf, base_class_dir, device, limit=args.n_clf_samples)

    # ── Figure 1: image grid ──────────────────────────────────────────────────
    if not args.no_grid:
        print("\nGenerating image grid ...")
        plot_image_grids(
            base_folder   = base_class_dir,
            forget_folder = forget_class_dir,
            label         = label,
            out_path      = os.path.join(args.output_dir,
                                         f"grid_forget_{label}_{CLASS_NAMES[label]}.png"),
        )

    # ── Figure 2: bar chart ───────────────────────────────────────────────────
    if clf_available and forget_probs is not None:
        print("Generating bar chart ...")
        plot_bar_chart(
            base_probs   = base_probs,
            forget_probs = forget_probs,
            label        = label,
            out_path     = os.path.join(args.output_dir,
                                        f"barchart_forget_{label}_{CLASS_NAMES[label]}.png"),
        )

    # ── Figure 3: entropy histogram ───────────────────────────────────────────
    if clf_available and forget_entropy is not None:
        print("Generating entropy histogram ...")
        plot_entropy_histogram(
            base_entropy   = base_entropy,
            forget_entropy = forget_entropy,
            label          = label,
            out_path       = os.path.join(args.output_dir,
                                          f"entropy_forget_{label}_{CLASS_NAMES[label]}.png"),
        )

    print(f"\nDone. All figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
