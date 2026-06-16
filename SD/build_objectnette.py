"""
build_objectnette.py — one-time builder for the SD-generated OBJECT dataset
===========================================================================
Generates a 10-class object dataset with the EXACT same on-disk layout as
imagenette2 (ImageFolder), so object-concept removal reuses the imagenette
class-removal code path verbatim.

Layout produced (mirrors /storage/s25017/Datasets/imagenette2):

    /storage/s25017/Datasets/objectnette2/
        train/
            airplane/  00000_0.png ...
            bicycle/   ...
            bird/  boat/  car/  cat/  dog/  horse/  train/  truck/

Each class folder holds ~N images (default 900, ~Imagenette scale), generated
from vanilla SD v1.4 with varied prompt templates and unique seeds.

Run once before training:

    python build_objectnette.py --gpus 0 1 2 3 --n_per_class 900
    python build_objectnette.py --gpus 4 7 --classes dog cat   # subset
    python build_objectnette.py --gpus 4 --smoke                # 8/class quick test
"""

import argparse
import csv
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "Evaluation", "objects"))

from generate_objects_multigpu import launch_multigpu
from train_scripts.dataset import OBJECTNETTE_CLASSES, OBJECTNETTE_ROOT


# ── Prompt vocabulary ─────────────────────────────────────────────────────────
# Per-class surface nouns give visual diversity; templates vary scene/style.
CLASS_NOUNS = {
    "dog":      ["dog", "puppy", "golden retriever dog", "labrador dog",
                 "german shepherd dog", "beagle dog"],
    "cat":      ["cat", "kitten", "tabby cat", "black cat", "white cat",
                 "siamese cat"],
    "car":      ["car", "sports car", "sedan car", "vintage car",
                 "red car", "convertible car"],
    "bicycle":  ["bicycle", "mountain bicycle", "road bicycle",
                 "vintage bicycle", "racing bicycle", "red bicycle"],
    "airplane": ["airplane", "commercial airplane", "passenger airplane",
                 "small airplane", "jet airplane", "propeller airplane"],
    "bird":     ["bird", "small bird", "colorful bird", "songbird",
                 "bird perched on a branch", "wild bird"],
    "horse":    ["horse", "brown horse", "running horse", "white horse",
                 "horse in a field", "wild horse"],
    "boat":     ["boat", "sailing boat", "fishing boat", "speed boat",
                 "wooden boat", "boat on the water"],
    "truck":    ["truck", "delivery truck", "pickup truck", "semi truck",
                 "dump truck", "cargo truck"],
    "train":    ["train", "steam train", "high speed train", "freight train",
                 "passenger train", "modern train"],
}

TEMPLATES = [
    "a photo of a {x}",
    "a high quality photograph of a {x}",
    "a realistic image of a {x}",
    "a {x} outdoors in daylight",
    "a professional photo of a {x}",
    "a {x} close up",
    "portrait of a {x}",
    "a detailed photo of a {x}",
    "a {x} during the day",
    "studio photograph of a {x}",
    "a {x} on a clear day",
    "a high resolution photo of a {x}",
]


def build_class_csv(class_name, n_images, rng, out_csv):
    """Write n_images rows of (case_number, prompt, evaluation_seed, concept).
    Prompts cycle through (noun x template) combos; every row gets a unique seed
    so identical prompts still produce distinct images."""
    nouns  = CLASS_NOUNS[class_name]
    combos = [t.format(x=n) for n in nouns for t in TEMPLATES]
    rng.shuffle(combos)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_number", "prompt", "evaluation_seed", "concept"])
        for i in range(n_images):
            prompt = combos[i % len(combos)]
            w.writerow([i, prompt, rng.randint(1, 10_000_000), class_name])
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Build SD-generated objectnette2 dataset")
    ap.add_argument("--root",        type=str, default=OBJECTNETTE_ROOT,
                    help="dataset root (train/<class>/ is created under it)")
    ap.add_argument("--classes",     type=str, nargs="+", default=OBJECTNETTE_CLASSES,
                    help="subset of classes to (re)generate")
    ap.add_argument("--n_per_class", type=int, default=900)
    ap.add_argument("--gpus",        type=int, nargs="+", default=[0])
    ap.add_argument("--guid",        type=float, default=7.5)
    ap.add_argument("--image_size",  type=int, default=512)
    ap.add_argument("--ddim_steps",  type=int, default=50)
    ap.add_argument("--batch_size",  type=int, default=4)
    ap.add_argument("--smoke",       action="store_true",
                    help="quick test: 8 images per class")
    ap.add_argument("--regen",       action="store_true",
                    help="regenerate even if a class folder already has images")
    args = ap.parse_args()

    if args.smoke:
        args.n_per_class = 8

    rng       = random.Random(42)
    train_dir = os.path.join(args.root, "train")
    csv_dir   = os.path.join(_THIS_DIR, "prompts", "objectnette")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print("=" * 64)
    print(f"Building objectnette2 | root={args.root}")
    print(f"classes={args.classes}")
    print(f"n_per_class={args.n_per_class}  gpus={args.gpus}  img={args.image_size}")
    print("=" * 64)

    for cls in args.classes:
        if cls not in CLASS_NOUNS:
            print(f"[skip] unknown class '{cls}' (no prompt vocabulary)")
            continue
        class_dir = os.path.join(train_dir, cls)
        os.makedirs(class_dir, exist_ok=True)

        existing = [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg"))]
        if existing and not args.regen:
            print(f"[{cls}] {len(existing)} images exist, skipping (--regen to force)")
            continue

        out_csv = os.path.join(csv_dir, f"{cls}.csv")
        build_class_csv(cls, args.n_per_class, rng, out_csv)
        print(f"\n[{cls}] generating {args.n_per_class} images -> {class_dir}")
        launch_multigpu(out_csv, class_dir, 1, args.gpus,
                        model_path="", guidance_scale=args.guid,
                        image_size=args.image_size, ddim_steps=args.ddim_steps,
                        batch_size=args.batch_size)

    print("\nAll requested classes done.")
    print(f"Dataset: {train_dir}")
    print("Class folders:", sorted(os.listdir(train_dir)))


if __name__ == "__main__":
    main()
