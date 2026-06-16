"""
Copy images from generated subdirectories into a flat testing folder,
renaming them as {class_idx}_{image_idx}.png

e.g.  00_tench/00000.png  ->  testing/0_0.png
      01_English_springer/00003.png  ->  testing/1_3.png
"""

import os
import shutil
from pathlib import Path

SRC = Path(
    "/scratch/s25017/MUKSB/SD/Evaluation/imagenette/pseudo_generated/diffusers-cls_9-MUKSB-g0.5-method_full-lr_1e-05_E3_U960_pseudo-epoch_1"
)
DST = Path("/scratch/s25017/MUKSB/SD/eval_scripts/Imagenette/cls9")

DST.mkdir(parents=True, exist_ok=True)

# Collect class dirs sorted by their numeric prefix (00, 01, ...)
class_dirs = sorted(
    [d for d in SRC.iterdir() if d.is_dir()],
    key=lambda d: d.name,
)

copied = 0
for class_idx, class_dir in enumerate(class_dirs):
    images = sorted(class_dir.glob("*.png"))
    for img_idx, img_path in enumerate(images):
        dst_name = f"{class_idx}_{img_idx}.png"
        shutil.copy2(img_path, DST / dst_name)
        copied += 1

print(f"Done. Copied {copied} images into {DST}")
print(f"Classes processed ({len(class_dirs)}):")
for i, d in enumerate(class_dirs):
    print(f"  {i}: {d.name}")
