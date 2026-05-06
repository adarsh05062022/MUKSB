import os
import re

def find_missing_ranges(folder_path, extension=".png"):
    files = os.listdir(folder_path)

    pattern = re.compile(r"(\d+)_0" + re.escape(extension))

    indices = []

    for f in files:
        match = pattern.fullmatch(f)
        if match:
            indices.append(int(match.group(1)))

    if not indices:
        print("No matching files found.")
        return

    indices = sorted(indices)

    min_idx = min(indices)
    max_idx = max(indices)

    full_range = set(range(min_idx, max_idx + 1))
    missing = sorted(full_range - set(indices))

    print(f"Range: {min_idx} → {max_idx}")
    print(f"Missing count: {len(missing)}")

    # 🔥 Convert missing indices → ranges
    ranges = []
    start = prev = None

    for num in missing:
        if start is None:
            start = prev = num
        elif num == prev + 1:
            prev = num
        else:
            ranges.append((start, prev))
            start = prev = num

    if start is not None:
        ranges.append((start, prev))

    # Print ranges
    print("\nMissing ranges:")
    for r in ranges:
        if r[0] == r[1]:
            print(f"{r[0]}")
        else:
            print(f"{r[0]}–{r[1]}")

    return ranges


# 🔧 Usage
folder_path = "/storage/s25017/MUNBa/SD/eval_scripts/CLASS/COCO_30k_NSFW_E3_512"
find_missing_ranges(folder_path)