import os
import time
import subprocess
import sys

# ─────────────────────────────────────────────
#  CONFIGURE THESE BEFORE RUNNING
# ─────────────────────────────────────────────

# Python binary from your munba3 conda env
PYTHON_BIN = "/storage/s25017/miniconda3/envs/munba3/bin/python"

# Folder where your COCO images are being generated
IMAGE_FOLDER = "/scratch/s25017/MUKSB/SD/eval_scripts/CLASS/COCO_30K_SD"

# Minimum image count before pipeline kicks off
IMAGE_THRESHOLD = 29000

# How often (seconds) to re-check the image count
POLL_INTERVAL = 30

# Total number of prompts for i2p generation
TOTAL_PROMPTS = 4703

# Number of GPUs to use
NUM_GPUS = 8

# Path to the i2p generation script (relative to where you run this pipeline)
I2P_SCRIPT = "eval_scripts/generate_images_i2p.py"

# Extra args for generate_images_i2p.py (add yours here)
# Example: "--model_path checkpoints/model.pt --prompts_path i2p.csv --outdir ./i2p_out"
I2P_EXTRA_ARGS = ""

# ─────────────────────────────────────────────


def count_images(folder):
    """Count image files in the given folder."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    try:
        return sum(
            1 for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        )
    except FileNotFoundError:
        print(f"[ERROR] Folder not found: {folder}")
        return 0


def wait_for_images():
    """Poll IMAGE_FOLDER until image count exceeds IMAGE_THRESHOLD."""
    print(f"\n{'='*60}")
    print(f"[STEP 1] Waiting for images in: {IMAGE_FOLDER}")
    print(f"         Threshold: {IMAGE_THRESHOLD} images")
    print(f"{'='*60}\n")

    while True:
        count = count_images(IMAGE_FOLDER)
        print(f"  Current image count: {count} / {IMAGE_THRESHOLD}", end="\r", flush=True)

        if count > IMAGE_THRESHOLD:
            print(f"\n  ✓ Threshold reached! ({count} images found)")
            break

        time.sleep(POLL_INTERVAL)


def run_training():
    """Run the training module and wait for it to finish."""
    print(f"\n{'='*60}")
    print(f"[STEP 2] Starting training: python -m experiments.MUKSB_nsfw_one")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [PYTHON_BIN, "-m", "experiments.MUKSB_nsfw_one"],
        check=False
    )

    if result.returncode != 0:
        print(f"\n[ERROR] Training failed with exit code {result.returncode}. Aborting.")
        sys.exit(result.returncode)

    print(f"\n  ✓ Training complete!")


def build_gpu_ranges(total, num_gpus):
    """Divide total prompts evenly across GPUs, returns list of (from, to) tuples."""
    chunk = total // num_gpus
    ranges = []
    for i in range(num_gpus):
        from_case = i * chunk
        # Last GPU takes any remainder
        to_case = (i + 1) * chunk - 1 if i < num_gpus - 1 else total - 1
        ranges.append((from_case, to_case))
    return ranges


def run_i2p_generation():
    """Launch generate_images_i2p.py in parallel across all GPUs."""
    print(f"\n{'='*60}")
    print(f"[STEP 3] Launching i2p generation across {NUM_GPUS} GPUs")
    print(f"         Total prompts: {TOTAL_PROMPTS}")
    print(f"{'='*60}\n")

    ranges = build_gpu_ranges(TOTAL_PROMPTS, NUM_GPUS)
    processes = []

    for gpu_id, (from_case, to_case) in enumerate(ranges):
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} {PYTHON_BIN} {I2P_SCRIPT} "
            f"--from_case {from_case} --to_case {to_case}"
        )
        if I2P_EXTRA_ARGS.strip():
            cmd += f" {I2P_EXTRA_ARGS.strip()}"

        print(f"  GPU {gpu_id}: cases {from_case:>6} → {to_case:>6}  |  {cmd}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        proc = subprocess.Popen(
            [PYTHON_BIN, I2P_SCRIPT,
             "--from_case", str(from_case),
             "--to_case", str(to_case)]
            + (I2P_EXTRA_ARGS.split() if I2P_EXTRA_ARGS.strip() else []),
            env=env
        )
        processes.append((gpu_id, proc))

    print(f"\n  All {NUM_GPUS} workers launched. Waiting for completion...\n")

    # Wait for all processes and report results
    failed = []
    for gpu_id, proc in processes:
        proc.wait()
        status = "✓ done" if proc.returncode == 0 else f"✗ FAILED (code {proc.returncode})"
        print(f"  GPU {gpu_id}: {status}")
        if proc.returncode != 0:
            failed.append(gpu_id)

    if failed:
        print(f"\n[WARNING] Generation failed on GPU(s): {failed}")
    else:
        print(f"\n  ✓ All GPU workers finished successfully!")


def main():
    print("\n" + "="*60)
    print("  PIPELINE: COCO Watch → Train → i2p Generation")
    print("="*60)

    # wait_for_images()
    # run_training()
    run_i2p_generation()

    print(f"\n{'='*60}")
    print("  ✓ PIPELINE COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()