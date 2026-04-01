import os
import subprocess

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PYTHON_BIN = "/storage/s25017/miniconda3/envs/munba3/bin/python"
I2P_SCRIPT = "eval_scripts/generate_images_i2p.py"

# Your custom GPU list
GPU_LIST = [0,1,2,3,4,6,7]

# Your custom case range
FROM_CASE = 4703
TO_CASE = 29999   # inclusive

# Extra args if needed
I2P_EXTRA_ARGS = ""


# ─────────────────────────────────────────────


def split_range(from_case, to_case, num_splits):
    """Split [from_case, to_case] evenly into num_splits parts."""
    total = to_case - from_case + 1
    chunk = total // num_splits

    ranges = []
    start = from_case

    for i in range(num_splits):
        end = start + chunk - 1

        # last GPU takes remainder
        if i == num_splits - 1:
            end = to_case

        ranges.append((start, end))
        start = end + 1

    return ranges


def run_custom_i2p():
    print("\n" + "="*60)
    print(" CUSTOM GPU PARALLEL I2P GENERATION ")
    print("="*60)

    num_gpus = len(GPU_LIST)
    ranges = split_range(FROM_CASE, TO_CASE, num_gpus)

    processes = []

    for idx, (gpu_id, (from_case, to_case)) in enumerate(zip(GPU_LIST, ranges)):
        print(f"GPU {gpu_id}: cases {from_case} → {to_case}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            PYTHON_BIN, I2P_SCRIPT,
            "--from_case", str(from_case),
            "--to_case", str(to_case)
        ]

        if I2P_EXTRA_ARGS.strip():
            cmd += I2P_EXTRA_ARGS.split()

        proc = subprocess.Popen(cmd, env=env)
        processes.append((gpu_id, proc))

    print("\nAll jobs launched. Waiting...\n")

    failed = []
    for gpu_id, proc in processes:
        proc.wait()
        if proc.returncode == 0:
            print(f"GPU {gpu_id}: ✓ done")
        else:
            print(f"GPU {gpu_id}: ✗ FAILED ({proc.returncode})")
            failed.append(gpu_id)

    if failed:
        print(f"\nFailed GPUs: {failed}")
    else:
        print("\n✓ All jobs completed successfully!")


if __name__ == "__main__":
    run_custom_i2p()