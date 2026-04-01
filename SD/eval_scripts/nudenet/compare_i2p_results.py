import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--csv1", required=False,default="/storage/s25017/MUKSB/SD/eval_scripts/nudenet/SD_class_counts.csv")
parser.add_argument("--csv2", required=False,default="/storage/s25017/MUKSB/SD/eval_scripts/nudenet/pure_nudity_class_counts.csv")

parser.add_argument("--name1", default="SD v1.4")
parser.add_argument("--name2", default="Ours")

args = parser.parse_args()


# =========================
# Load CSVs
# =========================

df1 = pd.read_csv(args.csv1)
df2 = pd.read_csv(args.csv2)

# Ensure same order
df = df1.merge(df2, on="Class", suffixes=("_1", "_2"))

classes = df["Class"]

values1 = df["Count_1"]
values2 = df["Count_2"]


# =========================
# Plot
# =========================

y = np.arange(len(classes))

height = 0.35

plt.figure(figsize=(8,5))

plt.barh(
    y - height/2,
    values1,
    height,
    label=args.name1,
    color="#4C72B0"
)

plt.barh(
    y + height/2,
    values2,
    height,
    label=args.name2,
    color="#DD8452"
)

plt.yticks(y, classes)

plt.xlabel("Number of detections")

plt.title("I2P Nudity Detection Comparison")

plt.legend()

plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig("comparison_barplot.png", dpi=300)

plt.show()

print("Saved: comparison_barplot.png")