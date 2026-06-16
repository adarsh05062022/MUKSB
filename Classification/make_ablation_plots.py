import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

methods = {
    "MUKSB":          "results_ablation_direction/MUKSB/cifar10/random_4500_bs512/output/seed1/epoch_metrics.json",
    "MUKSB_RawSum":   "results_ablation_direction/MUKSB_RawSum/cifar10/random_4500_bs512/output/seed1/epoch_metrics.json",
    "MUKSB_MeanUnit": "results_ablation_direction/MUKSB_MeanUnit/cifar10/random_4500_bs512/output/seed1/epoch_metrics.json",
}

colors = {
    "MUKSB":          "#1f77b4",
    "MUKSB_RawSum":   "#ff7f0e",
    "MUKSB_MeanUnit": "#2ca02c",
}
labels = {
    "MUKSB":          "MUKSB (full)",
    "MUKSB_RawSum":   "Var A: RawSum",
    "MUKSB_MeanUnit": "Var B: MeanUnit",
}

data = {}
base = os.path.dirname(__file__)
for name, rel in methods.items():
    path = os.path.join(base, rel)
    with open(path) as f:
        data[name] = json.load(f)

splits = list(data["MUKSB"][0]["accuracy"].keys())
split_titles = {
    "retain": "Retain Accuracy",
    "forget": "Forget Accuracy",
    "val":    "Validation Accuracy",
    "test":   "Test Accuracy",
}

out_dir = os.path.join(base, "results_ablation_direction", "plots")
os.makedirs(out_dir, exist_ok=True)

for split in splits:
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, metrics in data.items():
        epochs = [m["epoch"] for m in metrics]
        accs   = [m["accuracy"].get(split, float("nan")) for m in metrics]
        ax.plot(epochs, accs, marker="o", markersize=3, linewidth=1.8,
                color=colors[name], label=labels[name])

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Ablation 1 — {split_titles.get(split, split)}\n"
        f"(CIFAR-10, 10% random forgetting, seed 1)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    out_path = os.path.join(out_dir, f"ablation_direction_{split}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
