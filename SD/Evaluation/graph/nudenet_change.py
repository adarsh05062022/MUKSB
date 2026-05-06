import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------
# ENTER YOUR COUNTS HERE
# ---------------------------------
data = {
    "BodyPart": [
        "Male_Genitalia",
        "Belly",
        "Armpits",
        "Feet",
        "Male_Breast",
        "Female_Genitalia",
        "Female_Breast",
        "Buttocks"
    ],

    "SD":    [7,131,88,13,12,8,160,21],

    "ESD":   [3,20,31,7,4,1,16,3],
    # "SA": [5,1,1,0,0,2,2,0],
    "SalUn": [5,1,1,0,0,2,2,0],
    "MUNBa":   [3,18,11,6,2,1,6,1],
    "Ours":  [0,1,1,0,0,0,0,1],
}

df = pd.DataFrame(data)

# ---------------------------------
# Compute % change from SD
# ---------------------------------
methods = ["ESD","MUNBa","SalUn","Ours"]

for m in methods:
    df[m] = ((df[m] - df["SD"]) / df["SD"]) * 100

# ---------------------------------
# Plot
# ---------------------------------
colors = ['#4C78A8','#F28E2B','#72B7B2','#9C755F',
          '#FF9DA6','#B279A2','#EDC948','#BAB0AC']

fig, axes = plt.subplots(1,4, figsize=(25,5), sharey=True)

y = np.arange(len(df))

for ax, method in zip(axes, methods):

    for i in range(len(df)):
        ax.barh(i, df.loc[i, method], color=colors[i], height=0.72)

    ax.set_xlim(-100,0)
    ax.set_xticks([-100,-75,-50,-25,0])

    ax.set_title(f"I2P - {method}",fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlabel("% Change from SD v1.4",fontsize=15, fontweight='bold')

axes[0].set_yticks(y)
axes[0].set_yticklabels(
    [f"{bp} ({sd})" for bp,sd in zip(df["BodyPart"], df["SD"])],
    fontsize=15,
    fontweight='bold'
)
axes[0].invert_yaxis()

for ax in axes[1:]:
    ax.tick_params(left=False,labelleft=False)

plt.savefig("results/bodypart_unlearning_graph.png", dpi=300, bbox_inches='tight')
plt.close()