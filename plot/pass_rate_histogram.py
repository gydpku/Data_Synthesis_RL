import matplotlib.pyplot as plt
import numpy as np
from pass_rate_data import (
    gsm8k_human, gsm8k_synth_before, gsm8k_synth_after,
    logiqa_human, logiqa_synth_before, logiqa_synth_after,
    medqa_human,  medqa_synth_before,  medqa_synth_after
)

plt.rcParams.update({
    "figure.dpi"     : 300,
    "font.family"    : "DejaVu Sans",
    "axes.facecolor" : "#FAFAFA",
    "grid.color"     : "#DDDDDD",
    "grid.linestyle" : "--",
    "grid.alpha"     : 0.4
})

palette = {"Human": "#3778BF", "Synth Before": "#D65F5F", "Synth After": "#4C9F70"}

def draw_hist(ax, data, label):
    bins = np.linspace(0, 1, 21)
    color = palette[label]
    ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.55, linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.grid(True)

    # Set x-axis tick labels to show 0, 0.2, 0.4, 0.6, 0.8, and 1
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=13)

    # Set y-axis tick labels font size to 12
    ax.tick_params(axis='y', labelsize=11)

    # Dark frame around the subplot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")   # dark grey/black
        spine.set_linewidth(1.2)

    # Add a legend for the condition
    ax.legend([label], loc='upper right', frameon=False, fontsize=13)

rows = ["Human", "Synth Before", "Synth After"]
cols = ["GSM8K", "LogiQA", "MedQA"]
data_mat = [
    (gsm8k_human,        logiqa_human,        medqa_human),
    (gsm8k_synth_before, logiqa_synth_before, medqa_synth_before),
    (gsm8k_synth_after,  logiqa_synth_after,  medqa_synth_after)
]

# ── build the grid with minimal gaps ──
fig, axes = plt.subplots(
    3, 3, figsize=(16, 5),
    sharex=True, sharey=True,
    gridspec_kw={"wspace": 0.05, "hspace": 0.05}  # tighten width/height gaps
)

for r in range(3):
    for c in range(3):
        draw_hist(axes[r, c], data_mat[r][c], rows[r])

# Headers
for c, name in enumerate(cols):
    axes[0, c].set_title(name, pad=8, weight="bold", fontsize=14)
# for r, name in enumerate(rows):
#     axes[r, 0].annotate(name, xy=(-0.12, 0.5), xycoords="axes fraction",
#                         ha="right", va="center", fontsize=12, rotation=90, weight="bold")

fig.text(0.5, 0.02, "Pass Rate", ha="center", fontsize=14, weight="bold")

# Ensure the y-label 'Count' is included
fig.text(0.085, 0.5, "Count",  va="center",  fontsize=14, weight="bold", rotation=90)

fig.savefig("pass_rate_histograms.png", bbox_inches="tight", facecolor="white")
plt.close()
