#!/usr/bin/env python3
"""
Generate Figure 7: Translational vs Discovery target Venn diagram + comparison panel.
Produces fig7_translational_vs_discovery.pdf and .png
"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Load data ──────────────────────────────────────────────────
trans_targets = Counter()
disc_targets = Counter()

with open(ROOT / "results/triple_combinations.csv") as f:
    for row in csv.DictReader(f):
        for col in ("Target_1", "Target_2", "Target_3"):
            trans_targets[row[col].strip()] += 1

with open(ROOT / "results_discovery/triple_combinations.csv") as f:
    for row in csv.DictReader(f):
        for col in ("Target_1", "Target_2", "Target_3"):
            disc_targets[row[col].strip()] += 1

trans_set = set(trans_targets)
disc_set = set(disc_targets)
shared = trans_set & disc_set
trans_only = trans_set - disc_set
disc_only = disc_set - trans_set

# ── Figure ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.32)

# (A) Venn diagram
ax1 = fig.add_subplot(gs[0])
v = venn2(
    subsets=(len(trans_only), len(disc_only), len(shared)),
    set_labels=("", ""),
    ax=ax1,
)
# Colors
v.get_patch_by_id("10").set(facecolor="#4C72B0", alpha=0.55, edgecolor="none")
v.get_patch_by_id("01").set(facecolor="#DD8452", alpha=0.55, edgecolor="none")
v.get_patch_by_id("11").set(facecolor="#937DC2", alpha=0.7, edgecolor="none")
c = venn2_circles(subsets=(len(trans_only), len(disc_only), len(shared)), ax=ax1, linewidth=1.5)
c[0].set(edgecolor="#2E4A7A")
c[1].set(edgecolor="#A85E2F")

# Count labels with bold
for lid in ("10", "01", "11"):
    label = v.get_label_by_id(lid)
    if label:
        label.set_fontsize(18)
        label.set_fontweight("bold")

# Manual set labels below circles
ax1.text(-0.55, -0.56, "Translational\n(22 targets)", ha="center", va="top",
         fontsize=13, color="#2E4A7A", fontweight="bold")
ax1.text(0.55, -0.56, "Discovery\n(89 targets)", ha="center", va="top",
         fontsize=13, color="#A85E2F", fontweight="bold")

# Annotate key targets
# Translational-only
t_only_labels = sorted(trans_only, key=lambda x: -trans_targets[x])
ax1.text(-0.5, 0.47, "\n".join(t_only_labels[:6]),
         ha="center", va="top", fontsize=8, fontstyle="italic", color="#2E4A7A")

# Shared
shared_labels = sorted(shared, key=lambda x: -(trans_targets[x]+disc_targets[x]))
ax1.text(0.0, 0.22, "\n".join(shared_labels[:5]),
         ha="center", va="top", fontsize=7.5, fontstyle="italic", color="#4B3080")

# Discovery-only
d_only_labels = sorted(disc_only, key=lambda x: -disc_targets[x])
ax1.text(0.52, 0.47, "\n".join(d_only_labels[:6]),
         ha="center", va="top", fontsize=8, fontstyle="italic", color="#A85E2F")

ax1.set_title("(A) Target landscape", fontsize=14, fontweight="bold", pad=15)

# (B) Per-cancer agreement bar chart
ax2 = fig.add_subplot(gs[1])
categories = ["Identical\n(3/3 shared)", "Partial\n(1–2 shared)", "Different\n(0 shared)"]
values = [8, 26, 42]
colors = ["#937DC2", "#B8D4E3", "#F2C7C7"]
bars = ax2.bar(categories, values, color=colors, edgecolor="#333", linewidth=0.8, width=0.6)
for bar, val in zip(bars, values):
    pct = 100 * val / 76
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
             f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_ylabel("Cancer types (of 76)", fontsize=12)
ax2.set_ylim(0, 52)
ax2.set_title("(B) Per-cancer triple agreement", fontsize=14, fontweight="bold", pad=15)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Annotation box with key stats
stats_text = (
    "8 identical cancers:\n"
    "  Embryonal Tumor, Epithelioid Sarcoma,\n"
    "  Glassy Cell Cervix, Meningioma,\n"
    "  Mucosal Melanoma, Salivary, Sarcoma NOS,\n"
    "  Urethral Cancer\n\n"
    "Druggability gap:\n"
    "  Translational: 89.5% fully druggable\n"
    "  Discovery: 6.6% fully druggable\n"
    "  → 73 targets need drug development"
)
ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment="top", horizontalalignment="right",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#F9F9F9", edgecolor="#CCC", alpha=0.9))

fig.suptitle(
    "Translational vs. Discovery Mode: Dual-Track Target Identification",
    fontsize=15, fontweight="bold", y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"figures/fig7_translational_vs_discovery.{ext}", dpi=300,
                bbox_inches="tight")
print("Saved figures/fig7_translational_vs_discovery.pdf and .png")
