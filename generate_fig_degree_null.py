#!/usr/bin/env python3
"""
Generate Figure 7: Degree-Preserving Network Null Model.

Three panels:
  A – Edge-swap schematic (original → rewired, degrees unchanged)
  B – STAT3 null-distribution histogram vs observed
  C – Gene frequency: observed (real network) vs null (edge-swap)

Data: ablation_results/degree_null_results.json
Output: figures/fig_degree_null.{png,pdf}
"""

import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Colours ──────────────────────────────────────────────────────────────────
RED   = "#C0392B"
BLUE  = "#2E86AB"
LBLUE = "#A8D5E2"
DBLUE = "#1A5276"
GREY  = "#888888"
DGREY = "#333333"
ORANGE = "#E67E22"
GREEN = "#27AE60"


# ─────────────────────────────────────────────────────────────────────────────
# PANEL A  –  Edge-swap schematic
# ─────────────────────────────────────────────────────────────────────────────
def _node(ax, x, y, label, r=0.06):
    """Draw a single labelled node."""
    c = plt.Circle((x, y), r, fc=BLUE, ec=DBLUE, lw=1.8, zorder=5,
                   clip_on=False)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=9,
            fontweight="bold", color="white", zorder=6, clip_on=False)


def _arrow(ax, x1, y1, x2, y2, r=0.06, color=DGREY, lw=1.3, alpha=0.55):
    """Draw directed edge shortened by node radius."""
    dx, dy = x2 - x1, y2 - y1
    d = np.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    ax.annotate(
        "", xy=(x2 - ux * r, y2 - uy * r),
        xytext=(x1 + ux * r, y1 + uy * r),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
        alpha=alpha, zorder=3, clip_on=False,
    )


def draw_panel_a(ax):
    """Edge-swap schematic: two small directed graphs side by side."""
    ax.set_xlim(-0.15, 2.65)
    ax.set_ylim(-0.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Left graph positions ─────────────────────────────────────────────
    L = {
        "A": (0.0, 1.0), "B": (0.6, 1.0),
        "C": (0.0, 0.5), "D": (0.6, 0.5),
        "E": (0.0, 0.0), "F": (0.6, 0.0),
    }
    edges_L = [("A","C"), ("A","D"), ("B","D"), ("C","E"), ("D","F"), ("E","F")]
    swap_set = {("A","D"), ("C","E")}

    for n, (x, y) in L.items():
        _node(ax, x, y, n)
    for u, v in edges_L:
        col = RED if (u, v) in swap_set else DGREY
        alp = 1.0 if (u, v) in swap_set else 0.55
        lw  = 2.0 if (u, v) in swap_set else 1.3
        _arrow(ax, *L[u], *L[v], color=col, lw=lw, alpha=alp)

    ax.text(0.30, 1.30, "Original Network", ha="center", fontsize=10,
            fontweight="bold", color=DGREY)
    ax.text(0.42, 0.82, "A\u2192D", fontsize=7, color=RED, fontweight="bold",
            rotation=-40)
    ax.text(-0.12, 0.30, "C\u2192E", fontsize=7, color=RED, fontweight="bold",
            rotation=-40)

    # ── Swap arrow ───────────────────────────────────────────────────────
    ax.annotate(
        "", xy=(1.48, 0.50), xytext=(0.95, 0.50),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.8),
        zorder=2, clip_on=False,
    )
    ax.text(1.21, 0.62, "Edge Swap", ha="center", fontsize=9.5,
            fontweight="bold", color=ORANGE)
    ax.text(1.21, 0.38, "Degree preserved", ha="center", fontsize=7.5,
            color=ORANGE, fontstyle="italic")

    # ── Right graph positions ────────────────────────────────────────────
    ox = 1.65  # horizontal offset for right graph
    R = {k: (x + ox, y) for k, (x, y) in L.items()}
    edges_R = [("A","C"), ("A","E"), ("B","D"), ("C","D"), ("D","F"), ("E","F")]
    new_set = {("A","E"), ("C","D")}

    for n, (x, y) in R.items():
        _node(ax, x, y, n)
    for u, v in edges_R:
        col = RED if (u, v) in new_set else DGREY
        alp = 1.0 if (u, v) in new_set else 0.55
        lw  = 2.0 if (u, v) in new_set else 1.3
        _arrow(ax, *R[u], *R[v], color=col, lw=lw, alpha=alp)

    ax.text(0.30 + ox, 1.30, "Rewired Network", ha="center", fontsize=10,
            fontweight="bold", color=DGREY)
    ax.text(-0.12 + ox, 0.30, "A\u2192E", fontsize=7, color=RED,
            fontweight="bold", rotation=-40)
    ax.text(0.42 + ox, 0.57, "C\u2192D", fontsize=7, color=RED,
            fontweight="bold")

    # ── Degree table ─────────────────────────────────────────────────────
    ax.text(1.25, -0.22, "Node degrees unchanged:", ha="center",
            fontsize=8, fontweight="bold", color=DGREY)
    ax.text(1.25, -0.35,
            "A: out=2   B: out=1   C: out=1\n"
            "D: in=2    E: in=1    F: in=2",
            ha="center", fontsize=7, color=GREY, family="monospace")
    ax.text(1.25, -0.52,
            "\u00d710|E| swaps per permutation, 20 permutations total",
            ha="center", fontsize=7, color=GREY, fontstyle="italic")


# ─────────────────────────────────────────────────────────────────────────────
# PANEL B  –  STAT3 null distribution
# ─────────────────────────────────────────────────────────────────────────────
def draw_panel_b(ax, data):
    null_dist = data["null_stat3_distribution"]
    observed  = data["observed_stat3_freq"]
    p_val     = data["p_value"]
    null_mean = data["null_mean"]
    null_std  = data["null_std"]

    bins = np.arange(-0.5, 6.5, 1)
    counts, _, _ = ax.hist(null_dist, bins=bins, color=LBLUE, edgecolor=BLUE,
                           lw=1.2, alpha=0.85, zorder=3,
                           label=f"Null ({len(null_dist)} permutations)")

    ax.axvline(observed, color=RED, lw=2.5, ls="--", zorder=5,
               label=f"Observed ({observed}/5 cancers)")
    ax.axvspan(observed - 0.3, 5.5, alpha=0.07, color=RED, zorder=1)

    # p-value box
    ptxt = "p < 0.05" if p_val == 0.0 else f"p = {p_val:.3f}"
    ax.text(0.96, 0.94, ptxt, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=RED,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, alpha=0.9))

    # observed callout
    ax.annotate(
        f"Observed: {observed}/5",
        xy=(observed, max(counts) * 0.65),
        xytext=(observed - 1.5, max(counts) * 0.88),
        fontsize=9, fontweight="bold", color=RED,
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4), zorder=6,
    )

    # null stats
    ax.text(0.96, 0.78, f"Null: {null_mean:.1f} \u00b1 {null_std:.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", fc=LBLUE, ec=BLUE, alpha=0.45))

    ax.set_xlabel("STAT3 frequency (of 5 test cancers)", fontsize=10)
    ax.set_ylabel("Number of permutations", fontsize=10)
    ax.set_xlim(-0.8, 5.8)
    ax.set_xticks(range(6))
    ax.set_yticks(range(0, int(max(counts)) + 3, 2))
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL C  –  Gene frequency comparison
# ─────────────────────────────────────────────────────────────────────────────
def draw_panel_c(ax, data):
    obs  = data["observed_all_targets"]       # out of 5 cancers
    null = data["top_null_genes"]              # out of 100 cancer-perm pairs

    genes = ["STAT3", "CCND1", "CDK4", "CDK6", "PIK3CA", "KRAS", "CTNNB1",
             "BRAF"]
    obs_pct  = [obs.get(g, 0)  / 5   * 100 for g in genes]
    null_pct = [null.get(g, 0) / 100 * 100 for g in genes]

    x = np.arange(len(genes))
    w = 0.34

    ax.bar(x - w / 2, null_pct, w, color=LBLUE, edgecolor=BLUE,
           lw=1.0, alpha=0.85, label="Null (edge-swap)", zorder=3)
    gene_cols = [RED if g == "STAT3" else BLUE if g == "CCND1"
                 else GREEN if g in ("CDK4", "CDK6") else GREY
                 for g in genes]
    bars_obs = ax.bar(x + w / 2, obs_pct, w, color=gene_cols,
                      edgecolor="white", lw=1.0, alpha=0.90,
                      label="Observed (real network)", zorder=3)

    # Value labels
    for i, (nv, ov) in enumerate(zip(null_pct, obs_pct)):
        if nv > 0:
            ax.text(i - w / 2, nv + 1.5, f"{nv:.0f}%", ha="center",
                    va="bottom", fontsize=6.5, color=BLUE)
        if ov > 0:
            ax.text(i + w / 2, ov + 1.5, f"{ov:.0f}%", ha="center",
                    va="bottom", fontsize=6.5, color=gene_cols[i],
                    fontweight="bold")

    ax.set_xlabel("Gene", fontsize=10)
    ax.set_ylabel("Frequency (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, fontsize=8.5, rotation=30, ha="right")
    ax.set_ylim(0, 118)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Insight callout
    ax.annotate("STAT3: 100% observed\nvs. 12% in null",
                xy=(x[0] + w / 2, obs_pct[0]),
                xytext=(x[2], 105),
                fontsize=8, fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3,
                                connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED,
                          alpha=0.9),
                zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    data_path = Path("ablation_results/degree_null_results.json")
    with open(data_path) as f:
        data = json.load(f)

    fig = plt.figure(figsize=(16, 5.5), facecolor="white")

    # Top-level grid: Panel A (left third) | Panels B+C (right two-thirds)
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.15, 2],
                              left=0.02, right=0.98, top=0.90, bottom=0.02,
                              wspace=0.05)

    ax_a = fig.add_subplot(outer[0, 0])
    draw_panel_a(ax_a)
    ax_a.text(-0.05, 1.05, "A", transform=ax_a.transAxes,
              fontsize=15, fontweight="bold", va="top")

    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1],
                                            wspace=0.35)

    ax_b = fig.add_subplot(inner[0, 0])
    draw_panel_b(ax_b, data)
    ax_b.text(-0.15, 1.05, "B", transform=ax_b.transAxes,
              fontsize=15, fontweight="bold", va="top")

    ax_c = fig.add_subplot(inner[0, 1])
    draw_panel_c(ax_c, data)
    ax_c.text(-0.15, 1.05, "C", transform=ax_c.transAxes,
              fontsize=15, fontweight="bold", va="top")

    fig.suptitle("Degree-Preserving Network Null Model",
                 fontsize=14, fontweight="bold", y=0.98, color=DGREY)

    out = Path("figures")
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_degree_null.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\u2713 fig_degree_null saved (png + pdf)")


if __name__ == "__main__":
    main()
