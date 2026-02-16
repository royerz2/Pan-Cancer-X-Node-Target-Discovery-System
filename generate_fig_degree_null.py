#!/usr/bin/env python3
"""
Generate degree-preserving network null figure.

Panel A – Edge-swap schematic: original graph → degree-preserving rewiring
Panel B – STAT3 observed vs. null distribution across 5 test cancers
Panel C – Gene frequency: observed (real network) vs. null (shuffled)

Data source: ablation_results/degree_null_results.json

Outputs: figures/fig_degree_null.{png,pdf}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np
from pathlib import Path
import json

# ── Colour palette ───────────────────────────────────────────────────────────
C = {
    "observed":   "#C0392B",   # red – observed
    "null":       "#7FB3D8",   # light blue – null distribution
    "null_edge":  "#2E86AB",   # blue – null bar edges
    "node":       "#2E86AB",   # node fill
    "node_edge":  "#1A5276",   # node border
    "edge_orig":  "#333333",   # original edges
    "edge_new":   "#E74C3C",   # swapped edges (red)
    "edge_fade":  "#BBBBBB",   # faded edges
    "swap_arrow": "#E67E22",   # orange swap indicator
    "bg":         "#FFFFFF",
    "text":       "#222222",
    "stat3":      "#E74C3C",   # STAT3 highlight
    "ccnd1":      "#2E86AB",   # CCND1
    "cdk4":       "#3B7A57",   # CDK4
    "other":      "#AAAAAA",   # other genes
}


# ═════════════════════════════════════════════════════════════════════════════
# Panel A: Edge-swap schematic
# ═════════════════════════════════════════════════════════════════════════════
def draw_panel_a(ax):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    ax.set_title("A", fontsize=14, fontweight="bold", loc="left", pad=8)

    # --- Left: Original graph ---
    ax.text(0.17, 0.97, "Original Network", ha="center", va="top",
            fontsize=9, fontweight="bold", color=C["text"])

    # Node positions – compact networks with clear gap for swap arrow
    nodes_left = {
        "A": (0.05, 0.78), "B": (0.29, 0.78),
        "C": (0.05, 0.55), "D": (0.29, 0.55),
        "E": (0.05, 0.32), "F": (0.29, 0.32),
    }

    # Edges (directed): u→v
    edges_orig = [
        ("A", "C"), ("A", "D"),
        ("B", "D"),
        ("C", "E"), ("D", "F"),
        ("E", "F"),
    ]

    # Highlight the two edges to be swapped
    swap_edges = [("A", "D"), ("C", "E")]  # will become A→E, C→D

    def draw_graph(ax, nodes, edges, highlight_edges=None, highlight_color=None,
                   node_labels=True):
        nr = 0.032
        for name, (x, y) in nodes.items():
            circle = plt.Circle((x, y), nr, facecolor=C["node"],
                                edgecolor=C["node_edge"], linewidth=1.5,
                                zorder=5, alpha=0.9)
            ax.add_patch(circle)
            if node_labels:
                ax.text(x, y, name, ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white", zorder=6)

        for (u, v) in edges:
            x1, y1 = nodes[u]
            x2, y2 = nodes[v]
            dx, dy = x2 - x1, y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            ux, uy = dx / dist, dy / dist

            is_highlight = highlight_edges and (u, v) in highlight_edges
            color = highlight_color if is_highlight else C["edge_orig"]
            lw = 2.2 if is_highlight else 1.4
            ls = "-" if is_highlight else "-"
            alpha = 1.0 if is_highlight else 0.6

            ax.annotate("", xy=(x2 - ux * nr, y2 - uy * nr),
                        xytext=(x1 + ux * nr, y1 + uy * nr),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=lw, alpha=alpha),
                        zorder=4 if is_highlight else 3)

    draw_graph(ax, nodes_left, edges_orig,
               highlight_edges=swap_edges, highlight_color=C["edge_new"])

    # Label the highlighted edges
    ax.text(0.23, 0.69, "A→D", ha="center", va="center", fontsize=7,
            color=C["edge_new"], fontweight="bold", rotation=-35)
    ax.text(-0.01, 0.45, "C→E", ha="center", va="center", fontsize=7,
            color=C["edge_new"], fontweight="bold", rotation=-35)

    # --- Arrow in between ---
    ax.annotate("", xy=(0.62, 0.55), xytext=(0.40, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=C["swap_arrow"],
                                lw=2.5, shrinkA=5, shrinkB=5),
                zorder=2)
    ax.text(0.51, 0.61, "Edge Swap", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=C["swap_arrow"])
    ax.text(0.51, 0.49, "Degree preserved", ha="center", va="top",
            fontsize=7.5, color=C["swap_arrow"], fontstyle="italic")

    # --- Right: Rewired graph ---
    ax.text(0.83, 0.97, "Rewired Network", ha="center", va="top",
            fontsize=9, fontweight="bold", color=C["text"])

    nodes_right = {
        "A": (0.71, 0.78), "B": (0.95, 0.78),
        "C": (0.71, 0.55), "D": (0.95, 0.55),
        "E": (0.71, 0.32), "F": (0.95, 0.32),
    }

    # After swap: A→D becomes A→E, C→E becomes C→D
    edges_rewired = [
        ("A", "C"), ("A", "E"),   # A→D became A→E
        ("B", "D"),
        ("C", "D"),               # C→E became C→D
        ("D", "F"),
        ("E", "F"),
    ]
    new_edges = [("A", "E"), ("C", "D")]

    draw_graph(ax, nodes_right, edges_rewired,
               highlight_edges=new_edges, highlight_color=C["edge_new"])

    ax.text(0.65, 0.56, "A→E", ha="center", va="center", fontsize=7,
            color=C["edge_new"], fontweight="bold", rotation=-35)
    ax.text(0.88, 0.57, "C→D", ha="center", va="center", fontsize=7,
            color=C["edge_new"], fontweight="bold", rotation=0)

    # Degree table
    ax.text(0.50, 0.18, "Node degrees unchanged:", ha="center", va="top",
            fontsize=8, fontweight="bold", color=C["text"])
    deg_text = ("A: out=2  B: out=1  C: out=1\n"
                "D: in=2   E: in=1   F: in=2")
    ax.text(0.50, 0.11, deg_text, ha="center", va="top",
            fontsize=7, color="#555555", family="monospace")

    # Bottom caption
    ax.text(0.50, 0.01,
            "×10|E| swaps per permutation, 20 permutations total",
            ha="center", va="bottom", fontsize=7, color="#888888",
            fontstyle="italic")


# ═════════════════════════════════════════════════════════════════════════════
# Panel B: STAT3 observed vs. null distribution
# ═════════════════════════════════════════════════════════════════════════════
def draw_panel_b(ax):
    # Load data
    data_path = Path("ablation_results/degree_null_results.json")
    with open(data_path) as f:
        data = json.load(f)

    null_dist = data["null_stat3_distribution"]
    observed = data["observed_stat3_freq"]
    n_perm = data["n_permutations"]

    ax.set_title("B", fontsize=14, fontweight="bold", loc="left", pad=8)

    # Histogram of null distribution
    bins = np.arange(-0.5, 6.5, 1)
    counts, edges, patches = ax.hist(null_dist, bins=bins, color=C["null"],
                                      edgecolor=C["null_edge"], linewidth=1.2,
                                      alpha=0.8, zorder=3, label="Null (20 permutations)")

    # Observed line
    ax.axvline(observed, color=C["observed"], linewidth=2.5, linestyle="--",
               zorder=5, label=f"Observed ({observed}/5 cancers)")

    # Shade the observed region
    ax.axvspan(observed - 0.3, 5.5, alpha=0.08, color=C["observed"], zorder=1)

    # Annotations
    ax.annotate(f"Observed: {observed}/5",
                xy=(observed, max(counts) * 0.7),
                xytext=(observed - 1.2, max(counts) * 0.9),
                fontsize=9, fontweight="bold", color=C["observed"],
                arrowprops=dict(arrowstyle="-|>", color=C["observed"], lw=1.5),
                zorder=6)

    # p-value annotation
    p_val = data["p_value"]
    p_text = f"p < 0.05" if p_val == 0.0 else f"p = {p_val:.3f}"
    ax.text(0.97, 0.95, p_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=C["observed"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C["observed"], alpha=0.9))

    # Null mean annotation
    null_mean = data["null_mean"]
    null_std = data["null_std"]
    ax.text(0.97, 0.82,
            f"Null: {null_mean:.1f} ± {null_std:.1f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=C["null_edge"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["null"],
                      edgecolor=C["null_edge"], alpha=0.4))

    ax.set_xlabel("STAT3 frequency (of 5 test cancers)", fontsize=10)
    ax.set_ylabel("Number of permutations", fontsize=10)
    ax.set_xlim(-0.8, 5.8)
    ax.set_xticks(range(6))
    ax.set_yticks(range(0, int(max(counts)) + 3, 2))
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ═════════════════════════════════════════════════════════════════════════════
# Panel C: Gene frequency — observed vs. null
# ═════════════════════════════════════════════════════════════════════════════
def draw_panel_c(ax):
    data_path = Path("ablation_results/degree_null_results.json")
    with open(data_path) as f:
        data = json.load(f)

    # Observed: frequency out of 5 cancers
    obs_targets = data["observed_all_targets"]
    # Null: frequency out of 100 cancer-permutation pairs (20 perm × 5 cancers)
    null_genes = data["top_null_genes"]

    # Select genes to show (top null + all observed)
    show_genes = ["STAT3", "CCND1", "CDK4", "CDK6", "PIK3CA", "KRAS",
                  "CTNNB1", "BRAF"]

    obs_vals = []
    null_vals = []
    colors = []
    for g in show_genes:
        obs_vals.append(obs_targets.get(g, 0) / 5 * 100)   # % of 5 cancers
        null_vals.append(null_genes.get(g, 0) / 100 * 100)  # % of 100 pairs
        if g == "STAT3":
            colors.append(C["stat3"])
        elif g == "CCND1":
            colors.append(C["ccnd1"])
        elif g in ("CDK4", "CDK6"):
            colors.append(C["cdk4"])
        else:
            colors.append(C["other"])

    ax.set_title("C", fontsize=14, fontweight="bold", loc="left", pad=8)

    x = np.arange(len(show_genes))
    width = 0.35

    # Null bars
    bars_null = ax.bar(x - width / 2, null_vals, width, color=C["null"],
                       edgecolor=C["null_edge"], linewidth=1.0, alpha=0.8,
                       label="Null (edge-swap)", zorder=3)

    # Observed bars
    bars_obs = ax.bar(x + width / 2, obs_vals, width, color=colors,
                      edgecolor="white", linewidth=1.0, alpha=0.9,
                      label="Observed (real network)", zorder=3)

    # Value labels on bars
    for bar, val in zip(bars_null, null_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=6.5,
                    color=C["null_edge"])

    for bar, val, col in zip(bars_obs, obs_vals, colors):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=6.5,
                    color=col, fontweight="bold")

    ax.set_xlabel("Gene", fontsize=10)
    ax.set_ylabel("Frequency (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(show_genes, fontsize=8.5, rotation=30, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Key insight annotation
    ax.annotate("STAT3: 100% observed\nvs. 12% in null",
                xy=(x[0] + width / 2, obs_vals[0]),
                xytext=(x[1] + 0.5, 90),
                fontsize=8, fontweight="bold", color=C["stat3"],
                arrowprops=dict(arrowstyle="-|>", color=C["stat3"],
                                lw=1.3, connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C["stat3"], alpha=0.9),
                zorder=6)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    fig = plt.figure(figsize=(16, 5.8))
    fig.patch.set_facecolor("white")

    # Layout: Panel A (left, schematic), Panel B (center, histogram),
    #         Panel C (right, bar chart)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.9, 1.0],
                          left=0.04, right=0.96, top=0.88, bottom=0.13,
                          wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)

    # Suptitle
    fig.suptitle("Degree-Preserving Network Null Model",
                 fontsize=13, fontweight="bold", y=0.97, color="#333333")

    out = Path("figures")
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_degree_null.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✓ fig_degree_null saved (png + pdf)")


if __name__ == "__main__":
    main()
