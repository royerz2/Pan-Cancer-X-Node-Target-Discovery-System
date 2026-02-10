#!/usr/bin/env python3
"""
Generate Figure 1: ALIN Framework pipeline schematic.

Panel A – Computational pipeline flowchart (left ~60%)
Panel B – Tri-axial inhibition principle  (right ~40%)

Outputs: figures/fig1_pipeline_schematic.{png,pdf}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# ── Colour palette (matches rest of paper) ───────────────────────────────────
C = {
    "data":       "#2E86AB",   # blue – data sources
    "inference":  "#A23B72",   # magenta – inference
    "mhs":        "#F18F01",   # amber – MHS
    "ranking":    "#3B7A57",   # green – ranking / scoring
    "validation": "#6C3483",   # purple – validation
    "bg":         "#FAFAFA",
    "arrow":      "#444444",
    "text":       "#222222",
    "light_bg":   "#F0F4F8",
    # tri-axial
    "upstream":   "#2E86AB",
    "downstream": "#A23B72",
    "orthogonal": "#F18F01",
    "block":      "#C0392B",
    "resist":     "#888888",
}


def _rounded_box(ax, xy, w, h, label, sublabel, color, fontsize=9, sublabel_size=7):
    """Draw a rounded rectangle with a bold title and subtitle."""
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.012",
                         facecolor=color, edgecolor="white",
                         linewidth=1.5, alpha=0.92, zorder=3)
    ax.add_patch(box)
    # Title (white, bold)
    ax.text(x + w / 2, y + h * 0.62, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4)
    # Subtitle (white, smaller)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.28, sublabel,
                ha="center", va="center", fontsize=sublabel_size,
                color="white", alpha=0.95, zorder=4,
                style="italic")


def _arrow_down(ax, x, y_from, y_to, color="#444444"):
    """Vertical arrow between boxes."""
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, shrinkA=2, shrinkB=2),
                zorder=2)


def _arrow_right(ax, xy_from, xy_to, color="#444444", lw=1.8):
    """Horizontal arrow."""
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, shrinkA=2, shrinkB=2),
                zorder=2)


def _side_arrow(ax, xy_from, xy_to, color="#444444"):
    """Curved side arrow for validation feedback."""
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, connectionstyle="arc3,rad=0.3",
                                shrinkA=3, shrinkB=3),
                zorder=2)


# ═════════════════════════════════════════════════════════════════════════════
# Panel A: Pipeline flowchart
# ═════════════════════════════════════════════════════════════════════════════
def draw_panel_a(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A", fontsize=14, fontweight="bold", loc="left", pad=8)

    # Layout constants
    bw = 0.70       # box width
    bh = 0.095      # box height
    x0 = 0.15       # left edge
    gap = 0.025     # gap between boxes
    top = 0.93      # top of first box

    boxes = [
        # (label, sublabel, color)
        ("DepMap CRISPR Essentiality",
         "Chronos scores · 1,100+ cell lines · 77 cancer types",
         C["data"]),

        ("OmniPath Signaling Network",
         "Directed interactions · pathway topology · literature-curated",
         C["data"]),

        ("Survival Mechanism Inference",
         "Essentiality + Co-essentiality + Signaling paths + Statistical dep.",
         C["inference"]),

        ("Minimal Hitting Set (MHS)",
         "Greedy set-cover · hub-gene penalty · covers all mechanisms",
         C["mhs"]),

        ("Triple Combination Ranking",
         "Synergy · resistance risk · combo-toxicity · pathway coverage",
         C["ranking"]),

        ("Multi-Source Validation",
         "PubMed · STRING · ClinicalTrials.gov · PRISM · GDSC",
         C["validation"]),
    ]

    n = len(boxes)
    positions = []
    for i, (lbl, sub, col) in enumerate(boxes):
        y = top - i * (bh + gap)
        _rounded_box(ax, (x0, y - bh), bw, bh, lbl, sub, col,
                     fontsize=9.5, sublabel_size=7.5)
        positions.append((x0 + bw / 2, y - bh, y))  # cx, y_bottom, y_top

    # Arrows between consecutive boxes
    for i in range(n - 1):
        cx_i, yb_i, yt_i = positions[i]
        cx_j, yb_j, yt_j = positions[i + 1]
        _arrow_down(ax, cx_i, yb_i, yt_j, C["arrow"])

    # Side annotations for the two data sources merging
    # Draw a bracket-like merge from boxes 0 and 1 into box 2
    mid_x_right = x0 + bw + 0.03
    y_merge_top = positions[0][1] + (positions[0][2] - positions[0][1]) / 2
    y_merge_bot = positions[1][1] + (positions[1][2] - positions[1][1]) / 2
    y_merge_mid = (y_merge_top + y_merge_bot) / 2

    # Small braces on right
    ax.plot([mid_x_right - 0.01, mid_x_right], [y_merge_top, y_merge_top],
            color=C["arrow"], lw=1.2, zorder=2)
    ax.plot([mid_x_right - 0.01, mid_x_right], [y_merge_bot, y_merge_bot],
            color=C["arrow"], lw=1.2, zorder=2)
    ax.plot([mid_x_right, mid_x_right], [y_merge_top, y_merge_bot],
            color=C["arrow"], lw=1.2, zorder=2)

    # Label on the right
    ax.text(mid_x_right + 0.015, y_merge_mid, "Data\nIntegration",
            ha="left", va="center", fontsize=7, color=C["arrow"],
            fontstyle="italic")

    # Left annotation: "4 complementary\napproaches" next to inference box
    cx_inf, yb_inf, yt_inf = positions[2]
    ax.text(x0 - 0.03, yb_inf + (yt_inf - yb_inf) / 2,
            "4 modules",
            ha="right", va="center", fontsize=7, color=C["inference"],
            fontstyle="italic", fontweight="bold")

    # Output annotation below last box
    cx_last, yb_last, yt_last = positions[-1]
    ax.text(cx_last, yb_last - 0.035,
            "→ Ranked triple combinations with multi-source evidence scores",
            ha="center", va="top", fontsize=7.5, color=C["text"],
            fontstyle="italic")


# ═════════════════════════════════════════════════════════════════════════════
# Panel B: Tri-axial inhibition principle
# ═════════════════════════════════════════════════════════════════════════════
def draw_panel_b(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("B", fontsize=14, fontweight="bold", loc="left", pad=8)

    # Three circles arranged in a triangle
    # Top circle, bottom-left, bottom-right
    cx, cy_top = 0.50, 0.82
    cx_left, cy_bot = 0.22, 0.42
    cx_right, cy_bot_r = 0.78, 0.42
    r = 0.12

    circle_data = [
        (cx, cy_top, C["upstream"], "Upstream\nDriver",
         "KRAS, BRAF,\nEGFR", "upstream"),
        (cx_left, cy_bot, C["downstream"], "Downstream\nEffector",
         "CDK4, CCND1,\nCDK6", "downstream"),
        (cx_right, cy_bot_r, C["orthogonal"], "Orthogonal\nSurvival",
         "STAT3, MCL1,\nFYN", "orthogonal"),
    ]

    for (x, y, col, lbl, genes, role) in circle_data:
        circle = plt.Circle((x, y), r, facecolor=col, edgecolor="white",
                            linewidth=2, alpha=0.88, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y + 0.025, lbl, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=4)
        ax.text(x, y - 0.045, genes, ha="center", va="center",
                fontsize=6.5, color="white", alpha=0.9, zorder=4)

    # Connecting lines between circles (edges of triangle)
    for (x1, y1), (x2, y2) in [
        ((cx, cy_top), (cx_left, cy_bot)),
        ((cx, cy_top), (cx_right, cy_bot_r)),
        ((cx_left, cy_bot), (cx_right, cy_bot_r)),
    ]:
        # Shorten lines so they don't overlap circles
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        ux, uy = dx / dist, dy / dist
        ax.plot([x1 + ux * r, x2 - ux * r],
                [y1 + uy * r, y2 - uy * r],
                color="#AAAAAA", lw=1.5, ls="--", zorder=2, alpha=0.7)

    # "BLOCK" labels on each edge midpoint
    edge_mids = [
        ((cx + cx_left) / 2 - 0.06, (cy_top + cy_bot) / 2),
        ((cx + cx_right) / 2 + 0.06, (cy_top + cy_bot_r) / 2),
        ((cx_left + cx_right) / 2, (cy_bot + cy_bot_r) / 2 - 0.06),
    ]

    # Central concept
    center_x = (cx + cx_left + cx_right) / 3
    center_y = (cy_top + cy_bot + cy_bot_r) / 3
    ax.text(center_x, center_y + 0.015, "Tri-axial\nBlockade",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=C["block"], zorder=5)

    # X marks on edges (blocking)
    for (mx, my) in edge_mids:
        ax.text(mx, my, "✕", ha="center", va="center",
                fontsize=14, color=C["block"], fontweight="bold",
                zorder=5, alpha=0.8)

    # Bottom explanatory text – two lines, well separated
    ax.text(0.50, 0.17,
            "Dual blockade → compensatory pathway shifting → resistance",
            ha="center", va="center", fontsize=7.5, color=C["resist"],
            fontstyle="italic")
    ax.text(0.50, 0.10,
            "Tri-axial blockade → all escape routes blocked → durable response",
            ha="center", va="center", fontsize=7.5, color=C["block"],
            fontweight="bold")

    # Attribution
    ax.text(0.50, 0.03, "Liaki et al. (2025, PNAS)",
            ha="center", va="center", fontsize=6.5, color="#999999")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.8),
                                      gridspec_kw={"width_ratios": [1.1, 0.9]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04,
                        wspace=0.08)

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)

    out = Path("figures")
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig1_pipeline_schematic.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✓ fig1_pipeline_schematic saved (png + pdf)")


if __name__ == "__main__":
    main()
