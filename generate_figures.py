#!/usr/bin/env python3
"""
Generate publication-quality figures for ALIN Framework (Adaptive Lethal Intersection Network).
Uses SciencePlots + seaborn for Nature/Science-style aesthetics.

Reference: Liaki V, Barrambana S, et al. 2025. A targeted combination therapy achieves effective
pancreatic cancer regression and prevents tumor resistance. bioRxiv doi: 10.1101/2025.08.04.668325.
"""

import pandas as pd
from pathlib import Path
import json

LIAKI_ET_AL_CITATION = (
    "Liaki V, Barrambana S, Kostopoulou M, Lechuga CG, et al. 2025. "
    "A targeted combination therapy achieves effective pancreatic cancer regression "
    "and prevents tumor resistance. bioRxiv doi: 10.1101/2025.08.04.668325."
)

# Publication-quality color palette (distinct, colorblind-friendly)
PALETTE = {
    "primary": "#2E86AB",    # Deep blue
    "secondary": "#A23B72",   # Magenta
    "accent": "#F18F01",     # Amber
    "success": "#3B7A57",    # Forest green
    "neutral": "#5C5C5C",    # Charcoal
    "light": "#E8E8E8",
    "gradient": ["#2E86AB", "#A23B72", "#F18F01", "#3B7A57", "#5C5C5C"],
}


def _setup_style():
    """Apply publication-quality style. Fallback to seaborn if SciencePlots unavailable."""
    try:
        import scienceplots
        import matplotlib.pyplot as plt
        plt.style.use(["science", "nature", "no-latex"])
        return plt
    except (ImportError, OSError):
        pass
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="husl", font="sans-serif",
                     rc={"axes.facecolor": "#FAFAFA", "figure.facecolor": "white",
                         "axes.edgecolor": ".15", "grid.color": ".9",
                         "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
        return plt
    except ImportError:
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.family": "sans-serif", "font.size": 11,
            "axes.titlesize": 13, "axes.labelsize": 11,
            "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
        })
        return plt


def _save_fig(plt, path: Path, dpi=300):
    """Save figure with tight layout and high DPI."""
    plt.tight_layout(pad=1.2)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved {path}")


def fig1_pipeline_schematic(fig_dir: Path):
    """Figure 1: Pipeline overview schematic — polished flowchart."""
    plt = _setup_style()
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(-0.2, 10.5)
    ax.set_ylim(0.8, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")

    boxes = [
        (0.4, 2.2, 1.4, 1.2, "DepMap\nCRISPR, Model", PALETTE["primary"]),
        (2.0, 2.2, 1.4, 1.2, "Cancer mapping\nOncoTree", PALETTE["secondary"]),
        (3.6, 2.2, 1.4, 1.2, "Viability paths\nco-essentiality,\nsignaling", PALETTE["accent"]),
        (5.2, 2.2, 1.4, 1.2, "Minimal hitting set\nX-nodes", "#C73E1D"),
        (6.8, 2.2, 1.4, 1.2, "Triple ranking\nsynergy, resistance", PALETTE["success"]),
        (8.4, 2.2, 1.4, 1.2, "Validation\nPubMed, STRING,\nPRISM, trials", PALETTE["neutral"]),
    ]
    for i, (x, y, w, h, text, color) in enumerate(boxes):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                             facecolor=color, edgecolor="white", alpha=0.95, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9,
                color="white", fontweight="medium", linespacing=1.3)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + 0.12, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["neutral"], lw=2.5))

    ax.text(5.25, 3.5, "ALIN Framework: Pipeline Overview", fontsize=14, fontweight="bold",
            ha="center", color=PALETTE["neutral"])
    _save_fig(plt, fig_dir / "fig1_pipeline_schematic.png")


def fig2_xnode_concept(fig_dir: Path):
    """Figure 2: X-node / minimal hitting set concept."""
    plt = _setup_style()
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.8, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Venn-style overlapping paths with refined colors
    paths = [
        Ellipse((2.8, 2.8), 2.2, 1.6, angle=0, facecolor=PALETTE["primary"], alpha=0.35, edgecolor=PALETTE["primary"], lw=2.5),
        Ellipse((4.2, 3.0), 2.2, 1.6, angle=22, facecolor=PALETTE["secondary"], alpha=0.35, edgecolor=PALETTE["secondary"], lw=2.5),
        Ellipse((3.4, 3.6), 2.2, 1.6, angle=-18, facecolor=PALETTE["accent"], alpha=0.35, edgecolor=PALETTE["accent"], lw=2.5),
    ]
    for p in paths:
        ax.add_patch(p)

    # X-nodes marker
    ax.scatter([3.6], [3.0], s=550, c="#C73E1D", marker="*", zorder=10, edgecolors="white", linewidths=3)
    ax.text(3.6, 2.35, "X-nodes\nRAF1 + EGFR + STAT3", ha="center", fontsize=10, fontweight="bold", color=PALETTE["neutral"])

    ax.text(1.6, 1.6, "Path 1\n(downstream)", fontsize=9, color=PALETTE["primary"], fontweight="medium")
    ax.text(5.4, 1.9, "Path 2\n(upstream)", fontsize=9, color=PALETTE["secondary"], fontweight="medium")
    ax.text(2.2, 4.6, "Path 3\n(orthogonal)", fontsize=9, color=PALETTE["accent"], fontweight="medium")

    ax.text(3.5, 0.5, "Liaki et al., bioRxiv doi: 10.1101/2025.08.04.668325", fontsize=8,
            ha="center", style="italic", color="gray")
    ax.text(3.5, 5.0, "X-Node Concept: Minimal Hitting Set", fontsize=13, fontweight="bold", ha="center", color=PALETTE["neutral"])
    _save_fig(plt, fig_dir / "fig2_xnode_concept.png")


def fig3_benchmark(base: Path, fig_dir: Path):
    """Figure 3: Benchmark comparison — polished bar chart."""
    plt = _setup_style()
    import seaborn as sns

    metrics_file = base / "benchmark_results" / "benchmark_metrics.json"
    if not metrics_file.exists():
        print("Skipping fig3: benchmark_metrics.json not found")
        return
    with open(metrics_file) as f:
        metrics = json.load(f)

    df = pd.DataFrame({
        "Method": ["ALIN\n(ours)", "Random\nbaseline", "Top-genes\nbaseline"],
        "Recall (%)": [
            metrics.get("recall_any", 0) * 100,
            metrics.get("random_baseline_mean", 0) * 100,
            metrics.get("topgenes_baseline", 0) * 100,
        ],
        "Error": [0, metrics.get("random_baseline_std", 0) * 100, 0],
        "Color": [PALETTE["success"], PALETTE["neutral"], "#C73E1D"],
    })

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(df["Method"], df["Recall (%)"], color=df["Color"], edgecolor="white", linewidth=2, width=0.65)
    ax.errorbar(df["Method"], df["Recall (%)"], yerr=df["Error"], fmt="none", color=PALETTE["neutral"], capsize=6, capthick=2)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.set_ylabel("Recall (%)", fontsize=12, fontweight="medium")
    ax.set_title("Benchmark: Recovery of Known Drug Combinations", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(plt, fig_dir / "fig3_benchmark_comparison.png")


def fig4_triple_patterns(base: Path, fig_dir: Path):
    """Figure 4: Pan-cancer triple patterns."""
    plt = _setup_style()

    triples_file = base / "results_triples" / "triple_combinations.csv"
    if not triples_file.exists():
        print("Skipping fig4: triple_combinations.csv not found")
        return
    df = pd.read_csv(triples_file)
    if "Triple_Targets" not in df.columns:
        return

    pattern_counts = df["Triple_Targets"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [PALETTE["primary"] if i == 0 else PALETTE["secondary"] if i == 1 else PALETTE["accent"] for i in range(len(pattern_counts))]
    bars = ax.barh(range(len(pattern_counts)), pattern_counts.values, color=colors, edgecolor="white", linewidth=1.5, height=0.7)
    ax.set_yticks(range(len(pattern_counts)))
    ax.set_yticklabels(pattern_counts.index, fontsize=10, fontweight="medium")
    ax.set_xlabel("Number of cancer types", fontsize=12, fontweight="medium")
    ax.set_title("Top Triple Combination Patterns (Pan-Cancer)", fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    _save_fig(plt, fig_dir / "fig4_triple_patterns.png")


def fig5_target_frequency(base: Path, fig_dir: Path):
    """Figure 5: Most frequently predicted targets."""
    plt = _setup_style()

    freq_file = base / "results_triples" / "triple_target_frequency.csv"
    triples_file = base / "results_triples" / "triple_combinations.csv"

    if freq_file.exists():
        df = pd.read_csv(freq_file)
        count_col = "Appearances_in_Triples" if "Appearances_in_Triples" in df.columns else "Count"
        target_col = "Target_Gene" if "Target_Gene" in df.columns else "Target"
    elif triples_file.exists():
        df = pd.read_csv(triples_file)
        targets = []
        for col in ["Target_1", "Target_2", "Target_3"]:
            if col in df.columns:
                targets.extend(df[col].dropna().tolist())
        from collections import Counter
        counts = Counter(targets)
        df = pd.DataFrame([{"Target_Gene": k, "Appearances_in_Triples": v} for k, v in counts.most_common(15)])
        target_col, count_col = "Target_Gene", "Appearances_in_Triples"
    else:
        print("Skipping fig5: no triple data found")
        return

    if target_col not in df.columns or count_col not in df.columns:
        return

    top = df.nlargest(12, count_col)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.Blues(0.4 + 0.5 * (top[count_col].values / top[count_col].max()))
    ax.barh(top[target_col].iloc[::-1], top[count_col].iloc[::-1], color=colors, edgecolor="white", linewidth=1.5, height=0.7)
    ax.set_xlabel("Frequency in top triple combinations", fontsize=12, fontweight="medium")
    ax.set_title("Most Frequently Predicted Targets (Pan-Cancer)", fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(plt, fig_dir / "fig5_target_frequency.png")


def fig6_synergy_resistance(base: Path, fig_dir: Path):
    """Figure 6: Synergy vs resistance landscape."""
    plt = _setup_style()

    triples_file = base / "results_triples" / "triple_combinations.csv"
    if not triples_file.exists():
        return
    df = pd.read_csv(triples_file)
    if "Synergy_Score" not in df.columns or "Resistance_Score" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter = ax.scatter(df["Synergy_Score"], df["Resistance_Score"], alpha=0.7, s=60, c=PALETTE["accent"],
                        edgecolors="white", linewidths=0.5)
    ax.axvline(x=0.9, color=PALETTE["success"], linestyle="--", alpha=0.7, linewidth=1.5, label="High synergy")
    ax.axhline(y=0.4, color="#C73E1D", linestyle="--", alpha=0.7, linewidth=1.5, label="Low resistance")
    ax.set_xlabel("Synergy Score", fontsize=12, fontweight="medium")
    ax.set_ylabel("Resistance Score (lower = better)", fontsize=12, fontweight="medium")
    ax.set_title("Triple Combinations: Synergy vs Resistance Landscape", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, fancybox=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(plt, fig_dir / "fig6_synergy_resistance.png")


def figS1_detailed_pipeline(fig_dir: Path):
    """Figure S1: Detailed pipeline flowchart."""
    plt = _setup_style()
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(1, 5.5)
    ax.axis("off")

    steps = [
        (1.2, 3.5, "DepMap\nCRISPRGeneEffect\nModel, Subtype"),
        (3.2, 3.5, "OncoTree\nCancer mapping"),
        (5.2, 3.5, "Viability paths\nCo-essentiality\nSignaling (NetworkX)"),
        (7.2, 3.5, "Minimal hitting set\nCost function\nGreedy + exhaustive"),
        (9.2, 3.5, "Triple finder\nSynergy, resistance\nCombined score"),
        (11.2, 3.5, "Validation\nPubMed, STRING\nClinicalTrials, PRISM"),
    ]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], "#C73E1D", PALETTE["success"], PALETTE["neutral"]]
    for i, (x, y, text) in enumerate(steps):
        box = FancyBboxPatch((x - 0.5, y - 0.6), 1.8, 1.4, boxstyle="round,pad=0.04",
                             facecolor=colors[i], edgecolor="white", alpha=0.9, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.4, y + 0.1, text, ha="center", va="center", fontsize=8, color="white", fontweight="medium", linespacing=1.4)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.4, y), xytext=(x + 1.0, y),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["neutral"], lw=2))

    ax.text(6, 4.8, "ALIN Framework: Detailed Pipeline", fontsize=14, fontweight="bold", ha="center", color=PALETTE["neutral"])
    _save_fig(plt, fig_dir / "figS1_detailed_pipeline.png")


def figS2_coessentiality(fig_dir: Path):
    """Figure S2: Co-essentiality clustering schematic."""
    plt = _setup_style()
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0.5, 4)
    ax.axis("off")

    boxes = [
        (0.8, 2.8, "Cell line 1:\nKRAS, STAT3, CDK6", "#E3F2FD"),
        (0.8, 1.6, "Cell line 2:\nKRAS, STAT3, MET", "#E3F2FD"),
        (0.8, 0.6, "Cell line 3:\nSTAT3, CDK6, EGFR", "#E3F2FD"),
        (3.2, 2.0, "Co-occurrence\nmatrix", "#BBDEFB"),
        (5.0, 2.0, "Hierarchical\nclustering", "#90CAF9"),
        (6.8, 2.0, "Pathway-like\nmodules", "#42A5F5"),
    ]
    for x, y, text, color in boxes:
        box = FancyBboxPatch((x - 0.1, y - 0.25), 1.4 if x < 2 else 1.2, 0.9 if x < 2 else 0.7,
                             boxstyle="round,pad=0.02", facecolor=color, edgecolor=PALETTE["neutral"], alpha=0.9)
        ax.add_patch(box)
        ax.text(x + (0.6 if x < 2 else 0.5), y + 0.1, text, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(2.8, 2), xytext=(2.2, 2), arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax.annotate("", xy=(4.8, 2), xytext=(3.5, 2), arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax.annotate("", xy=(6.5, 2), xytext=(5.4, 2), arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax.text(4, 3.6, "Co-Essentiality Clustering → Viability Paths", fontsize=12, fontweight="bold", ha="center", color=PALETTE["neutral"])
    _save_fig(plt, fig_dir / "figS2_coessentiality.png")


def figS3_network_paths(fig_dir: Path):
    """Figure S3: Network path inference example."""
    plt = _setup_style()
    try:
        import networkx as nx
    except ImportError:
        print("Skipping figS3: networkx not installed")
        return

    G = nx.DiGraph()
    G.add_edges_from([
        ("KRAS", "BRAF"), ("KRAS", "PIK3CA"), ("BRAF", "MAP2K1"), ("MAP2K1", "MYC"),
        ("EGFR", "KRAS"), ("EGFR", "STAT3"), ("STAT3", "MYC"), ("STAT3", "BCL2"),
        ("PIK3CA", "AKT1"), ("AKT1", "MTOR"),
    ])
    pos = nx.spring_layout(G, k=2.0, seed=42, iterations=100)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    nx.draw_networkx_nodes(G, pos, node_color=PALETTE["primary"], node_size=1200, ax=ax, alpha=0.9, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=PALETTE["neutral"], arrows=True, ax=ax, alpha=0.7,
                           connectionstyle="arc3,rad=0.1", arrowsize=20, width=2)
    ax.set_title("Signaling Path Inference: Driver → Effector (NetworkX)", fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")
    _save_fig(plt, fig_dir / "figS3_network_paths.png")


def figS4_benchmark_rank(base: Path, fig_dir: Path):
    """Figure S4: Benchmark rank distribution."""
    plt = _setup_style()

    metrics_file = base / "benchmark_results" / "benchmark_metrics.json"
    if not metrics_file.exists():
        return
    with open(metrics_file) as f:
        metrics = json.load(f)
    mean_rank = metrics.get("mean_rank_when_matched", 1.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Mean rank when matched"], [mean_rank], color=PALETTE["success"], edgecolor="white", linewidth=2, width=0.5)
    ax.set_ylabel("Rank (1 = top prediction)", fontsize=11, fontweight="medium")
    ax.set_title("ALIN places matched gold-standard combinations at rank 1", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(0, 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(plt, fig_dir / "figS4_benchmark_rank.png")


def figS5_novel_combinations(base: Path, fig_dir: Path):
    """Figure S5: Novel combinations table."""
    plt = _setup_style()

    triples_file = base / "results_triples" / "triple_combinations.csv"
    if not triples_file.exists():
        return
    df = pd.read_csv(triples_file)
    if "Synergy_Score" not in df.columns or "Resistance_Score" not in df.columns:
        return

    df["potential"] = df["Synergy_Score"] - df["Resistance_Score"]
    top5 = df.nlargest(5, "potential")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    rows = [["Cancer Type", "Triple", "Synergy", "Resistance", "Drugs"]]
    for _, r in top5.iterrows():
        drugs = f"{r.get('Drug_1','')}+{r.get('Drug_2','')}+{r.get('Drug_3','')}"[:45]
        rows.append([
            str(r.get("Cancer_Type", ""))[:28],
            str(r.get("Triple_Targets", ""))[:28],
            f"{r['Synergy_Score']:.2f}",
            f"{r['Resistance_Score']:.2f}",
            drugs,
        ])

    table = ax.table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.22, 0.22, 0.12, 0.12, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)
    for (i, j), cell in table.get_celld().items():
        cell.set_facecolor("#F5F5F5" if i == 0 else "white")
        cell.set_text_props(fontweight="bold" if i == 0 else "normal")
        cell.set_edgecolor(PALETTE["light"])
    ax.set_title("Top 5 Combinations by Therapeutic Potential (Synergy − Resistance)", fontsize=12, fontweight="bold", pad=20)
    _save_fig(plt, fig_dir / "figS5_novel_combinations.png")


def main():
    base = Path(__file__).parent
    fig_dir = base / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig1_pipeline_schematic(fig_dir)
    fig2_xnode_concept(fig_dir)
    fig3_benchmark(base, fig_dir)
    fig4_triple_patterns(base, fig_dir)
    fig5_target_frequency(base, fig_dir)
    fig6_synergy_resistance(base, fig_dir)
    figS1_detailed_pipeline(fig_dir)
    figS2_coessentiality(fig_dir)
    figS3_network_paths(fig_dir)
    figS4_benchmark_rank(base, fig_dir)
    figS5_novel_combinations(base, fig_dir)

    # Legacy filenames
    import shutil
    for src, dst in [
        ("fig3_benchmark_comparison.png", "benchmark_comparison.png"),
        ("fig4_triple_patterns.png", "triple_patterns.png"),
        ("fig5_target_frequency.png", "target_frequency.png"),
        ("fig6_synergy_resistance.png", "synergy_resistance.png"),
    ]:
        s, d = fig_dir / src, fig_dir / dst
        if s.exists():
            shutil.copy(s, d)

    print(f"\nFigures saved to {fig_dir}/")
    print(f"Reference: {LIAKI_ET_AL_CITATION}")


if __name__ == "__main__":
    main()
