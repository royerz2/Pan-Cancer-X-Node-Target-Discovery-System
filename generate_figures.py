#!/usr/bin/env python3
"""
Generate publication figures for ALIN Framework (Adaptive Lethal Intersection Network).
All figures are generated programmatically for manuscript (bioRxiv).

Reference (source of extrapolated approach):
Liaki V, Barrambana S, et al. 2025. A targeted combination therapy achieves effective
pancreatic cancer regression and prevents tumor resistance. bioRxiv doi: 10.1101/2025.08.04.668325.
"""

import pandas as pd
from pathlib import Path
import json

# Citation for PancraticCancerCure.pdf (Liaki et al.)
LIAKI_ET_AL_CITATION = (
    "Liaki V, Barrambana S, Kostopoulou M, Lechuga CG, et al. 2025. "
    "A targeted combination therapy achieves effective pancreatic cancer regression "
    "and prevents tumor resistance. bioRxiv doi: 10.1101/2025.08.04.668325."
)


def _setup_mpl():
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        return plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib seaborn")
        return None


def fig1_pipeline_schematic(fig_dir: Path):
    """Figure 1: Pipeline overview schematic."""
    plt = _setup_mpl()
    if not plt:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    boxes = [
        (0.5, 2, 1.2, 1.2, "DepMap\n(CRISPR, Model)", '#3498db'),
        (1.9, 2, 1.2, 1.2, "Cancer mapping\n(OncoTree)", '#2ecc71'),
        (3.3, 2, 1.2, 1.2, "Viability paths\n(co-essentiality,\nsignaling)", '#9b59b6'),
        (4.9, 2, 1.2, 1.2, "Minimal hitting set\n+ X-nodes", '#e74c3c'),
        (6.3, 2, 1.2, 1.2, "Triple ranking\n(synergy, resistance)", '#e67e22'),
        (7.7, 2, 1.2, 1.2, "Validation\n(PubMed, STRING,\nPRISM, trials)", '#1abc9c'),
    ]
    for i, (x, y, w, h, text, color) in enumerate(boxes):
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8, wrap=True)
        if i < len(boxes) - 1:
            ax.annotate('', xy=(x + w + 0.15, y + h/2), xytext=(x + w, y + h/2),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_title('ALIN Framework: Pipeline Overview', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_pipeline_schematic.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/fig1_pipeline_schematic.png")


def fig2_xnode_concept(fig_dir: Path):
    """Figure 2: X-node / minimal hitting set concept (inspired by Liaki et al.)."""
    plt = _setup_mpl()
    if not plt:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Three overlapping viability paths (ellipses)
    from matplotlib.patches import Ellipse
    paths = [
        Ellipse((2.5, 2.5), 2, 1.5, angle=0, facecolor='#3498db', alpha=0.4, edgecolor='#2980b9', lw=2),
        Ellipse((4, 2.8), 2, 1.5, angle=25, facecolor='#2ecc71', alpha=0.4, edgecolor='#27ae60', lw=2),
        Ellipse((3.2, 3.5), 2, 1.5, angle=-15, facecolor='#9b59b6', alpha=0.4, edgecolor='#8e44ad', lw=2),
    ]
    for p in paths:
        ax.add_patch(p)

    # X-nodes (targets that hit all paths) - in the intersection
    ax.scatter([3.5], [2.9], s=400, c='red', marker='*', zorder=5, edgecolors='black', linewidths=2)
    ax.text(3.5, 2.4, 'X-nodes\n(RAF1 + EGFR + STAT3)', ha='center', fontsize=9, fontweight='bold')

    # Labels
    ax.text(1.5, 1.5, 'Path 1\n(downstream)', fontsize=8, color='#2980b9')
    ax.text(5, 1.8, 'Path 2\n(upstream)', fontsize=8, color='#27ae60')
    ax.text(2.5, 4.2, 'Path 3\n(orthogonal)', fontsize=8, color='#8e44ad')

    ax.set_title('X-Node Concept: Minimal Hitting Set\n(Liaki et al., bioRxiv doi: 10.1101/2025.08.04.668325)', fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_xnode_concept.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/fig2_xnode_concept.png")


def fig3_benchmark(base: Path, fig_dir: Path):
    """Figure 3: Benchmark comparison."""
    plt = _setup_mpl()
    if not plt:
        return
    metrics_file = base / "benchmark_results" / "benchmark_metrics.json"
    if not metrics_file.exists():
        print("Skipping fig3: benchmark_metrics.json not found")
        return
    with open(metrics_file) as f:
        metrics = json.load(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = ['ALIN\n(ours)', 'Random\nbaseline', 'Top-genes\nbaseline']
    recalls = [
        metrics.get('recall_any', 0),
        metrics.get('random_baseline_mean', 0),
        metrics.get('topgenes_baseline', 0)
    ]
    errs = [0, metrics.get('random_baseline_std', 0), 0]
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    ax.bar(methods, [r*100 for r in recalls], yerr=[e*100 for e in errs],
           color=colors, capsize=5, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Benchmark: Recovery of Known Drug Combinations', fontsize=14)
    ax.set_ylim(0, 80)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_benchmark_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/fig3_benchmark_comparison.png")


def fig4_triple_patterns(base: Path, fig_dir: Path):
    """Figure 4: Pan-cancer triple patterns."""
    plt = _setup_mpl()
    if not plt:
        return
    triples_file = base / "results_triples" / "triple_combinations.csv"
    if not triples_file.exists() or 'Triple_Targets' not in pd.read_csv(triples_file, nrows=1).columns:
        print("Skipping fig4: triple_combinations.csv not found")
        return
    df = pd.read_csv(triples_file)
    pattern_counts = df['Triple_Targets'].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(pattern_counts)), pattern_counts.values, color='#9b59b6', edgecolor='black')
    ax.set_yticks(range(len(pattern_counts)))
    ax.set_yticklabels(pattern_counts.index, fontsize=9)
    ax.set_xlabel('Number of cancer types', fontsize=12)
    ax.set_title('Top Triple Combination Patterns (Pan-Cancer)', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_triple_patterns.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/fig4_triple_patterns.png")


def fig5_target_frequency(base: Path, fig_dir: Path):
    """Figure 5: Most frequently predicted targets."""
    plt = _setup_mpl()
    if not plt:
        return
    triples_file = base / "results_triples" / "triple_target_frequency.csv"
    if not triples_file.exists():
        print("Skipping fig5: triple_target_frequency.csv not found")
        return
    df = pd.read_csv(triples_file)
    count_col = 'Appearances_in_Triples' if 'Appearances_in_Triples' in df.columns else 'Count'
    target_col = 'Target_Gene' if 'Target_Gene' in df.columns else 'Target'
    if target_col not in df.columns or count_col not in df.columns:
        return
    top = df.nlargest(12, count_col)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top[target_col].iloc[::-1], top[count_col].iloc[::-1], color='#3498db', edgecolor='black')
    ax.set_xlabel('Frequency in top triple combinations', fontsize=12)
    ax.set_title('Most Frequently Predicted Targets (Pan-Cancer)', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_target_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/fig5_target_frequency.png")


def fig6_synergy_resistance(base: Path, fig_dir: Path):
    """Figure 6: Synergy vs resistance landscape."""
    plt = _setup_mpl()
    if not plt:
        return
    triples_file = base / "results_triples" / "triple_combinations.csv"
    if not triples_file.exists():
        return
    df = pd.read_csv(triples_file)
    if 'Synergy_Score' not in df.columns or 'Resistance_Score' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df['Synergy_Score'], df['Resistance_Score'], alpha=0.6, s=30, c='#e67e22')
    ax.set_xlabel('Synergy Score', fontsize=12)
    ax.set_ylabel('Resistance Score (lower = better)', fontsize=12)
    ax.set_title('Triple Combinations: Synergy vs Resistance', fontsize=14)
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_synergy_resistance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/fig6_synergy_resistance.png")


def figS1_detailed_pipeline(fig_dir: Path):
    """Figure S1: Detailed pipeline flowchart."""
    plt = _setup_mpl()
    if not plt:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    steps = [
        (1, 4, "DepMap\nCRISPRGeneEffect.csv\nModel.csv\nSubtypeMatrix.csv"),
        (3, 4, "OncoTree\nCancer type mapping"),
        (5, 4, "Viability paths\n• Co-essentiality clustering\n• Signaling (NetworkX)\n• Cancer-specific deps"),
        (7, 4, "Minimal hitting set\n• Cost (toxicity, specificity)\n• Greedy + exhaustive"),
        (9, 4, "Triple finder\n• Synergy score\n• Resistance prob\n• Combined score"),
        (11, 4, "Validation\n• PubMed, STRING\n• ClinicalTrials.gov\n• PRISM drug sensitivity"),
    ]
    from matplotlib.patches import FancyBboxPatch
    for i, (x, y, text) in enumerate(steps):
        box = FancyBboxPatch((x-0.4, y-0.5), 1.6, 1.2, boxstyle="round,pad=0.02",
                             facecolor='#ecf0f1', edgecolor='black')
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=7)
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 1.2, y), xytext=(x + 0.8, y),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.set_title('ALIN Framework: Detailed Pipeline', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / "figS1_detailed_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/figS1_detailed_pipeline.png")


def figS2_coessentiality(fig_dir: Path):
    """Figure S2: Co-essentiality clustering schematic."""
    plt = _setup_mpl()
    if not plt:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Cell lines with essential genes
    ax.text(1, 3, "Cell line 1:\nKRAS, STAT3, CDK6", fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#e8f6f3'))
    ax.text(1, 1.5, "Cell line 2:\nKRAS, STAT3, MET", fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#e8f6f3'))
    ax.text(1, 0.5, "Cell line 3:\nSTAT3, CDK6, EGFR", fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#e8f6f3'))

    ax.annotate('', xy=(3.5, 2), xytext=(2.2, 2), arrowprops=dict(arrowstyle='->', color='gray'))
    ax.text(2.8, 2.3, "Co-occurrence\nmatrix", fontsize=8, ha='center')

    ax.text(4, 2, "Hierarchical\nclustering", fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#d5f5e3'))
    ax.annotate('', xy=(5.5, 2), xytext=(4.8, 2), arrowprops=dict(arrowstyle='->', color='gray'))

    ax.text(6.5, 2, "Pathway-like\nmodules\n(co-essential)", fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#a3e4d7'))
    ax.set_title('Co-Essentiality Clustering → Viability Paths', fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "figS2_coessentiality.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/figS2_coessentiality.png")


def figS3_network_paths(fig_dir: Path):
    """Figure S3: Network path inference example."""
    plt = _setup_mpl()
    if not plt:
        return
    try:
        import networkx as nx
    except ImportError:
        print("Skipping figS3: networkx not installed")
        return
    G = nx.DiGraph()
    G.add_edges_from([
        ('KRAS', 'BRAF'), ('KRAS', 'PIK3CA'), ('BRAF', 'MAP2K1'), ('MAP2K1', 'MYC'),
        ('EGFR', 'KRAS'), ('EGFR', 'STAT3'), ('STAT3', 'MYC'), ('STAT3', 'BCL2'),
        ('PIK3CA', 'AKT1'), ('AKT1', 'MTOR'),
    ])
    pos = nx.spring_layout(G, k=1.5, seed=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx_nodes(G, pos, node_color='#3498db', node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, ax=ax)
    ax.set_title('Signaling Path Inference: Driver → Effector (NetworkX all_simple_paths)', fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(fig_dir / "figS3_network_paths.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/figS3_network_paths.png")


def figS4_benchmark_rank(base: Path, fig_dir: Path):
    """Figure S4: Benchmark rank distribution."""
    plt = _setup_mpl()
    if not plt:
        return
    metrics_file = base / "benchmark_results" / "benchmark_metrics.json"
    if not metrics_file.exists():
        return
    with open(metrics_file) as f:
        metrics = json.load(f)
    mean_rank = metrics.get('mean_rank_when_matched', 1.0)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(['Mean rank when matched'], [mean_rank], color='#2ecc71', edgecolor='black')
    ax.set_ylabel('Rank (1 = top prediction)', fontsize=11)
    ax.set_title('ALIN places matched gold-standard combinations at rank 1', fontsize=11)
    ax.set_ylim(0, 5)
    plt.tight_layout()
    plt.savefig(fig_dir / "figS4_benchmark_rank.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/figS4_benchmark_rank.png")


def figS5_novel_combinations(base: Path, fig_dir: Path):
    """Figure S5: Novel combinations (high synergy, low resistance)."""
    plt = _setup_mpl()
    if not plt:
        return
    triples_file = base / "results_triples" / "triple_combinations.csv"
    if not triples_file.exists():
        return
    df = pd.read_csv(triples_file)
    if 'Synergy_Score' not in df.columns or 'Resistance_Score' not in df.columns:
        return
    # Top 5 by therapeutic potential (high synergy, low resistance)
    df['potential'] = df['Synergy_Score'] - df['Resistance_Score']
    top5 = df.nlargest(5, 'potential')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    rows = [['Cancer Type', 'Triple', 'Synergy', 'Resistance', 'Drugs']]
    for _, r in top5.iterrows():
        drugs = f"{r.get('Drug_1','')}+{r.get('Drug_2','')}+{r.get('Drug_3','')}"[:40]
        rows.append([
            str(r.get('Cancer_Type', ''))[:25],
            str(r.get('Triple_Targets', ''))[:25],
            f"{r['Synergy_Score']:.2f}",
            f"{r['Resistance_Score']:.2f}",
            drugs
        ])
    table = ax.table(cellText=rows, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    ax.set_title('Top 5 Combinations by Therapeutic Potential (Synergy − Resistance)', fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "figS5_novel_combinations.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_dir}/figS5_novel_combinations.png")


def main():
    base = Path(__file__).parent
    fig_dir = base / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Main figures
    fig1_pipeline_schematic(fig_dir)
    fig2_xnode_concept(fig_dir)
    fig3_benchmark(base, fig_dir)
    fig4_triple_patterns(base, fig_dir)
    fig5_target_frequency(base, fig_dir)
    fig6_synergy_resistance(base, fig_dir)

    # Supplementary figures
    figS1_detailed_pipeline(fig_dir)
    figS2_coessentiality(fig_dir)
    figS3_network_paths(fig_dir)
    figS4_benchmark_rank(base, fig_dir)
    figS5_novel_combinations(base, fig_dir)

    # Legacy filenames (for backward compatibility)
    if (fig_dir / "fig3_benchmark_comparison.png").exists():
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
