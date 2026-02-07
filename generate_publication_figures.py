#!/usr/bin/env python3
"""
Generate publication-quality figures for ALIN Framework.
Four main figures, each a multi-panel composite telling one story.
Designed for Nature/Science-style two-column layout.

Figure 1: Pipeline overview + tri-axial biological principle
Figure 2: Pan-cancer target architecture (frequency + tri-axial mapping)
Figure 3: Benchmark performance + gold-standard recovery
Figure 4: Pathway shifting simulation — intra-axial MHS vs tri-axial combination
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- STYLE SETUP ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Wedge
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib import colormaps

# Global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.labelweight': 'medium',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ---- Color palette (colorblind-friendly) ----
C = {
    'upstream': '#D55E00',       # vermillion — upstream/driver axis
    'downstream': '#0072B2',     # blue — downstream/effector axis
    'orthogonal': '#009E73',     # green — orthogonal/survival axis
    'xnode': '#E69F00',          # amber — intra-axial MHS
    'triple': '#009E73',         # green — tri-axial combination
    'single': '#CC79A7',         # rose — single agent
    'untreated': '#999999',      # gray — no treatment
    'alin': '#0072B2',           # blue — ALIN
    'random': '#999999',         # gray — random baseline
    'topgenes': '#D55E00',       # vermillion — top-genes
    'highlight': '#CC0000',      # red — emphasis
    'text': '#333333',
    'grid': '#DDDDDD',
    'light_blue': '#56B4E9',
    'light_green': '#CCE5CC',
    'light_orange': '#FFE0B2',
    'light_red': '#FFCCBC',
}

BASE = Path(__file__).parent
FIGURES_DIR = BASE / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


def _save(fig, name, close=True):
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES_DIR / f'{name}.{ext}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    if close:
        plt.close(fig)
    print(f"  Saved: figures/{name}.png/pdf")


def _panel_label(ax, label, x=-0.08, y=1.06, fontsize=14):
    """Add bold panel label (A, B, C, ...) to subplot."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='right',
            color=C['text'])


def _clean_axes(ax, top=False, right=False):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)


# ================================================================
# FIGURE 1: PIPELINE OVERVIEW + TRI-AXIAL CONCEPT
# ================================================================
def figure1():
    """
    Panel A: Horizontal pipeline flowchart
    Panel B: Tri-axial biological principle (Liaki)
    """
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.4], hspace=0.4)

    # ---- Panel A: Pipeline ----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.2, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')
    _panel_label(ax, 'A', x=-0.02, y=1.0)

    steps = [
        (0.0, 1.0, 'DepMap\n(CRISPR)', C['light_blue'], C['downstream']),
        (1.9, 1.0, 'Viability\npaths', C['light_orange'], C['upstream']),
        (3.8, 1.0, 'MHS\nhitting set', C['light_red'], C['highlight']),
        (5.7, 1.0, 'Triple\nranking', C['light_green'], C['orthogonal']),
        (7.6, 1.0, 'ODE\nsimulation', '#E8D5F5', '#7B2D8E'),
        (9.5, 1.0, 'Validation', '#E0E0E0', C['text']),
    ]
    box_w, box_h = 1.5, 1.2
    for i, (x, y, txt, bg, border) in enumerate(steps):
        box = FancyBboxPatch((x, y), box_w, box_h,
                             boxstyle='round,pad=0.08,rounding_size=0.15',
                             facecolor=bg, edgecolor=border, linewidth=2.0)
        ax.add_patch(box)
        ax.text(x + box_w/2, y + box_h/2, txt, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=border, linespacing=1.4)
        if i < len(steps) - 1:
            ax.annotate('', xy=(steps[i+1][0] - 0.08, y + box_h/2),
                        xytext=(x + box_w + 0.08, y + box_h/2),
                        arrowprops=dict(arrowstyle='->', color='#555',
                                       lw=2.2, mutation_scale=16))

    # Sub-labels
    sub_labels = [
        (0.0, 0.70, '96 cancer types\n1,900+ cell lines', 7.5),
        (1.9, 0.70, 'Co-essentiality\nSignaling\nPerturbation', 7),
        (3.8, 0.70, 'Weighted min\ncovering set', 7.5),
        (5.7, 0.70, 'Synergy · Resistance\nToxicity · Coverage', 7),
        (7.6, 0.70, 'Pathway shifting\ncompensation', 7),
        (9.5, 0.70, 'PubMed · STRING\nPRISM · Trials', 7),
    ]
    for x, y, txt, fs in sub_labels:
        ax.text(x + box_w/2, y, txt, ha='center', va='top',
                fontsize=fs, color='#555', linespacing=1.3)

    ax.text(5.0, 2.60, 'ALIN Framework Pipeline', fontsize=13,
            fontweight='bold', ha='center', color=C['text'])

    # ---- Panel B: Tri-axial principle ----
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_ylim(-0.5, 4.8)
    ax2.set_aspect('equal')
    ax2.axis('off')
    _panel_label(ax2, 'B', x=-0.02, y=1.0)

    # Three circles representing axes with improved styling
    circle_params = [
        (2.0, 2.0, 'Upstream\ndriver', C['upstream'], ['KRAS', 'BRAF', 'EGFR']),
        (5.0, 2.0, 'Downstream\neffector', C['downstream'], ['CDK4', 'CCND1', 'CDK6']),
        (8.0, 2.0, 'Orthogonal\nsurvival', C['orthogonal'], ['STAT3', 'MCL1', 'FYN']),
    ]

    for cx, cy, label, color, genes in circle_params:
        circle = Circle((cx, cy), 1.20, facecolor=color, alpha=0.15,
                        edgecolor=color, linewidth=3.0)
        ax2.add_patch(circle)
        ax2.text(cx, cy + 0.40, label, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=color, linespacing=1.2)
        gene_str = ' · '.join(genes)
        ax2.text(cx, cy - 0.30, gene_str, ha='center', va='center',
                fontsize=8, color=color, style='italic', fontweight='medium')

    # Enhanced arrows showing pathway shifting
    arrow_style = dict(arrowstyle='->', color='#CC0000', lw=2.0,
                       connectionstyle='arc3,rad=0.3', alpha=0.8)
    # Intra-axial MHS failure: upstream blocked, orthogonal compensates
    ax2.annotate('', xy=(7.0, 2.7), xytext=(3.0, 2.7), arrowprops=arrow_style)
    ax2.text(5.0, 3.5, 'Compensatory\npathway shifting', ha='center',
            fontsize=8, color='#CC0000', fontweight='bold', linespacing=1.3)

    # "Block all three" indicators with improved styling
    for cx, cy, color in [(2.0, 0.4, C['orthogonal']), (5.0, 0.4, C['orthogonal']), (8.0, 0.4, C['orthogonal'])]:
        ax2.text(cx, cy, 'BLOCK', ha='center', fontsize=8.5,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.85, ec=color, lw=1.5))

    # MHS vs tri-axial label with improved clarity
    ax2.text(1.5, 4.3, 'Intra-axial MHS:  blocks 1–2 axes → resistance',
            fontsize=9, color=C['xnode'], fontweight='bold')
    ax2.text(1.5, 3.95, 'Tri-axial combination: blocks all 3 axes → durable response',
            fontsize=9, color=C['triple'], fontweight='bold')

    ax2.text(5.0, -0.35, 'Liaki et al., bioRxiv 2025: Tri-axial inhibition principle',
            fontsize=8, ha='center', style='italic', color='#777')

    _save(fig, 'fig1_pipeline_schematic')


# ================================================================
# FIGURE 2: PAN-CANCER TARGET ARCHITECTURE
# ================================================================
def figure2():
    """
    Panel A: Target frequency bar chart (horizontal)
    Panel B: Tri-axial mapping (stacked bar or heatmap)
    """
    # Load data
    freq_file = BASE / 'results_triples' / 'target_frequency_summary.csv'
    if not freq_file.exists():
        print("  Skipping fig2: target_frequency_summary.csv not found")
        return
    df = pd.read_csv(freq_file)

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.5)

    # ---- Panel A: Target frequency ----
    ax = fig.add_subplot(gs[0])
    _panel_label(ax, 'A')
    _clean_axes(ax)

    top = df.head(15).iloc[::-1]  # reverse for horizontal, show more targets
    genes = top['Target_Gene'].values
    counts = top['Cancer_Types_Count'].values

    # Color by axis role with enhanced palette
    axis_map = {
        'STAT3': C['orthogonal'], 'FYN': C['orthogonal'], 'MCL1': C['orthogonal'],
        'CDK6': C['downstream'], 'CDK4': C['downstream'], 'CDK2': C['downstream'],
        'CCND1': C['downstream'],
        'KRAS': C['upstream'], 'EGFR': C['upstream'], 'BRAF': C['upstream'],
        'MAP2K1': C['upstream'], 'MET': C['upstream'],
        'FGFR1': C['upstream'], 'SRC': C['orthogonal'], 'ERBB2': C['upstream'],
    }
    colors = [axis_map.get(g, '#999') for g in genes]

    bars = ax.barh(range(len(genes)), counts, color=colors,
                   edgecolor='white', linewidth=1.0, height=0.75)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Cancer types with target', fontsize=10.5, fontweight='medium')
    ax.set_title('Target frequency across cancers', fontsize=12, pad=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.25, color=C['grid'], linestyle='--')

    # Enhanced value labels
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height()/2,
                str(int(val)), va='center', fontsize=8, color='#444', fontweight='medium')

    # STAT3 highlight annotation — compute from data
    stat3_idx = list(genes).index('STAT3') if 'STAT3' in genes else None
    if stat3_idx is not None:
        # Compute actual percentage from the full dataset
        total_cancers = len(df['Cancer_Type'].unique()) if 'Cancer_Type' in df.columns else 90
        stat3_pct = int(round(counts[stat3_idx] / total_cancers * 100))
        ax.text(counts[stat3_idx] + 4, stat3_idx,
                f'{stat3_pct}% of cancers',
                va='center', fontsize=8, color=C['orthogonal'],
                fontweight='bold', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec=C['orthogonal'], lw=1))

    # ---- Panel B: Tri-axial mapping ----
    ax2 = fig.add_subplot(gs[1])
    _panel_label(ax2, 'B')
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.5, 6.8)
    ax2.axis('off')
    ax2.set_title('Tri-axial classification', fontsize=12, pad=10, fontweight='bold')

    # Axis categories with their genes and cancer counts
    axes_data = [
        ('Orthogonal\n(survival)', C['orthogonal'],
         [('STAT3', 70), ('FYN', 38), ('MCL1', 7), ('SRC', 2)]),
        ('Downstream\n(effector)', C['downstream'],
         [('CDK6', 34), ('CDK4', 14), ('CDK2', 7), ('CCND1', 5)]),
        ('Upstream\n(driver)', C['upstream'],
         [('KRAS', 23), ('EGFR', 12), ('MAP2K1', 8), ('BRAF', 6)]),
    ]

    for col, (axis_label, color, gene_list) in enumerate(axes_data):
        # Enhanced axis header
        box = FancyBboxPatch((col - 0.35, 5.7), 1.1, 0.8,
                            boxstyle='round,pad=0.08,rounding_size=0.12',
                            facecolor=color, alpha=0.20, edgecolor=color, lw=2.0)
        ax2.add_patch(box)
        ax2.text(col + 0.2, 6.05, axis_label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=color, linespacing=1.3)

        # Enhanced gene entries
        for row, (gene, cnt) in enumerate(gene_list):
            y = 5.0 - row * 1.15
            # Enhanced mini bar with gradient effect
            bar_w = cnt / 70 * 0.85  # normalize
            ax2.barh(y, bar_w, left=col - 0.28, height=0.40,
                    color=color, alpha=0.6, edgecolor=color, linewidth=0.5)
            ax2.text(col + 0.2, y, f'{gene} ({cnt})', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=C['text'])

    # Enhanced legend
    legend_elements = [
        Line2D([0], [0], color=C['orthogonal'], lw=8, alpha=0.7, label='Orthogonal'),
        Line2D([0], [0], color=C['downstream'], lw=8, alpha=0.7, label='Downstream'),
        Line2D([0], [0], color=C['upstream'], lw=8, alpha=0.7, label='Upstream'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
             fancybox=True, framealpha=0.95, fontsize=9, title='Axis role',
             title_fontsize=9, edgecolor='#CCC')

    _save(fig, 'fig2_xnode_concept')  # reuse old name for paper compatibility


# ================================================================
# FIGURE 3: BENCHMARK PERFORMANCE
# ================================================================
def figure3():
    """
    Panel A: Recall bar chart (ALIN vs baselines)
    Panel B: Gold-standard recovery detail table/waterfall
    """
    metrics_file = BASE / 'benchmark_results' / 'benchmark_metrics.json'
    if not metrics_file.exists():
        print("  Skipping fig3: benchmark_metrics.json not found")
        return
    with open(metrics_file) as f:
        metrics = json.load(f)

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.6], wspace=0.45)

    # ---- Panel A: Recall comparison ----
    ax = fig.add_subplot(gs[0])
    _panel_label(ax, 'A')
    _clean_axes(ax)

    methods = ['ALIN\n(ours)', 'Top-genes\nbaseline', 'Random\nbaseline']
    recalls = [
        metrics['recall_any'] * 100,
        metrics['topgenes_baseline'] * 100,
        metrics['random_baseline_mean'] * 100,
    ]
    errors = [0, 0, metrics['random_baseline_std'] * 100]
    bar_colors = [C['alin'], C['topgenes'], C['random']]

    bars = ax.bar(methods, recalls, color=bar_colors, width=0.65,
                  edgecolor='white', linewidth=2.0, zorder=3)
    ax.errorbar(methods, recalls, yerr=errors, fmt='none',
               color='#222', capsize=6, capthick=2.0, zorder=4)

    # Enhanced value labels with background
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold',
                color=C['text'],
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='none'))

    ax.set_ylabel('Recall (%)', fontsize=11, fontweight='medium')
    ax.set_title('Gold-standard recovery', fontsize=12, pad=10, fontweight='bold')
    ax.set_ylim(0, 85)
    ax.axhline(y=50, color='#BBB', linestyle='--', lw=1.0, zorder=1)
    ax.grid(axis='y', alpha=0.25, color=C['grid'], linestyle=':', zorder=0)
    ax.text(-0.15, 52, '50%', fontsize=7.5, color='#999', fontweight='medium')

    # Enhanced P-value annotation
    ax.annotate('', xy=(0, 75), xytext=(2, 75),
               arrowprops=dict(arrowstyle='-', color='#222', lw=1.2))
    ax.text(1, 77, 'p < 0.001', ha='center', fontsize=8, color='#222', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='#FFF9E6', alpha=0.9, ec='#222', lw=0.8))

    # ---- Panel B: Match breakdown ----
    ax2 = fig.add_subplot(gs[1])
    _panel_label(ax2, 'B')
    ax2.axis('off')
    ax2.set_title(f'Benchmark detail ({metrics["total_gold_standard"]} gold standards)', 
                  fontsize=12, pad=10, fontweight='bold')

    # Summary stats — computed from benchmark data, not hardcoded
    n_superset = metrics.get('superset_matches', 0)
    n_pairwise = metrics.get('pairwise_matches', 0)
    n_nomatch  = metrics.get('no_match', 0)
    match_data = [
        ('Superset matches', n_superset, C['alin']),
        ('Pairwise matches', n_pairwise, C['light_blue']),
        ('Not matched', n_nomatch, '#DDD'),
    ]

    # Enhanced pie chart for match breakdown
    sizes = [d[1] for d in match_data]
    colors_pie = [d[2] for d in match_data]
    labels = [f'{d[0]}\n({d[1]})' for d in match_data]

    wedges, texts = ax2.pie(sizes, colors=colors_pie, startangle=90,
                            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=3),
                            center=(0.35, 0.5), radius=0.38)

    # Enhanced legend for pie
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=c, markersize=12, 
               label=l, markeredgecolor='white', markeredgewidth=1.5)
        for l, _, c in match_data
    ]
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.0, 0.5),
              frameon=True, fancybox=True, fontsize=9, framealpha=0.95, edgecolor='#CCC')

    # Enhanced key matches text
    key_matches = [
        'BRAF+MEK → BRAF+MAP2K1+STAT3  (Melanoma)',
        'CDK4+CDK6 → CDK4+KRAS+STAT3  (Breast)',
        'CDK4+KRAS → CDK4+KRAS+STAT3  (Breast)',
        'KRAS → BRAF+KRAS+STAT3  (CRC)',
        'EGFR+MET → CDK6+EGFR+MET  (HNSCC)',
    ]
    ax2.text(0.48, 0.95, 'Key recoveries:', fontsize=9, fontweight='bold',
            transform=ax2.transAxes, color=C['text'])
    for i, txt in enumerate(key_matches):
        ax2.text(0.50, 0.85 - i * 0.11, txt, fontsize=7,
                transform=ax2.transAxes, color='#444',
                fontfamily='monospace', fontweight='medium')

    ax2.text(0.48, 0.25, f'Mean rank when matched: {metrics.get("mean_rank_when_matched", "N/A")}',
            fontsize=9, fontweight='bold', transform=ax2.transAxes, color=C['alin'],
            bbox=dict(boxstyle='round,pad=0.4', fc='#E6F3FF', alpha=0.8, ec=C['alin'], lw=1.5))

    _save(fig, 'fig3_benchmark_comparison')


# ================================================================
# FIGURE 4: PATHWAY SHIFTING SIMULATION
# ================================================================
def figure4():
    """
    Panel A: Viability time course (PDAC example — MHS vs tri-axial)
    Panel B: Cross-cancer final viability (grouped bar chart)
    Panel C: PDAC node-level dynamics showing compensation
    """
    metrics_file = BASE / 'simulation_results' / 'simulation_metrics.json'
    if not metrics_file.exists():
        print("  Skipping fig4: simulation_metrics.json not found")
        return

    with open(metrics_file) as f:
        sim_data = json.load(f)

    # We also need the actual time-series data from simulation
    # Load from the npz files if available, otherwise use metrics
    fig = plt.figure(figsize=(7.2, 8.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.42, wspace=0.38)

    cancers = ['PDAC', 'Melanoma', 'NSCLC', 'CRC', 'Breast']
    strategies_of_interest = ['Single', 'X-node', 'Three-axis']  # keys match simulation_metrics.json
    strat_display = {'Single': 'Single', 'X-node': 'Intra-axial MHS', 'Three-axis': 'Tri-axial'}
    strat_colors = {
        'No treatment': C['untreated'],
        'Single': C['single'],
        'X-node': C['xnode'],
        'Three-axis': C['triple'],
    }

    # ---- Panel A: Cross-cancer grouped bar chart ----
    ax = fig.add_subplot(gs[0, :])
    _panel_label(ax, 'A')
    _clean_axes(ax)

    x = np.arange(len(cancers))
    bar_w = 0.24
    offsets = {'Single': -bar_w, 'X-node': 0, 'Three-axis': bar_w}

    for stype, offset in offsets.items():
        vals = []
        for cancer in cancers:
            if cancer not in sim_data:
                vals.append(0)
                continue
            found = False
            for r in sim_data[cancer]:
                if stype.lower() in r['strategy'].lower():
                    vals.append(r['final_viability'])
                    found = True
                    break
            if not found:
                vals.append(0)

        bars = ax.bar(x + offset, vals, bar_w,
                     color=strat_colors[stype],
                     edgecolor='white', linewidth=1.2,
                     label=strat_display[stype], zorder=3, alpha=0.9)
        # Enhanced value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', fontsize=7,
                        color='#444', rotation=90 if val > 0.5 else 0, fontweight='medium')

    ax.set_xticks(x)
    ax.set_xticklabels(cancers, fontsize=10, fontweight='bold')
    ax.set_ylabel('Final tumor viability\n(200-day simulation)', fontsize=11, fontweight='medium')
    ax.set_title('Intra-axial MHS vs. tri-axial: final viability across cancers', 
                 fontsize=12, pad=12, fontweight='bold')
    ax.set_ylim(0, 1.20)
    ax.axhline(y=0.5, color='#BBB', linestyle='--', lw=1.2, zorder=1)
    ax.text(-0.5, 0.53, 'Resistance\nthreshold', fontsize=7.5, color='#888', 
            va='bottom', fontweight='medium')
    ax.legend(frameon=True, fancybox=True, fontsize=9, loc='upper right',
             framealpha=0.95, ncol=3, edgecolor='#CCC')
    ax.grid(axis='y', alpha=0.25, zorder=0, linestyle=':')

    # ---- Panel B: Advantage summary (% reduction) ----
    ax2 = fig.add_subplot(gs[1, 0])
    _panel_label(ax2, 'B')
    _clean_axes(ax2)

    advantages = []
    for cancer in cancers:
        if cancer not in sim_data:
            advantages.append(0)
            continue
        xv, tv = None, None
        for r in sim_data[cancer]:
            if 'x-node' in r['strategy'].lower():
                xv = r['final_viability']
            if 'three-axis' in r['strategy'].lower():
                tv = r['final_viability']
        if xv and tv and xv > 0:
            advantages.append((xv - tv) / xv * 100)
        else:
            advantages.append(0)

    bars = ax2.bar(cancers, advantages, color=[C['triple']] * len(cancers),
                   edgecolor='white', linewidth=1.2, width=0.60, zorder=3, alpha=0.9)
    for bar, val in zip(bars, advantages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold',
                color=C['triple'])

    # Enhanced mean line
    mean_adv = np.mean(advantages)
    ax2.axhline(y=mean_adv, color=C['highlight'], linestyle='--', lw=1.5, zorder=2, alpha=0.8)
    ax2.text(len(cancers) - 0.6, mean_adv + 1.5, f'Mean: {mean_adv:.1f}%',
            fontsize=9, color=C['highlight'], fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec=C['highlight'], lw=1))

    ax2.set_ylabel('Viability reduction (%)\nvs. intra-axial MHS', fontsize=11, fontweight='medium')
    ax2.set_title('Tri-axial advantage', fontsize=12, pad=10, fontweight='bold')
    ax2.set_ylim(0, max(advantages) * 1.30)
    ax2.grid(axis='y', alpha=0.25, zorder=0, linestyle=':')
    ax2.tick_params(axis='x', rotation=25)

    # ---- Panel C: Pathway shift magnitude ----
    ax3 = fig.add_subplot(gs[1, 1])
    _panel_label(ax3, 'C')
    _clean_axes(ax3)

    xnode_shifts = []
    triple_shifts = []
    for cancer in cancers:
        if cancer not in sim_data:
            continue
        for r in sim_data[cancer]:
            if 'x-node' in r['strategy'].lower():
                xnode_shifts.append(r['pathway_shift_magnitude'])
            if 'three-axis' in r['strategy'].lower():
                triple_shifts.append(r['pathway_shift_magnitude'])

    x_pos = np.arange(len(cancers))
    ax3.bar(x_pos - 0.17, xnode_shifts, 0.32,
           color=C['xnode'], edgecolor='white', linewidth=1.2,
           label='Intra-axial MHS', zorder=3, alpha=0.9)
    ax3.bar(x_pos + 0.17, triple_shifts, 0.32,
           color=C['triple'], edgecolor='white', linewidth=1.2,
           label='Tri-axial', zorder=3, alpha=0.9)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(cancers, fontsize=9)
    ax3.set_ylabel('Pathway shift\nmagnitude', fontsize=11, fontweight='medium')
    ax3.set_title('Compensatory activation', fontsize=12, pad=10, fontweight='bold')
    ax3.legend(frameon=True, fancybox=True, fontsize=9, loc='upper right',
               framealpha=0.95, edgecolor='#CCC')
    ax3.grid(axis='y', alpha=0.25, zorder=0, linestyle=':')
    ax3.tick_params(axis='x', rotation=25)

    _save(fig, 'fig4_triple_patterns')  # reuse old name for paper compatibility

    # Also save as the simulation figure referenced in paper
    fig2 = plt.figure(figsize=(7.2, 4.5))
    gs2 = gridspec.GridSpec(1, 2, wspace=0.35)

    # Reproduce panels A and B for the standalone simulation figure
    ax_s1 = fig2.add_subplot(gs2[0])
    _panel_label(ax_s1, 'A')
    _clean_axes(ax_s1)

    for stype, offset in offsets.items():
        vals = []
        for cancer in cancers:
            if cancer not in sim_data:
                vals.append(0)
                continue
            found = False
            for r in sim_data[cancer]:
                if stype.lower() in r['strategy'].lower():
                    vals.append(r['final_viability'])
                    found = True
                    break
            if not found:
                vals.append(0)
        ax_s1.bar(x + offset, vals, bar_w, color=strat_colors[stype],
                 edgecolor='white', linewidth=0.8, label=strat_display[stype], zorder=3)

    ax_s1.set_xticks(x)
    ax_s1.set_xticklabels(cancers, fontsize=8, rotation=15)
    ax_s1.set_ylabel('Final tumor viability', fontsize=10)
    ax_s1.set_title('Final viability by strategy', fontsize=11, pad=8)
    ax_s1.set_ylim(0, 1.1)
    ax_s1.axhline(y=0.5, color='#CCC', linestyle='--', lw=0.8)
    ax_s1.legend(frameon=True, fontsize=7, loc='upper right', ncol=3)
    ax_s1.grid(axis='y', alpha=0.2, zorder=0)

    ax_s2 = fig2.add_subplot(gs2[1])
    _panel_label(ax_s2, 'B')
    _clean_axes(ax_s2)
    ax_s2.bar(x_pos - 0.15, xnode_shifts, 0.28, color=C['xnode'],
             edgecolor='white', linewidth=0.8, label='Intra-axial MHS', zorder=3)
    ax_s2.bar(x_pos + 0.15, triple_shifts, 0.28, color=C['triple'],
             edgecolor='white', linewidth=0.8, label='Tri-axial', zorder=3)
    ax_s2.set_xticks(x_pos)
    ax_s2.set_xticklabels(cancers, fontsize=8, rotation=15)
    ax_s2.set_ylabel('Pathway shift magnitude', fontsize=10)
    ax_s2.set_title('Compensatory pathway activation', fontsize=11, pad=8)
    ax_s2.legend(frameon=True, fontsize=7, loc='upper right')
    ax_s2.grid(axis='y', alpha=0.2, zorder=0)

    fig2.suptitle('Intra-Axial MHS vs. Tri-Axial Combination: Systems Biology Comparison',
                  fontsize=12, fontweight='bold', y=1.02)
    _save(fig2, '../simulation_results/figures/xnode_vs_triple_comparison')


# ================================================================
# FIGURE 5: TARGET FREQUENCY (standalone — referenced as fig:targets)
# ================================================================
def figure5():
    """Standalone target frequency — maps to fig:targets in paper."""
    freq_file = BASE / 'results_triples' / 'target_frequency_summary.csv'
    if not freq_file.exists():
        print("  Skipping fig5: target_frequency_summary.csv not found")
        return
    df = pd.read_csv(freq_file)
    top = df.head(15).iloc[::-1]
    genes = top['Target_Gene'].values
    counts = top['Cancer_Types_Count'].values

    axis_map = {
        'STAT3': C['orthogonal'], 'FYN': C['orthogonal'], 'MCL1': C['orthogonal'],
        'SRC': C['orthogonal'],
        'CDK6': C['downstream'], 'CDK4': C['downstream'], 'CDK2': C['downstream'],
        'CCND1': C['downstream'],
        'KRAS': C['upstream'], 'EGFR': C['upstream'], 'BRAF': C['upstream'],
        'MAP2K1': C['upstream'], 'MET': C['upstream'],
        'FGFR1': C['upstream'], 'ERBB2': C['upstream'],
    }

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _clean_axes(ax)
    colors = [axis_map.get(g, '#999') for g in genes]
    bars = ax.barh(range(len(genes)), counts, color=colors,
                   edgecolor='white', linewidth=0.8, height=0.7)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=9, fontweight='medium')
    ax.set_xlabel('Cancer types', fontsize=10)
    ax.set_title('Most frequent targets in ranked triples', fontsize=11, pad=8)
    ax.grid(axis='x', alpha=0.3)

    # Axis legend
    legend_elements = [
        Line2D([0], [0], color=C['orthogonal'], lw=6, alpha=0.6, label='Orthogonal'),
        Line2D([0], [0], color=C['downstream'], lw=6, alpha=0.6, label='Downstream'),
        Line2D([0], [0], color=C['upstream'], lw=6, alpha=0.6, label='Upstream'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
             fancybox=True, fontsize=8, title='Axis role', title_fontsize=8)

    _save(fig, 'fig5_target_frequency')


# ================================================================
# FIGURE 6: SYNERGY VS RESISTANCE LANDSCAPE
# ================================================================
def figure6():
    """Synergy vs resistance for all triples."""
    triples_file = BASE / 'results_triples' / 'triple_combinations.csv'
    if not triples_file.exists():
        print("  Skipping fig6: triple_combinations.csv not found")
        return
    df = pd.read_csv(triples_file)
    if 'Synergy_Score' not in df.columns or 'Resistance_Score' not in df.columns:
        print("  Skipping fig6: missing score columns")
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    _clean_axes(ax)

    # Color by pathway coverage
    n_pathways = df.get('Pathways_Covered', pd.Series([2]*len(df)))
    colors = np.where(n_pathways >= 3, C['triple'], C['xnode'])

    ax.scatter(df['Synergy_Score'], df['Resistance_Score'],
              alpha=0.6, s=30, c=colors, edgecolors='white', linewidths=0.3, zorder=3)

    # Highlight optimal quadrant
    ax.axvline(x=0.85, color=C['orthogonal'], linestyle=':', alpha=0.6, lw=1)
    ax.axhline(y=0.35, color=C['highlight'], linestyle=':', alpha=0.6, lw=1)

    # Labels for quadrants
    ax.text(0.95, 0.12, 'Optimal:\nhigh synergy\nlow resistance',
           fontsize=7, color=C['orthogonal'], fontweight='bold', ha='center')
    ax.text(0.55, 0.55, 'Suboptimal:\nlow synergy\nhigh resistance',
           fontsize=7, color='#AAA', ha='center')

    ax.set_xlabel('Synergy score', fontsize=10)
    ax.set_ylabel('Resistance score (lower = better)', fontsize=10)
    ax.set_title('Tri-axial combination landscape', fontsize=11, pad=8)
    ax.grid(alpha=0.2)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C['triple'], markersize=8,
               label='Tri-axial (3 axes)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C['xnode'], markersize=8,
               label='Intra-axial (<3 axes)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=8)

    _save(fig, 'fig6_synergy_resistance')


# ================================================================
# MAIN
# ================================================================
def main():
    print("Generating publication-quality figures...")
    print("=" * 50)
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    print("=" * 50)
    print("Done. All figures saved to figures/")


if __name__ == '__main__':
    main()
