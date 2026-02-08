#!/usr/bin/env python3
"""
Generate publication-quality figures for ALIN Framework.
Four main figures, each a multi-panel composite telling one story.
Designed for Nature/Science-style two-column layout.

Figure 1: Pipeline overview + tri-axial biological principle
Figure 2: Benchmark performance + gold-standard recovery
Figure 3: Pathway shifting simulation — intra-axial MHS vs tri-axial combination
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Wedge, Arc
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtransforms

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

def _draw_shadow_box(ax, x, y, w, h, fc, ec, rounding=0.20, shadow_offset=0.06):
    """Draw a rounded box with a subtle drop shadow for depth."""
    # Shadow
    shadow = FancyBboxPatch(
        (x + shadow_offset, y - shadow_offset), w, h,
        boxstyle=f'round,pad=0.05,rounding_size={rounding}',
        facecolor='#00000012', edgecolor='none', linewidth=0, zorder=1)
    ax.add_patch(shadow)
    # Main box
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.05,rounding_size={rounding}',
        facecolor=fc, edgecolor=ec, linewidth=2.2, zorder=2)
    ax.add_patch(box)
    return box


def _draw_chevron_arrow(ax, x1, x2, y, color='#B0B0B0', lw=2.0):
    """Draw a clean triangular chevron arrow between pipeline steps."""
    ax.annotate(
        '', xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(
            arrowstyle='-|>', color=color, lw=lw,
            mutation_scale=18, shrinkA=0, shrinkB=0),
        zorder=3)


def figure1():
    """
    Panel A: Horizontal pipeline flowchart — BioRender-style
    Panel B: Tri-axial biological principle (Liaki et al.)
    """
    fig = plt.figure(figsize=(7.2, 9.0))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.32)

    # ===== Panel A: Pipeline =====
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-0.3, 11.0)
    ax.set_ylim(-0.6, 3.4)
    ax.set_aspect('equal')
    ax.axis('off')
    _panel_label(ax, 'A', x=-0.01, y=1.02)

    # Refined pipeline color palette — saturated, modern
    pipeline_steps = [
        {'x': 0.0,  'label': 'DepMap\n(CRISPR)',   'bg': '#42A5F5', 'ec': '#1565C0'},
        {'x': 1.92, 'label': 'Viability\npaths',    'bg': '#FFA726', 'ec': '#E65100'},
        {'x': 3.84, 'label': 'MHS\nhitting set',    'bg': '#EF5350', 'ec': '#B71C1C'},
        {'x': 5.76, 'label': 'Triple\nranking',     'bg': '#66BB6A', 'ec': '#2E7D32'},
        {'x': 7.68, 'label': 'ODE\nsimulation',     'bg': '#AB47BC', 'ec': '#6A1B9A'},
        {'x': 9.55, 'label': 'Validation',          'bg': '#BDBDBD', 'ec': '#616161'},
    ]
    box_w, box_h = 1.55, 1.35

    for i, step in enumerate(pipeline_steps):
        x, y = step['x'], 1.15
        _draw_shadow_box(ax, x, y, box_w, box_h, step['bg'], step['ec'])
        # White text on coloured box — high contrast
        ax.text(x + box_w / 2, y + box_h / 2, step['label'],
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', linespacing=1.35, zorder=4,
                path_effects=[pe.withStroke(linewidth=2, foreground=step['ec'])])
        # Chevron arrows between boxes
        if i < len(pipeline_steps) - 1:
            x_end = pipeline_steps[i + 1]['x'] - 0.05
            x_start = x + box_w + 0.05
            _draw_chevron_arrow(ax, x_start, x_end, y + box_h / 2,
                                color='#9E9E9E', lw=2.2)

    # Sub-labels beneath each box
    sub_info = [
        '96 cancer types\n1,900+ cell lines',
        'Co-essentiality\nSignaling\nPerturbation',
        'Weighted min\ncovering set',
        'Synergy · Resistance\nToxicity · Coverage',
        'Pathway shifting\ncompensation',
        'PubMed · STRING\nPRISM · Trials',
    ]
    for i, txt in enumerate(sub_info):
        cx = pipeline_steps[i]['x'] + box_w / 2
        ax.text(cx, 0.85, txt, ha='center', va='top',
                fontsize=6.8, color='#555555', linespacing=1.25,
                fontweight='medium')

    # Title
    ax.text(5.35, 2.95, 'ALIN Framework Pipeline', fontsize=14,
            fontweight='bold', ha='center', color='#212121',
            fontfamily='sans-serif')

    # ===== Panel B: Tri-axial concept =====
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(-0.8, 11.0)
    ax2.set_ylim(-1.0, 5.8)
    ax2.set_aspect('equal')
    ax2.axis('off')
    _panel_label(ax2, 'B', x=-0.01, y=1.02)

    # ---- Legend / annotation at top ----
    ax2.text(2.0, 5.45,
             'Intra-axial MHS:  blocks 1\u20132 axes',
             fontsize=9.5, color='#E65100', fontweight='bold')
    ax2.text(6.4, 5.45, 'resistance', fontsize=9.5, color='#E65100',
             fontweight='bold', va='center')
    ax2.annotate('', xy=(6.30, 5.45), xytext=(5.80, 5.45),
                 arrowprops=dict(arrowstyle='-|>', color='#E65100', lw=1.8, mutation_scale=14),
                 zorder=5)

    ax2.text(2.0, 5.00,
             'Tri-axial combination:  blocks all 3 axes',
             fontsize=9.5, color='#2E7D32', fontweight='bold')
    ax2.text(6.8, 5.00, 'durable response', fontsize=9.5, color='#2E7D32',
             fontweight='bold', va='center')
    ax2.annotate('', xy=(6.70, 5.00), xytext=(6.20, 5.00),
                 arrowprops=dict(arrowstyle='-|>', color='#2E7D32', lw=1.8, mutation_scale=14),
                 zorder=5)

    # ---- Three circles ----
    circle_data = [
        {'cx': 2.0, 'cy': 2.6, 'label': 'Upstream\ndriver',
         'color': '#D55E00', 'light': '#FDDCCC',
         'genes': ['KRAS', 'BRAF', 'EGFR']},
        {'cx': 5.2, 'cy': 2.6, 'label': 'Downstream\neffector',
         'color': '#0072B2', 'light': '#CCE5F6',
         'genes': ['CDK4', 'CCND1', 'CDK6']},
        {'cx': 8.4, 'cy': 2.6, 'label': 'Orthogonal\nsurvival',
         'color': '#009E73', 'light': '#CFF0E5',
         'genes': ['STAT3', 'MCL1', 'FYN']},
    ]
    R = 1.40  # radius

    for cd in circle_data:
        cx, cy = cd['cx'], cd['cy']
        # Outer glow / halo
        halo = Circle((cx, cy), R + 0.10, facecolor=cd['light'], alpha=0.35,
                       edgecolor='none', zorder=1)
        ax2.add_patch(halo)
        # Main circle
        circ = Circle((cx, cy), R, facecolor=cd['light'], alpha=0.60,
                       edgecolor=cd['color'], linewidth=2.8, zorder=2)
        ax2.add_patch(circ)
        # Inner ring (decorative)
        inner = Circle((cx, cy), R * 0.88, facecolor='none',
                        edgecolor=cd['color'], linewidth=0.6, alpha=0.35, zorder=3,
                        linestyle='--')
        ax2.add_patch(inner)
        # Label
        ax2.text(cx, cy + 0.45, cd['label'], ha='center', va='center',
                 fontsize=10.5, fontweight='bold', color=cd['color'],
                 linespacing=1.2, zorder=4)
        # Gene list
        gene_str = ' \u00B7 '.join(cd['genes'])
        ax2.text(cx, cy - 0.40, gene_str, ha='center', va='center',
                 fontsize=8.5, color=cd['color'], style='italic',
                 fontweight='medium', zorder=4)

    # ---- Curved compensation arrow (upstream → orthogonal, arcing over downstream) ----
    arrow_kw = dict(
        arrowstyle='-|>', color='#CC0000', lw=2.5,
        connectionstyle='arc3,rad=-0.35', mutation_scale=20)
    ax2.annotate('', xy=(7.05, 3.15), xytext=(2.95, 3.15),
                 arrowprops=arrow_kw, zorder=5)

    # "Compensatory pathway shifting" label on the arrow
    ax2.text(5.0, 4.55, 'Compensatory\npathway shifting', ha='center',
             fontsize=9, color='#CC0000', fontweight='bold',
             linespacing=1.25, zorder=5,
             bbox=dict(boxstyle='round,pad=0.25', fc='#FFF5F5',
                       ec='#CC000044', lw=0.8))

    # ---- "BLOCK" tags beneath each circle ----
    for cd in circle_data:
        cx = cd['cx']
        # Small downward arrow from circle to BLOCK tag
        ax2.annotate('', xy=(cx, 0.85), xytext=(cx, 1.15),
                     arrowprops=dict(arrowstyle='-|>', color=cd['color'],
                                     lw=1.8, mutation_scale=14),
                     zorder=5)
        # BLOCK pill
        ax2.text(cx, 0.50, 'BLOCK', ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white', zorder=6,
                 bbox=dict(boxstyle='round,pad=0.30', fc=cd['color'],
                           ec=cd['color'], lw=2.0, alpha=0.92))

    # ---- Reference ----
    ax2.text(5.2, -0.65,
             'Liaki et al., PNAS 2025: Tri-axial inhibition principle',
             fontsize=8, ha='center', style='italic', color='#888888')

    _save(fig, 'fig1_pipeline_schematic')


# ================================================================
# FIGURE 2: BENCHMARK PERFORMANCE
# ================================================================
def figure3():
    """
    Panel A: Recall bar chart (ALIN vs baselines)
    Panel B: Gold-standard recovery detail
    """
    metrics_file = BASE / 'benchmark_results' / 'benchmark_metrics.json'
    if not metrics_file.exists():
        print("  Skipping fig3: benchmark_metrics.json not found")
        return
    with open(metrics_file) as f:
        metrics = json.load(f)

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.40)

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

    bars = ax.bar(methods, recalls, color=bar_colors, width=0.62,
                  edgecolor='white', linewidth=2.0, zorder=3)
    ax.errorbar(methods, recalls, yerr=errors, fmt='none',
               color='#222', capsize=6, capthick=2.0, zorder=4)

    # Value labels above bars
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.5,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold',
                color=C['text'])

    ax.set_ylabel('Recall (%)', fontsize=11, fontweight='medium')
    ax.set_title('Gold-standard recovery', fontsize=12, pad=10, fontweight='bold')
    ax.set_ylim(0, 82)
    ax.axhline(y=50, color='#CCC', linestyle='--', lw=0.8, zorder=1)
    ax.grid(axis='y', alpha=0.2, color=C['grid'], linestyle=':', zorder=0)

    # P-value significance bracket: proper bracket with vertical legs
    bracket_y = 72
    leg_h = 2.5
    # Left leg (ALIN, x=0)
    ax.plot([0, 0], [bracket_y - leg_h, bracket_y], color='#333', lw=1.2, zorder=5, clip_on=False)
    # Horizontal bar
    ax.plot([0, 2], [bracket_y, bracket_y], color='#333', lw=1.2, zorder=5, clip_on=False)
    # Right leg (Random, x=2)
    ax.plot([2, 2], [bracket_y - leg_h, bracket_y], color='#333', lw=1.2, zorder=5, clip_on=False)
    # Stars + text
    ax.text(1, bracket_y + 1.5, '***', ha='center', fontsize=11, color='#333',
            fontweight='bold', zorder=5)
    ax.text(1, bracket_y + 5, 'p < 0.001', ha='center', fontsize=7.5, color='#555',
            zorder=5)

    # ---- Panel B: Key recoveries (clean text layout, no pie chart) ----
    ax2 = fig.add_subplot(gs[1])
    _panel_label(ax2, 'B')
    ax2.axis('off')
    ax2.set_title(f'Benchmark detail ({metrics["total_gold_standard"]} gold standards)',
                  fontsize=12, pad=10, fontweight='bold')

    # Match summary as colored bars at top
    n_superset = metrics.get('superset_matches', 0)
    n_pairwise = metrics.get('pairwise_matches', 0)
    n_nomatch  = metrics.get('no_match', 0)
    total = n_superset + n_pairwise + n_nomatch

    # Stacked horizontal bar showing match breakdown
    bar_y = 0.88
    bar_h = 0.06
    x_start = 0.08
    bar_total_w = 0.84
    # Superset
    w_sup = n_superset / total * bar_total_w
    ax2.add_patch(FancyBboxPatch((x_start, bar_y), w_sup, bar_h,
                  boxstyle='round,pad=0.005,rounding_size=0.01',
                  facecolor=C['alin'], edgecolor='white', lw=1.5,
                  transform=ax2.transAxes, clip_on=False, zorder=3))
    ax2.text(x_start + w_sup/2, bar_y + bar_h/2,
             f'Superset: {n_superset}', ha='center', va='center',
             fontsize=7.5, fontweight='bold', color='white',
             transform=ax2.transAxes, zorder=4)
    # Pairwise
    w_pair = n_pairwise / total * bar_total_w
    ax2.add_patch(FancyBboxPatch((x_start + w_sup, bar_y), w_pair, bar_h,
                  boxstyle='round,pad=0.005,rounding_size=0.01',
                  facecolor=C['light_blue'], edgecolor='white', lw=1.5,
                  transform=ax2.transAxes, clip_on=False, zorder=3))
    if w_pair > 0.06:
        ax2.text(x_start + w_sup + w_pair/2, bar_y + bar_h/2,
                 f'{n_pairwise}', ha='center', va='center',
                 fontsize=7.5, fontweight='bold', color=C['alin'],
                 transform=ax2.transAxes, zorder=4)
    # Not matched
    w_no = n_nomatch / total * bar_total_w
    ax2.add_patch(FancyBboxPatch((x_start + w_sup + w_pair, bar_y), w_no, bar_h,
                  boxstyle='round,pad=0.005,rounding_size=0.01',
                  facecolor='#DDD', edgecolor='white', lw=1.5,
                  transform=ax2.transAxes, clip_on=False, zorder=3))
    ax2.text(x_start + w_sup + w_pair + w_no/2, bar_y + bar_h/2,
             f'No match: {n_nomatch}', ha='center', va='center',
             fontsize=7.5, fontweight='bold', color='#666',
             transform=ax2.transAxes, zorder=4)

    # Key recoveries list
    key_matches = [
        ('BRAF+MEK', 'BRAF+MAP2K1+STAT3', 'Melanoma'),
        ('CDK4+CDK6', 'CDK4+KRAS+STAT3', 'Breast'),
        ('CDK4+KRAS', 'CDK4+KRAS+STAT3', 'Breast'),
        ('KRAS', 'BRAF+KRAS+STAT3', 'CRC'),
        ('EGFR+MET', 'CDK6+EGFR+MET', 'HNSCC'),
    ]

    ax2.text(0.08, 0.78, 'Key recoveries', fontsize=10, fontweight='bold',
             transform=ax2.transAxes, color=C['text'])

    for i, (gold, pred, cancer) in enumerate(key_matches):
        y = 0.68 - i * 0.115
        # Gold standard (left)
        ax2.text(0.08, y, gold, fontsize=8, fontweight='bold',
                 transform=ax2.transAxes, color='#444', fontfamily='monospace')
        # Arrow
        ax2.annotate('', xy=(0.34, y + 0.01), xytext=(0.29, y + 0.01),
                     xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(arrowstyle='-|>', color=C['alin'],
                                     lw=1.5, mutation_scale=12))
        # Predicted triple + cancer type on same line, smaller
        ax2.text(0.36, y, f'{pred}  ', fontsize=8, fontweight='bold',
                 transform=ax2.transAxes, color=C['alin'], fontfamily='monospace')
        # Cancer type - right-aligned, clearly separated
        ax2.text(0.98, y, cancer, fontsize=7.5, transform=ax2.transAxes,
                 color='#999', ha='right', style='italic')

    # Mean rank stat
    mean_rank = metrics.get('mean_rank_when_matched', 'N/A')
    ax2.text(0.08, 0.08, f'Mean rank when matched: {mean_rank}',
             fontsize=9.5, fontweight='bold', transform=ax2.transAxes, color=C['alin'],
             bbox=dict(boxstyle='round,pad=0.35', fc='#E6F3FF', alpha=0.85,
                       ec=C['alin'], lw=1.2))

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
    
    # Save to simulation_results/figures directory
    sim_fig_dir = BASE / 'simulation_results' / 'figures'
    sim_fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig2.savefig(sim_fig_dir / f'xnode_vs_triple_comparison.{ext}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  Saved: simulation_results/figures/xnode_vs_triple_comparison.png/pdf")


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
    figure3()
    figure4()
    figure5()
    figure6()
    print("=" * 50)
    print("Done. All figures saved to figures/")


if __name__ == '__main__':
    main()
