#!/usr/bin/env python3
"""Regenerate fig2_xnode_concept with fixed annotation overlap."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from pathlib import Path

# Style
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
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

C = {
    'upstream': '#D55E00',
    'downstream': '#0072B2',
    'orthogonal': '#009E73',
    'text': '#333333',
    'grid': '#DDDDDD',
}

BASE = Path(__file__).parent
FIGURES_DIR = BASE / 'figures'

# Build target frequency from MHS data
mhs_file = BASE / 'supplementary_tables' / 'Table_S1_MHS_full.csv'
if not mhs_file.exists():
    print(f"ERROR: {mhs_file} not found")
    exit(1)

mhs_full = pd.read_csv(mhs_file)
# Count unique cancer types per target gene
from collections import Counter
freq = Counter()
for _, row in mhs_full.drop_duplicates(['Cancer_Type', 'Target_Gene']).iterrows():
    freq[row['Target_Gene']] += 1
n_cancers = mhs_full['Cancer_Type'].nunique()

# Build a DataFrame matching expected format
freq_items = freq.most_common(15)
df = pd.DataFrame(freq_items, columns=['Target_Gene', 'Cancer_Types_Count'])

fig = plt.figure(figsize=(10.0, 7.0))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.2], wspace=0.45)

# ---- Panel A: Target frequency ----
ax = fig.add_subplot(gs[0])
ax.text(-0.08, 1.06, 'A', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='right', color=C['text'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

top = df.head(15).iloc[::-1]
genes = top['Target_Gene'].values
counts = top['Cancer_Types_Count'].values

axis_map = {
    'STAT3': C['orthogonal'], 'FYN': C['orthogonal'], 'MCL1': C['orthogonal'],
    'CDK6': C['downstream'], 'CDK4': C['downstream'], 'CDK2': C['downstream'],
    'CCND1': C['downstream'],
    'KRAS': C['upstream'], 'EGFR': C['upstream'], 'BRAF': C['upstream'],
    'MAP2K1': C['upstream'], 'MET': C['upstream'],
    'FGFR1': C['upstream'], 'SRC': C['orthogonal'], 'ERBB2': C['upstream'],
    'YES1': C['orthogonal'], 'BCL2': C['orthogonal'],
}
colors = [axis_map.get(g, '#999') for g in genes]

bars = ax.barh(range(len(genes)), counts, color=colors,
               edgecolor='white', linewidth=1.0, height=0.75)
ax.set_yticks(range(len(genes)))
ax.set_yticklabels(genes, fontsize=9.5, fontweight='bold')
ax.set_xlabel('Cancer types with target', fontsize=10.5, fontweight='medium')
ax.set_title('Target frequency across cancers', fontsize=12, pad=10, fontweight='bold')
ax.grid(axis='x', alpha=0.25, color=C['grid'], linestyle='--')

# Value labels — for STAT3, place count + percentage INSIDE the bar
for bar, val, gene in zip(bars, counts, genes):
    if gene == 'STAT3':
        stat3_pct = int(round(val / n_cancers * 100))
        # Both count and percentage inside the bar
        ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height()/2,
                f'{int(val)}  ({stat3_pct}% of cancers)',
                va='center', ha='right', fontsize=8,
                color='white', fontweight='bold', style='italic')
    else:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(int(val)), va='center', fontsize=8, color='#444',
                fontweight='medium')

# No separate STAT3 annotation needed — it's inside the bar now

# ---- Panel B: Tri-axial classification (vertical layout with brackets) ----
ax2 = fig.add_subplot(gs[1])
ax2.text(-0.08, 1.06, 'B', transform=ax2.transAxes,
         fontsize=14, fontweight='bold', va='top', ha='right', color=C['text'])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('Tri-axial classification', fontsize=12, pad=10, fontweight='bold')

# All genes in order: orthogonal, then downstream, then upstream
axes_data = [
    ('Orthogonal (survival)', C['orthogonal'],
     [('STAT3', 70), ('FYN', 38), ('MCL1', 7), ('SRC', 2)]),
    ('Downstream (effector)', C['downstream'],
     [('CDK6', 34), ('CDK4', 14), ('CDK2', 7), ('CCND1', 5)]),
    ('Upstream (driver)', C['upstream'],
     [('KRAS', 23), ('EGFR', 12), ('MAP2K1', 8), ('BRAF', 6)]),
]

# Flatten all genes into a single ordered list with colors
all_genes = []
all_counts = []
all_colors_b = []
group_ranges = []  # (start_idx, end_idx, label, color)

idx = 0
for label, color, gene_list in axes_data:
    start = idx
    for gene, cnt in gene_list:
        all_genes.append(gene)
        all_counts.append(cnt)
        all_colors_b.append(color)
        idx += 1
    group_ranges.append((start, idx - 1, label, color))

n_genes_b = len(all_genes)
# Reverse for horizontal bar (top gene at top)
all_genes_r = all_genes[::-1]
all_counts_r = all_counts[::-1]
all_colors_r = all_colors_b[::-1]

bars2 = ax2.barh(range(n_genes_b), all_counts_r, color=all_colors_r,
                 edgecolor='white', linewidth=1.0, height=0.72)
ax2.set_yticks(range(n_genes_b))
ax2.set_yticklabels(all_genes_r, fontsize=9, fontweight='bold')
ax2.set_xlabel('Cancer types', fontsize=10)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.25, color=C['grid'], linestyle='--')

# Value labels on bars
for bar, val in zip(bars2, all_counts_r):
    ax2.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
             str(int(val)), va='center', fontsize=7.5, color='#444',
             fontweight='medium')

# Draw bracket annotations on the right side
max_count = max(all_counts)
x_bracket = max_count + 6  # bracket x position

for start, end, label, color in group_ranges:
    # Convert to reversed indices
    y_top = n_genes_b - 1 - end
    y_bot = n_genes_b - 1 - start
    y_mid = (y_top + y_bot) / 2

    # Draw bracket line
    ax2.plot([x_bracket, x_bracket], [y_top - 0.1, y_bot + 0.1],
             color=color, linewidth=2.5, solid_capstyle='round', clip_on=False)
    # Top cap
    ax2.plot([x_bracket - 0.8, x_bracket], [y_top - 0.1, y_top - 0.1],
             color=color, linewidth=2.5, solid_capstyle='round', clip_on=False)
    # Bottom cap
    ax2.plot([x_bracket - 0.8, x_bracket], [y_bot + 0.1, y_bot + 0.1],
             color=color, linewidth=2.5, solid_capstyle='round', clip_on=False)
    # Label
    ax2.text(x_bracket + 1.5, y_mid, label, va='center', ha='left',
             fontsize=8, color=color, fontweight='bold', clip_on=False,
             rotation=0)

# Expand xlim to fit bracket annotations
ax2.set_xlim(0, max_count + 25)

# Legend
legend_elements = [
    Line2D([0], [0], color=C['orthogonal'], lw=8, alpha=0.7, label='Orthogonal'),
    Line2D([0], [0], color=C['downstream'], lw=8, alpha=0.7, label='Downstream'),
    Line2D([0], [0], color=C['upstream'], lw=8, alpha=0.7, label='Upstream'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True,
          fancybox=True, framealpha=0.95, fontsize=9, title='Axis role',
          title_fontsize=9, edgecolor='#CCC')

for ext in ('png', 'pdf'):
    fig.savefig(FIGURES_DIR / f'fig2_xnode_concept.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("✓ fig2_xnode_concept regenerated with fixed annotation")
