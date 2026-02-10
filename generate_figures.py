#!/usr/bin/env python3
"""Generate publication-quality figures for the ALIN Framework paper."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
triples = pd.read_csv('results/triple_combinations_merged.csv')
mhs_full = pd.read_csv('supplementary_tables/Table_S1_MHS_full.csv')
benchmark = pd.read_csv('supplementary_tables/Table_S2_benchmark.csv')

# ── Tri-axial role color mapping ─────────────────────────────────────────────
ROLE_MAP = {
    # Orthogonal (green)
    'STAT3': 'orthogonal', 'FYN': 'orthogonal', 'SRC': 'orthogonal',
    'JAK1': 'orthogonal', 'JAK2': 'orthogonal', 'BCL2': 'orthogonal',
    'BCL2L1': 'orthogonal', 'MCL1': 'orthogonal',
    # Downstream effector (blue)
    'CCND1': 'downstream', 'CDK4': 'downstream', 'CDK6': 'downstream',
    'CDK2': 'downstream', 'RB1': 'downstream',
    # Upstream driver (orange)
    'KRAS': 'upstream', 'EGFR': 'upstream', 'BRAF': 'upstream',
    'MAP2K1': 'upstream', 'MET': 'upstream', 'ERBB2': 'upstream',
    'FGFR1': 'upstream', 'PIK3CA': 'upstream', 'ALK': 'upstream',
}
ROLE_COLORS = {
    'orthogonal': '#2ca02c',   # green
    'downstream': '#1f77b4',   # blue
    'upstream': '#ff7f0e',     # orange
    'unknown': '#999999',      # grey
}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Cancer × Target heatmap (convergent architecture)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_cancer_target_heatmap():
    """Heatmap showing which targets appear in which cancer's ranked triples."""
    # Build cancer × target matrix from triples
    targets_all = []
    for col in ['Target_1', 'Target_2', 'Target_3']:
        targets_all.extend(triples[col].dropna().unique())
    target_genes = sorted(set(targets_all))

    # Count frequency per target
    from collections import Counter
    freq = Counter()
    for _, row in triples.iterrows():
        for col in ['Target_1', 'Target_2', 'Target_3']:
            if pd.notna(row[col]):
                freq[row[col]] += 1

    # Select top 12 most frequent targets
    top_targets = [g for g, _ in freq.most_common(12)]

    # Select 20 most well-powered cancers (by cell line count)
    cancer_lines = triples[['Cancer_Type', 'Cell_Lines']].drop_duplicates()
    cancer_lines['Cell_Lines'] = pd.to_numeric(cancer_lines['Cell_Lines'], errors='coerce')
    top_cancers = cancer_lines.nlargest(20, 'Cell_Lines')['Cancer_Type'].tolist()

    # Build binary matrix
    matrix = np.zeros((len(top_cancers), len(top_targets)))
    for i, cancer in enumerate(top_cancers):
        row_data = triples[triples['Cancer_Type'] == cancer].iloc[0]
        for j, target in enumerate(top_targets):
            if target in [row_data.get('Target_1'), row_data.get('Target_2'), row_data.get('Target_3')]:
                matrix[i, j] = 1.0

    # Shorten cancer names for display
    short_names = {
        'Non-Small Cell Lung Cancer': 'NSCLC',
        'Invasive Breast Carcinoma': 'Breast',
        'Colorectal Adenocarcinoma': 'CRC',
        'Pancreatic Adenocarcinoma': 'PDAC',
        'Hepatocellular Carcinoma': 'HCC',
        'Ovarian Epithelial Tumor': 'Ovarian',
        'Head and Neck Squamous Cell Carcinoma': 'HNSCC',
        'Endometrial Carcinoma': 'Endometrial',
        'Skin Cancer, Non-Melanoma': 'Non-Mel. Skin',
        'Bladder Urothelial Carcinoma': 'Bladder',
        'Esophagogastric Adenocarcinoma': 'Esophagogastric',
        'Prostate Adenocarcinoma': 'Prostate',
        'B-Lymphoblastic Leukemia/Lymphoma': 'B-ALL',
        'Acute Myeloid Leukemia': 'AML',
        'Rhabdomyosarcoma': 'RMS',
        'T-Lymphoblastic Leukemia/Lymphoma': 'T-ALL',
        'Neuroblastoma': 'NBL',
        'Myeloproliferative Neoplasms': 'MPN',
    }
    display_names = [short_names.get(c, c[:18]) for c in top_cancers]

    # Custom colormap: white → teal
    cmap = LinearSegmentedColormap.from_list('presence', ['#f7f7f7', '#1a9876'], N=2)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(top_targets)))
    ax.set_xticklabels(top_targets, rotation=45, ha='right', fontweight='bold')
    ax.set_yticks(range(len(top_cancers)))
    ax.set_yticklabels(display_names)

    # Color x-tick labels by tri-axial role
    for i, label in enumerate(ax.get_xticklabels()):
        role = ROLE_MAP.get(top_targets[i], 'unknown')
        label.set_color(ROLE_COLORS[role])

    # Add cell annotations
    for i in range(len(top_cancers)):
        for j in range(len(top_targets)):
            if matrix[i, j] > 0:
                ax.text(j, i, '●', ha='center', va='center', fontsize=10,
                        color='white', fontweight='bold')

    # Add frequency bar along top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(len(top_targets)))
    freq_labels = [f'{freq[g]}/{len(triples)}' for g in top_targets]
    ax2.set_xticklabels(freq_labels, fontsize=6.5, color='#555555')
    ax2.tick_params(length=0)

    # Grid
    ax.set_xticks([x - 0.5 for x in range(1, len(top_targets))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(top_cancers))], minor=True)
    ax.grid(True, which='minor', color='#e0e0e0', linewidth=0.5)
    ax.tick_params(which='minor', length=0)

    # Legend for tri-axial roles
    patches = [
        mpatches.Patch(color=ROLE_COLORS['upstream'], label='Upstream driver'),
        mpatches.Patch(color=ROLE_COLORS['downstream'], label='Downstream effector'),
        mpatches.Patch(color=ROLE_COLORS['orthogonal'], label='Orthogonal survival'),
    ]
    ax.legend(handles=patches, loc='lower right', frameon=True, framealpha=0.9,
              edgecolor='#cccccc', fontsize=7)

    ax.set_title('Ranked Triple Targets Across 20 Most Well-Powered Cancers', pad=22)

    fig.savefig(os.path.join(OUT, 'fig5_cancer_target_heatmap.png'))
    fig.savefig(os.path.join(OUT, 'fig5_cancer_target_heatmap.pdf'))
    plt.close(fig)
    print('✓ fig5_cancer_target_heatmap')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: MHS druggability landscape (combined 3-panel)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_mhs_and_frequency():
    """Two-panel figure: (A) MHS size distribution, (B) target frequency."""
    from collections import Counter

    mhs_cancers = mhs_full.drop_duplicates('Cancer_Type')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.96,
                        top=0.92, bottom=0.13)

    # ── Panel A: MHS size distribution ──
    sizes = mhs_cancers['Combination_Size'].value_counts().sort_index()
    colors_bar = ['#66c2a5', '#1f77b4', '#ff7f0e', '#d62728']
    bars = ax1.bar(sizes.index, sizes.values, color=colors_bar[:len(sizes)],
                   edgecolor='white', linewidth=0.8, width=0.65)
    for bar, count in zip(bars, sizes.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{count}\n({count/len(mhs_cancers)*100:.0f}%)',
                 ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax1.set_xlabel('MHS Size (number of targets)')
    ax1.set_ylabel('Number of cancer types')
    ax1.set_title('A  MHS Size Distribution', loc='left', fontweight='bold')
    ax1.set_xticks(sizes.index)
    ax1.set_ylim(0, max(sizes.values) * 1.3)

    # ── Panel B: Target frequency (top 10) ──
    freq = Counter()
    for _, row in mhs_full.drop_duplicates(['Cancer_Type', 'Target_Gene']).iterrows():
        freq[row['Target_Gene']] += 1
    top10 = freq.most_common(10)
    genes = [g for g, _ in top10]
    counts = [c for _, c in top10]
    n_cancers = mhs_cancers.shape[0]
    bar_colors = [ROLE_COLORS.get(ROLE_MAP.get(g, 'unknown'), '#999') for g in genes]
    bars2 = ax2.barh(range(len(genes)), counts, color=bar_colors,
                      edgecolor='white', linewidth=0.8, height=0.65)
    for i, (bar, count) in enumerate(zip(bars2, counts)):
        ax2.text(bar.get_width() + 0.5, i,
                 f'{count/n_cancers*100:.0f}%', va='center', fontsize=7)
    ax2.set_yticks(range(len(genes)))
    ax2.set_yticklabels(genes, fontweight='bold', fontsize=7.5)
    ax2.invert_yaxis()
    ax2.set_xlabel('Number of cancer types')
    ax2.set_title('B  Target Frequency in MHS', loc='left', fontweight='bold')
    patches = [
        mpatches.Patch(color=ROLE_COLORS['orthogonal'], label='Orthogonal'),
        mpatches.Patch(color=ROLE_COLORS['downstream'], label='Downstream'),
        mpatches.Patch(color=ROLE_COLORS['upstream'], label='Upstream'),
        mpatches.Patch(color=ROLE_COLORS['unknown'], label='Other'),
    ]
    ax2.legend(handles=patches, loc='lower right', frameon=True,
               framealpha=0.9, edgecolor='#ccc', fontsize=6)

    fig.savefig(os.path.join(OUT, 'fig6_mhs_distribution.png'))
    fig.savefig(os.path.join(OUT, 'fig6_mhs_distribution.pdf'))
    plt.close(fig)
    print('✓ fig6_mhs_distribution')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Benchmark match waterfall
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_benchmark_waterfall():
    """Waterfall chart showing gold-standard match type per entry."""
    df = benchmark.copy()

    # Order: exact > superset > pair_overlap > any_overlap > none
    match_order = {'exact': 4, 'superset': 3, 'pair_overlap': 2, 'any_overlap': 1, 'none': 0}
    match_colors = {
        'exact': '#1b7837',
        'superset': '#5aae61',
        'pair_overlap': '#a6dba0',
        'any_overlap': '#d9f0d3',
        'none': '#e0e0e0',
    }

    df['match_rank'] = df['Match_Type'].map(match_order)
    df = df.sort_values(['match_rank', 'Cancer_Type'], ascending=[False, True])
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (_, row) in enumerate(df.iterrows()):
        mt = row['Match_Type']
        color = match_colors.get(mt, '#ccc')
        ax.barh(i, 1, color=color, edgecolor='white', linewidth=0.3)

    # Short cancer name + gold targets as labels
    labels = []
    short = {
        'Non-Small Cell Lung Cancer': 'NSCLC',
        'Invasive Breast Carcinoma': 'Breast',
        'Colorectal Adenocarcinoma': 'CRC',
        'Pancreatic Adenocarcinoma': 'PDAC',
        'Hepatocellular Carcinoma': 'HCC',
        'Acute Myeloid Leukemia': 'AML',
        'Ampullary Cancer': 'Ampullary',
        'Renal Cell Carcinoma': 'RCC',
        'Lung Neuroendocrine Tumor': 'Lung NE',
        'Head and Neck Squamous Cell Carcinoma': 'HNSCC',
        'Diffuse Glioma': 'Glioma',
        'Synovial Sarcoma': 'Syn. Sarc.',
        'Adenosquamous Carcinoma of the Pancreas': 'PDAC (Adenosq.)',
    }
    for _, row in df.iterrows():
        cname = short.get(row['Cancer_Type'], row['Cancer_Type'][:16])
        gold = row['Gold_Targets']
        labels.append(f'{cname}: {gold}')

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.6)
    ax.set_xticks([])

    # Add match type text on bars
    for i, (_, row) in enumerate(df.iterrows()):
        mt = row['Match_Type']
        label_text = mt.replace('_', '-')
        if mt == 'none':
            our = row.get('Our_Targets', '')
            if pd.isna(our) or our.strip() == '':
                label_text = 'unmatchable'
            else:
                label_text = 'no match'
        color = '#333' if mt != 'none' else '#888'
        ax.text(1.05, i, label_text, va='center', fontsize=5.5, color=color,
                fontstyle='italic')

    # Legend
    handles = [mpatches.Patch(color=match_colors[k], label=k.replace('_', '-'))
               for k in ['exact', 'superset', 'pair_overlap', 'any_overlap', 'none']]
    ax.legend(handles=handles, loc='lower right', frameon=True, fontsize=6.5,
              framealpha=0.95, edgecolor='#ccc')

    ax.set_title('Gold-Standard Benchmark Matches (43 entries)', fontweight='bold')
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig7_benchmark_waterfall.png'))
    fig.savefig(os.path.join(OUT, 'fig7_benchmark_waterfall.pdf'))
    plt.close(fig)
    print('✓ fig7_benchmark_waterfall')


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating publication figures...')
    fig5_cancer_target_heatmap()
    fig6_mhs_and_frequency()
    fig7_benchmark_waterfall()
    print('Done. Figures saved to', OUT)
