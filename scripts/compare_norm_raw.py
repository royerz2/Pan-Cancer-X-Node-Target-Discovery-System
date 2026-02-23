#!/usr/bin/env python3
"""Compare normalized vs non-normalized replicate data, compute dual CVs."""
import pandas as pd
import numpy as np
import re

# Read normalized Table S3
df_norm = pd.read_excel(
    'depmap_data/Table_S3_Biological_Replicates_Protein_Quant_Normalized.xlsx',
    sheet_name='Replicates Expression'
)
print(f'Normalized: {df_norm.shape}')
print(f'Genes: {df_norm["Gene_Symbol"].nunique()}')
data_cols_norm = [c for c in df_norm.columns
                  if c not in ['Protein_Id','Gene_Symbol','Description',
                               'Group_ID','Uniprot','Uniprot_Acc']]
print(f'Data cols ({len(data_cols_norm)}): {data_cols_norm[:5]}...')

# Read non-normalized
df_raw = pd.read_csv('depmap_data/ccle_biological_replicates_nonnormalized.csv')
print(f'\nNon-normalized: {df_raw.shape}')
print(f'Genes: {df_raw["Gene.Symbol"].nunique()}')
data_cols_raw = [c for c in df_raw.columns
                 if not c.startswith(('Protein.Id','Gene.Symbol','Description','TenPx'))]
print(f'Data cols ({len(data_cols_raw)}): {data_cols_raw[:5]}...')

# Gene overlap
norm_genes = set(df_norm['Gene_Symbol'].dropna().unique())
raw_genes = set(df_raw['Gene.Symbol'].dropna().unique())
print(f'\nNorm-only genes: {len(norm_genes - raw_genes)}')
print(f'Raw-only genes: {len(raw_genes - norm_genes)}')
print(f'Overlap: {len(norm_genes & raw_genes)}')

# Parse cell lines from non-normalized
raw_cell_lines = set()
for c in data_cols_raw:
    m = re.match(r'(.+?)\.(TenPx\d+)\.(R\d+)', c)
    if m and m.group(1) != 'bridge':
        raw_cell_lines.add(m.group(1))
raw_cell_lines = sorted(raw_cell_lines)

# Parse cell lines from normalized
norm_cell_lines_reps = {}
rep_pattern = re.compile(r'^(.+?)[-_](R|Rep)(\d+)$', re.IGNORECASE)
for c in data_cols_norm:
    m = rep_pattern.match(c)
    if m:
        base = m.group(1)
        rep = int(m.group(3))
        norm_cell_lines_reps.setdefault(base, {})[rep] = c
norm_cell_lines = sorted(norm_cell_lines_reps.keys())
print(f'\nNorm cell lines: {len(norm_cell_lines)}')
print(f'Raw cell lines: {len(raw_cell_lines)}')

# Compute per-gene CV for both datasets
def compute_cv_from_raw(df, gene_col, data_cols, cell_lines):
    """Compute per-gene mean CV across cell lines from raw replicate data."""
    result = {}
    for _, row in df.iterrows():
        gene = str(row.get(gene_col, '')).strip()
        if not gene or gene == 'nan':
            continue
        cvs = []
        for cl in cell_lines:
            cl_cols = [c for c in data_cols if c.startswith(cl + '.')]
            if len(cl_cols) < 2:
                continue
            vals = []
            for c in cl_cols:
                v = row.get(c)
                try:
                    v = float(v)
                    if not np.isnan(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    pass
            if len(vals) >= 2:
                mean_v = np.mean(vals)
                if abs(mean_v) > 1e-9:
                    cvs.append(np.std(vals, ddof=1) / abs(mean_v))
        if cvs:
            result[gene] = float(np.mean(cvs))
    return result

def compute_cv_from_norm(df, gene_col, cell_line_reps):
    """Compute per-gene mean CV across cell lines from normalized data."""
    result = {}
    for _, row in df.iterrows():
        gene = str(row.get(gene_col, '')).strip()
        if not gene or gene == 'nan':
            continue
        cvs = []
        for base, reps in cell_line_reps.items():
            if len(reps) < 2:
                continue
            vals = []
            for rep_num, col_name in sorted(reps.items()):
                v = row.get(col_name)
                try:
                    v = float(v)
                    if not np.isnan(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    pass
            if len(vals) >= 2:
                mean_v = np.mean(vals)
                if abs(mean_v) > 1e-9:
                    cvs.append(np.std(vals, ddof=1) / abs(mean_v))
        if cvs:
            result[gene] = float(np.mean(cvs))
    return result

print('\nComputing normalized CVs...')
cv_norm = compute_cv_from_norm(df_norm, 'Gene_Symbol', norm_cell_lines_reps)
print(f'Genes with normalized CV: {len(cv_norm)}')
print(f'Median normalized CV: {np.median(list(cv_norm.values())):.4f}')

print('\nComputing raw CVs...')
cv_raw = compute_cv_from_raw(df_raw, 'Gene.Symbol', data_cols_raw, raw_cell_lines)
print(f'Genes with raw CV: {len(cv_raw)}')
print(f'Median raw CV: {np.median(list(cv_raw.values())):.4f}')

# Correlation between normalized and raw CV for shared genes
shared = set(cv_norm.keys()) & set(cv_raw.keys())
print(f'\nShared genes: {len(shared)}')
if shared:
    norm_vals = [cv_norm[g] for g in shared]
    raw_vals = [cv_raw[g] for g in shared]
    from scipy.stats import spearmanr, pearsonr
    r_s, p_s = spearmanr(norm_vals, raw_vals)
    r_p, p_p = pearsonr(norm_vals, raw_vals)
    print(f'Spearman r={r_s:.3f}, p={p_s:.2e}')
    print(f'Pearson r={r_p:.3f}, p={p_p:.2e}')
    
    # Show key genes
    sample_genes = ['EGFR', 'KRAS', 'TP53', 'BRAF', 'CDK4', 'MAP2K1', 'CDK6', 
                    'BCL2L1', 'PIK3CA', 'STAT3']
    print('\n--- Key gene CVs ---')
    print(f'{"Gene":<10} {"Norm_CV":<10} {"Raw_CV":<10} {"Ratio":<10}')
    for g in sample_genes:
        nc = cv_norm.get(g, float('nan'))
        rc = cv_raw.get(g, float('nan'))
        ratio = rc / nc if nc > 0 and not np.isnan(nc) and not np.isnan(rc) else float('nan')
        print(f'{g:<10} {nc:<10.4f} {rc:<10.4f} {ratio:<10.2f}')

# Bridge sample analysis
print('\n--- Bridge sample analysis ---')
bridge_cols = [c for c in data_cols_raw if c.startswith('bridge.')]
print(f'Bridge columns: {len(bridge_cols)}')
if bridge_cols:
    bridge_cvs = []
    for _, row in df_raw.iterrows():
        vals = []
        for c in bridge_cols:
            v = row.get(c)
            try:
                v = float(v)
                if not np.isnan(v):
                    vals.append(v)
            except (ValueError, TypeError):
                pass
        if len(vals) >= 2 and np.mean(vals) > 1e-9:
            bridge_cvs.append(np.std(vals, ddof=1) / np.mean(vals))
    print(f'Proteins with bridge CV: {len(bridge_cvs)}')
    print(f'Bridge median CV: {np.median(bridge_cvs):.4f}')
    print(f'Bridge mean CV: {np.mean(bridge_cvs):.4f}')
