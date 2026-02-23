#!/usr/bin/env python3
"""Compare normalized vs non-normalized replicate data (vectorized)."""
import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr, pearsonr

# ── Non-normalized ──
df_raw = pd.read_csv('depmap_data/ccle_biological_replicates_nonnormalized.csv')
print(f'Non-normalized: {df_raw.shape}')

# Parse cell line groups
data_cols_raw = [c for c in df_raw.columns
                 if not c.startswith(('Protein.Id','Gene.Symbol','Description','TenPx'))]

cl_groups = {}
for c in data_cols_raw:
    m = re.match(r'(.+?)\.(TenPx\d+)\.(R\d+)', c)
    if m and m.group(1) != 'bridge':
        cl_groups.setdefault(m.group(1), []).append(c)

print(f'Cell lines: {len(cl_groups)}, data cols: {len(data_cols_raw)}')

# Vectorized CV: for each cell line group, compute CV across replicates
# Then average across cell lines per gene
genes = df_raw['Gene.Symbol'].values
n_genes = len(df_raw)
cv_sums = np.zeros(n_genes)
cv_counts = np.zeros(n_genes, dtype=int)

for cl, cols in cl_groups.items():
    vals = df_raw[cols].values.astype(float)  # (n_genes, n_reps)
    with np.errstate(invalid='ignore'):
        means = np.nanmean(vals, axis=1)
        stds = np.nanstd(vals, axis=1, ddof=1)
        n_valid = np.sum(~np.isnan(vals), axis=1)
        cvs = stds / means
        mask = (n_valid >= 2) & (means > 1e-9) & np.isfinite(cvs)
    cv_sums[mask] += cvs[mask]
    cv_counts[mask] += 1

raw_cv_dict = {}
for i in range(n_genes):
    g = genes[i]
    if cv_counts[i] > 0 and isinstance(g, str):
        raw_cv_dict[g] = cv_sums[i] / cv_counts[i]

print(f'Raw CVs computed for {len(raw_cv_dict)} genes')
raw_vals = list(raw_cv_dict.values())
print(f'  Median: {np.median(raw_vals):.4f}')
print(f'  Mean: {np.mean(raw_vals):.4f}')
print(f'  25th %ile: {np.percentile(raw_vals, 25):.4f}')
print(f'  75th %ile: {np.percentile(raw_vals, 75):.4f}')

# ── Normalized ──
df_norm = pd.read_excel(
    'depmap_data/Table_S3_Biological_Replicates_Protein_Quant_Normalized.xlsx',
    sheet_name='Replicates Expression'
)
print(f'\nNormalized: {df_norm.shape}')

meta_cols = {'Protein_Id','Gene_Symbol','Description','Group_ID','Uniprot','Uniprot_Acc'}
norm_data_cols = [c for c in df_norm.columns if c not in meta_cols]

# Group by cell line base
norm_cl_groups = {}
rep_pat = re.compile(r'^(.+?)[-_](R|Rep)(\d+)$', re.IGNORECASE)
pep_pat = re.compile(r'_Peptides$', re.IGNORECASE)
for c in norm_data_cols:
    if pep_pat.search(c):
        continue
    m = rep_pat.match(c)
    if m:
        norm_cl_groups.setdefault(m.group(1), []).append(c)

print(f'Cell lines: {len(norm_cl_groups)}')
genes_norm = df_norm['Gene_Symbol'].values
n_norm = len(df_norm)
cv_sums_n = np.zeros(n_norm)
cv_counts_n = np.zeros(n_norm, dtype=int)

for cl, cols in norm_cl_groups.items():
    vals = df_norm[cols].values.astype(float)
    with np.errstate(invalid='ignore'):
        means = np.nanmean(vals, axis=1)
        stds = np.nanstd(vals, axis=1, ddof=1)
        n_valid = np.sum(~np.isnan(vals), axis=1)
        cvs = stds / means
        mask = (n_valid >= 2) & (means > 1e-9) & np.isfinite(cvs)
    cv_sums_n[mask] += cvs[mask]
    cv_counts_n[mask] += 1

norm_cv_dict = {}
for i in range(n_norm):
    g = genes_norm[i]
    if cv_counts_n[i] > 0 and isinstance(g, str):
        norm_cv_dict[g] = cv_sums_n[i] / cv_counts_n[i]

print(f'Norm CVs computed for {len(norm_cv_dict)} genes')
norm_vals_all = list(norm_cv_dict.values())
print(f'  Median: {np.median(norm_vals_all):.4f}')
print(f'  Mean: {np.mean(norm_vals_all):.4f}')

# ── Correlation ──
shared = sorted(set(raw_cv_dict.keys()) & set(norm_cv_dict.keys()))
print(f'\nShared genes: {len(shared)}')
nv = np.array([norm_cv_dict[g] for g in shared])
rv = np.array([raw_cv_dict[g] for g in shared])
r_s, p_s = spearmanr(nv, rv)
r_p, p_p = pearsonr(nv, rv)
print(f'Spearman r={r_s:.3f}, p={p_s:.2e}')
print(f'Pearson  r={r_p:.3f}, p={p_p:.2e}')

# ── Key genes ──
sample_genes = ['EGFR','KRAS','TP53','BRAF','CDK4','MAP2K1','CDK6','BCL2L1','PIK3CA','STAT3']
print(f'\n{"Gene":<10} {"Norm_CV":<12} {"Raw_CV":<12} {"Raw/Norm":<10}')
for g in sample_genes:
    nc = norm_cv_dict.get(g, float('nan'))
    rc = raw_cv_dict.get(g, float('nan'))
    ratio = rc/nc if nc > 0 and np.isfinite(nc) and np.isfinite(rc) else float('nan')
    print(f'{g:<10} {nc:<12.4f} {rc:<12.4f} {ratio:<10.2f}')

# ── Bridge samples ──
bridge_cols = [c for c in data_cols_raw if c.startswith('bridge.')]
print(f'\nBridge columns: {len(bridge_cols)}')
if bridge_cols:
    bvals = df_raw[bridge_cols].values.astype(float)
    with np.errstate(invalid='ignore'):
        bmeans = np.nanmean(bvals, axis=1)
        bstds = np.nanstd(bvals, axis=1, ddof=1)
        bn = np.sum(~np.isnan(bvals), axis=1)
        bcvs = bstds / bmeans
        bmask = (bn >= 2) & (bmeans > 1e-9) & np.isfinite(bcvs)
    bridge_cv_vals = bcvs[bmask]
    print(f'Proteins with bridge CV: {len(bridge_cv_vals)}')
    print(f'Bridge median CV: {np.median(bridge_cv_vals):.4f}')
    print(f'Bridge mean CV: {np.mean(bridge_cv_vals):.4f}')

# ── Summary recommendation ──
print('\n=== SUMMARY ===')
print(f'Raw CV adds {len(raw_cv_dict) - len(shared)} genes not in normalized data')
extra_raw = set(raw_cv_dict.keys()) - set(norm_cv_dict.keys())
print(f'Extra genes in raw: {len(extra_raw)}')
print(f'Norm CV ≈ Raw CV correlation: Spearman={r_s:.3f}')
if r_s > 0.7:
    print('High correlation — raw CV confirms normalized CV but adds little new info')
    print('Recommendation: Use raw CV as cross-validation + fill gap genes')
else:
    print('Moderate correlation — raw CV provides complementary quality signal')
    print('Recommendation: Blend both CVs for composite confidence')
