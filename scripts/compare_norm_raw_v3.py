#!/usr/bin/env python3
"""Compare normalized vs non-normalized replicate data — output to file."""
import pandas as pd
import numpy as np
import re
import warnings
import sys
warnings.filterwarnings('ignore')

out = open('scripts/nonnorm_analysis_results.txt', 'w')
def p(s):
    print(s)
    out.write(s + '\n')
    out.flush()

# ── Non-normalized ──
df_raw = pd.read_csv('depmap_data/ccle_biological_replicates_nonnormalized.csv')
p(f'Non-normalized: {df_raw.shape}')

data_cols_raw = [c for c in df_raw.columns
                 if not c.startswith(('Protein.Id','Gene.Symbol','Description','TenPx'))]

cl_groups = {}
for c in data_cols_raw:
    m = re.match(r'(.+?)\.(TenPx\d+)\.(R\d+)', c)
    if m and m.group(1) != 'bridge':
        cl_groups.setdefault(m.group(1), []).append(c)

p(f'Cell lines: {len(cl_groups)}, data cols: {len(data_cols_raw)}')

genes = df_raw['Gene.Symbol'].values
n_genes = len(df_raw)
cv_sums = np.zeros(n_genes)
cv_counts = np.zeros(n_genes, dtype=int)

for cl, cols in cl_groups.items():
    vals = df_raw[cols].values.astype(float)
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

raw_vals_arr = np.array(list(raw_cv_dict.values()))
p(f'Raw CVs: {len(raw_cv_dict)} genes')
p(f'  Median={np.median(raw_vals_arr):.4f}, Mean={np.mean(raw_vals_arr):.4f}')
p(f'  25th={np.percentile(raw_vals_arr,25):.4f}, 75th={np.percentile(raw_vals_arr,75):.4f}')

# ── Normalized ──
p('\nReading normalized xlsx...')
sys.stdout.flush()
df_norm = pd.read_excel(
    'depmap_data/Table_S3_Biological_Replicates_Protein_Quant_Normalized.xlsx',
    sheet_name='Replicates Expression'
)
p(f'Normalized: {df_norm.shape}')

meta_cols = {'Protein_Id','Gene_Symbol','Description','Group_ID','Uniprot','Uniprot_Acc'}
norm_data_cols = [c for c in df_norm.columns if c not in meta_cols]

norm_cl_groups = {}
rep_pat = re.compile(r'^(.+?)[-_](R|Rep)(\d+)$', re.IGNORECASE)
pep_pat = re.compile(r'_Peptides$', re.IGNORECASE)
for c in norm_data_cols:
    if pep_pat.search(c):
        continue
    m = rep_pat.match(c)
    if m:
        norm_cl_groups.setdefault(m.group(1), []).append(c)

p(f'Norm cell lines: {len(norm_cl_groups)}')

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

norm_vals_arr = np.array(list(norm_cv_dict.values()))
p(f'Norm CVs: {len(norm_cv_dict)} genes')
p(f'  Median={np.median(norm_vals_arr):.4f}, Mean={np.mean(norm_vals_arr):.4f}')

# ── Correlation ──
shared = sorted(set(raw_cv_dict.keys()) & set(norm_cv_dict.keys()))
p(f'\nShared genes: {len(shared)}')
nv = np.array([norm_cv_dict[g] for g in shared])
rv = np.array([raw_cv_dict[g] for g in shared])

from scipy.stats import spearmanr, pearsonr
r_s, p_s = spearmanr(nv, rv)
r_p, p_p = pearsonr(nv, rv)
p(f'Spearman r={r_s:.4f}, p={p_s:.2e}')
p(f'Pearson  r={r_p:.4f}, p={p_p:.2e}')

# ── Key genes ──
sample_genes = ['EGFR','KRAS','TP53','BRAF','CDK4','MAP2K1','CDK6','BCL2L1','PIK3CA','STAT3']
p(f'\n{"Gene":<10} {"Norm_CV":<12} {"Raw_CV":<12} {"Ratio":<10}')
for g in sample_genes:
    nc = norm_cv_dict.get(g, float('nan'))
    rc = raw_cv_dict.get(g, float('nan'))
    ratio = rc/nc if nc > 0 and np.isfinite(nc) and np.isfinite(rc) else float('nan')
    p(f'{g:<10} {nc:<12.4f} {rc:<12.4f} {ratio:<10.2f}')

# ── Bridge samples ──
bridge_cols = [c for c in data_cols_raw if c.startswith('bridge.')]
p(f'\nBridge columns: {len(bridge_cols)}')
if bridge_cols:
    bvals = df_raw[bridge_cols].values.astype(float)
    with np.errstate(invalid='ignore'):
        bmeans = np.nanmean(bvals, axis=1)
        bstds = np.nanstd(bvals, axis=1, ddof=1)
        bn = np.sum(~np.isnan(bvals), axis=1)
        bcvs = bstds / bmeans
        bmask = (bn >= 2) & (bmeans > 1e-9) & np.isfinite(bcvs)
    bridge_cv_vals = bcvs[bmask]
    p(f'Proteins with bridge CV: {len(bridge_cv_vals)}')
    p(f'Bridge median CV: {np.median(bridge_cv_vals):.4f}')

# ── Extra genes only in raw ──
extra_raw = set(raw_cv_dict.keys()) - set(norm_cv_dict.keys())
extra_norm = set(norm_cv_dict.keys()) - set(raw_cv_dict.keys())
p(f'\nExtra genes in raw only: {len(extra_raw)}')
p(f'Extra genes in norm only: {len(extra_norm)}')

p('\n=== RECOMMENDATION ===')
if r_s > 0.7:
    p(f'High Spearman ({r_s:.3f}): Raw CV strongly corroborates normalized CV')
    p('→ Use raw CV to fill gaps for genes missing in normalized, cross-validate')
elif r_s > 0.4:
    p(f'Moderate Spearman ({r_s:.3f}): Raw CV partially corroborates but adds info')
    p('→ Blend raw+norm CVs for composite confidence weight')
else:
    p(f'Low Spearman ({r_s:.3f}): Raw and norm CVs diverge significantly')
    p('→ Keep both as independent quality layers')

out.close()
p_final = 'DONE - results in scripts/nonnorm_analysis_results.txt'
print(p_final)
