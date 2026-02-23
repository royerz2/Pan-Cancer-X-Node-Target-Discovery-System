#!/usr/bin/env python3
"""Analyze non-normalized biological replicate data and compare with Table S3."""
import pandas as pd
import numpy as np
import re

# Load non-normalized data
df = pd.read_csv('depmap_data/ccle_biological_replicates_nonnormalized.csv')
print(f'Non-normalized shape: {df.shape}')
print(f'Genes: {df["Gene.Symbol"].nunique()}')

# Separate metadata cols and data cols
data_cols = [c for c in df.columns if not c.startswith(('Protein.Id','Gene.Symbol','Description','TenPx'))]
print(f'Data columns: {len(data_cols)}')

# Parse cell line names and replicates
cell_lines = set()
for c in data_cols:
    m = re.match(r'(.+)\.(TenPx\d+)\.(R\d+)', c)
    if m:
        cell_lines.add(m.group(1))
cell_lines = sorted(cell_lines)
print(f'Cell lines: {len(cell_lines)}')
print(f'First 5: {cell_lines[:5]}')

# Compare with Table S3 (normalized)
try:
    df_norm = pd.read_excel('depmap_data/Table_S3_Biological_Replicates_Protein_Quant_Normalized.xlsx', sheet_name=0)
    print(f'\nTable S3 (normalized) shape: {df_norm.shape}')
    print(f'Table S3 columns: {df_norm.columns[:8].tolist()}')
except FileNotFoundError:
    df_norm = None
    print('\nTable S3 normalized file not found')

# Quick CV comparison for a few genes in non-normalized
sample_genes = ['EGFR', 'KRAS', 'TP53', 'BRAF', 'CDK4', 'MAP2K1', 'CDK6', 'BCL2L1']
print('\n--- Per-gene raw CV (non-normalized) ---')
for gene in sample_genes:
    row = df[df['Gene.Symbol'] == gene]
    if len(row) == 0:
        print(f'{gene}: not found')
        continue
    cvs = []
    for cl in cell_lines:
        cl_cols = [c for c in data_cols if c.startswith(cl + '.')]
        if cl_cols:
            cl_vals = row[cl_cols].values.flatten()
            cl_vals = cl_vals[~np.isnan(cl_vals)]
            if len(cl_vals) >= 2 and np.mean(cl_vals) > 0:
                cvs.append(np.std(cl_vals, ddof=1) / np.mean(cl_vals))
    if cvs:
        print(f'{gene}: median raw CV={np.median(cvs):.3f}, mean={np.mean(cvs):.3f}, n={len(cvs)}')

# Compute genome-wide raw CV distribution
print('\n--- Genome-wide raw CV distribution ---')
all_cvs = {}
for _, row in df.iterrows():
    gene = row['Gene.Symbol']
    if pd.isna(gene):
        continue
    cvs = []
    for cl in cell_lines:
        cl_cols = [c for c in data_cols if c.startswith(cl + '.')]
        if cl_cols:
            cl_vals = row[cl_cols].values.flatten()
            cl_vals = cl_vals[~np.isnan(cl_vals)]
            if len(cl_vals) >= 2 and np.mean(cl_vals) > 0:
                cvs.append(np.std(cl_vals, ddof=1) / np.mean(cl_vals))
    if cvs:
        all_cvs[gene] = np.median(cvs)

cv_vals = list(all_cvs.values())
print(f'Genes with CV: {len(cv_vals)}')
print(f'Median CV: {np.median(cv_vals):.4f}')
print(f'Mean CV: {np.mean(cv_vals):.4f}')
print(f'25th percentile: {np.percentile(cv_vals, 25):.4f}')
print(f'75th percentile: {np.percentile(cv_vals, 75):.4f}')
print(f'Genes with CV < 0.2: {sum(1 for v in cv_vals if v < 0.2)}')
print(f'Genes with CV < 0.5: {sum(1 for v in cv_vals if v < 0.5)}')
print(f'Genes with CV > 1.0: {sum(1 for v in cv_vals if v > 1.0)}')

# Compare: non-normalized replicate data has raw intensities
# Table S3 has normalized CVs from the Gygi paper QC analysis
# The non-normalized data could provide:
# 1. Raw (pre-normalization) CV as a complementary quality metric
# 2. Bridge sample concordance (bridge columns present)
# 3. Cross-validation of Table S3 normalized CV values
print('\n--- Bridge sample analysis ---')
bridge_cols = [c for c in data_cols if c.startswith('bridge.')]
print(f'Bridge columns: {bridge_cols}')
if bridge_cols:
    bridge_vals = df[bridge_cols].values
    bridge_nonnan = np.sum(~np.isnan(bridge_vals), axis=0)
    print(f'Non-NaN per bridge col: {dict(zip(bridge_cols, bridge_nonnan))}')
    
    # Bridge CV per gene
    bridge_cvs = []
    for _, row in df.iterrows():
        bv = row[bridge_cols].values.flatten()
        bv = bv[~np.isnan(bv)]
        if len(bv) >= 2 and np.mean(bv) > 0:
            bridge_cvs.append(np.std(bv, ddof=1) / np.mean(bv))
    print(f'Bridge median CV: {np.median(bridge_cvs):.4f}')
    print(f'Bridge mean CV: {np.mean(bridge_cvs):.4f}')
