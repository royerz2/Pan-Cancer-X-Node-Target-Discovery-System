#!/usr/bin/env python3
"""Quick validation that all 3 data loaders work with real downloaded files."""

import logging
import sys
sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from alin.protein_scoring import load_ccle_proteomics, load_ccle_rnaseq, load_protacdb

print('=' * 60)
print('TESTING DATA LOADERS WITH REAL FILES')
print('=' * 60)

# 1. Proteomics
print('\n=== Gygi CCLE Proteomics ===')
result = load_ccle_proteomics('./depmap_data')
if result:
    df, cmap = result
    print(f'  Shape: {df.shape}')
    print(f'  Index (first 5): {list(df.index[:5])}')
    print(f'  Columns (first 5): {list(df.columns[:5])}')
    print(f'  Cancer map entries: {len(cmap)}')
    if 'EGFR' in df.columns:
        print(f'  EGFR non-null: {df["EGFR"].notna().sum()}')
        print(f'  EGFR mean: {df["EGFR"].mean():.3f}')
    matched = sum(1 for idx in df.index if idx in cmap)
    print(f'  Cell lines matched to cancer: {matched}/{len(df)}')
else:
    print('  FAILED to load proteomics')

# 2. RNA-seq
print('\n=== DepMap CCLE RNA-seq ===')
result2 = load_ccle_rnaseq('./depmap_data')
if result2:
    df2, cmap2 = result2
    print(f'  Shape: {df2.shape}')
    print(f'  Index (first 5): {list(df2.index[:5])}')
    print(f'  Columns (first 5): {list(df2.columns[:5])}')
    print(f'  Cancer map entries: {len(cmap2)}')
    if 'EGFR' in df2.columns:
        print(f'  EGFR non-null: {df2["EGFR"].notna().sum()}')
        print(f'  EGFR mean: {df2["EGFR"].mean():.3f}')
    matched2 = sum(1 for idx in df2.index if idx in cmap2)
    print(f'  Cell lines matched to cancer: {matched2}/{len(df2)}')
else:
    print('  FAILED to load RNA-seq')

# 3. PROTAC-DB
print('\n=== PROTAC-DB Degrader Targets ===')
targets = load_protacdb('./depmap_data')
print(f'  Degrader targets: {len(targets)}')
for gene, info in list(targets.items())[:5]:
    print(f'    {gene}: {info["status"]} ({info["exemplar"]})')

print('\n' + '=' * 60)
print('LOADER VALIDATION COMPLETE')
print('=' * 60)
