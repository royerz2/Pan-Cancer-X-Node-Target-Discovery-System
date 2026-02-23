#!/usr/bin/env python3
"""Quick validation of all v3 data loaders against real Gygi data files."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import alin.protein_scoring as ps

# Reset global state for clean test
ps._PROTEOMICS_UNIPROT_LOADED = False
ps._PROTEOMICS_UNIPROT_MAP = {}

# 1. Proteomics UniProt map
umap = ps._load_proteomics_uniprot_map('./depmap_data')
print(f'1. Proteomics UniProt map: {len(umap)} genes')
for g in ['EGFR', 'SOX9', 'MEAF6', 'GJA3', 'DET1', 'MAEA']:
    print(f'   {g} -> {umap.get(g, "NOT FOUND")}')

# 2. Gygi correlations (Table S4)
corr = ps.load_gygi_correlations('./depmap_data')
if corr:
    print(f'\n2. Gygi correlations: {len(corr)} genes')
    for g in list(corr)[:5]:
        print(f'   {g}: {corr[g]}')
else:
    print('\n2. Gygi correlations: NONE')

# 3. Replicate CV (Table S3)
cv = ps.load_gygi_replicate_cv('./depmap_data')
if cv:
    vals = list(cv.values())
    print(f'\n3. Replicate CV: {len(cv)} genes')
    print(f'   median={np.median(vals):.3f}, mean={np.mean(vals):.3f}')
    print(f'   min={np.min(vals):.3f}, max={np.max(vals):.3f}')
else:
    print('\n3. Replicate CV: NONE')

# 4. Mutation associations (Table S7)
mut = ps.load_gygi_mutations('./depmap_data')
if mut:
    sig = sum(1 for entries in mut.values() for e in entries if e.get('fdr', 1) < 0.1)
    print(f'\n4. Mutation associations: {len(mut)} genes')
    print(f'   Significant (FDR<0.1): {sig} associations')
    for g in list(mut)[:3]:
        print(f'   {g}: {len(mut[g])} entries')
else:
    print('\n4. Mutation associations: NONE')

# 5. Coverage analysis: how many previously-missing genes are now resolved?
from alin.protein_scoring import GENE_TO_UNIPROT
static_genes = set(GENE_TO_UNIPROT.keys())
prot_genes = set(umap.keys())
new_coverage = prot_genes - static_genes
print(f'\n5. Coverage improvement:')
print(f'   Static GENE_TO_UNIPROT: {len(static_genes)} genes')
print(f'   Proteomics Uniprot_Acc: {len(prot_genes)} genes')
print(f'   NEW genes covered: {len(new_coverage)}')
print(f'   Total coverage: {len(static_genes | prot_genes)} genes')
