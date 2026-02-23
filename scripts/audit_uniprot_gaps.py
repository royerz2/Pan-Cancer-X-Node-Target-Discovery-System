#!/usr/bin/env python3
"""Audit missing UniProt mappings and assess data coverage."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from alin.protein_scoring import GENE_TO_UNIPROT

# 1. Load missing genes from pipeline warnings
missing_file = Path("/tmp/missing_uniprot.txt")
if missing_file.exists():
    missing = set(missing_file.read_text().split())
else:
    missing = set()
    
print(f"Current GENE_TO_UNIPROT: {len(GENE_TO_UNIPROT)} genes")
print(f"Missing (fallback) genes from pipeline: {len(missing)}")
print()

# 2. Check what proteomics data we have
prot_candidates = [
    ROOT / "depmap_data" / "protein_quant_current_normalized.csv.gz",
    ROOT / "depmap_data" / "protein_quant_current_normalized.csv",
    ROOT / "depmap_data" / "proteomics.csv",
]
prot_file = None
for f in prot_candidates:
    if f.exists():
        prot_file = f
        break

if prot_file:
    print(f"Proteomics file: {prot_file.name} ({prot_file.stat().st_size/1e6:.1f} MB)")
    df = pd.read_csv(prot_file, nrows=3)
    print(f"  Columns (first 10): {list(df.columns[:10])}")
    
    # Get gene column
    gene_col = None
    for c in ['Gene_Symbol', 'Gene.Symbol', 'gene_name', 'Gene']:
        if c in df.columns:
            gene_col = c
            break
    if gene_col is None:
        gene_col = df.columns[0]
    
    df_genes = pd.read_csv(prot_file, usecols=[gene_col])
    genes_in_prot = set(df_genes[gene_col].dropna().unique())
    print(f"  Unique genes in proteomics: {len(genes_in_prot)}")
    
    rescued = missing & genes_in_prot
    print(f"  Missing genes WITH proteomics data: {len(rescued)}/{len(missing)} ({100*len(rescued)/len(missing):.0f}%)")
    print(f"  Missing genes WITHOUT proteomics: {len(missing - genes_in_prot)}")
else:
    genes_in_prot = set()
    print("No proteomics file found!")
    print(f"  Checked: {[str(f) for f in prot_candidates]}")

# 3. Check RNA-seq data
rna_candidates = [
    ROOT / "depmap_data" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
    ROOT / "depmap_data" / "expression.csv",
]
rna_file = None
for f in rna_candidates:
    if f.exists():
        rna_file = f
        break

print()
if rna_file:
    print(f"RNA-seq file: {rna_file.name} ({rna_file.stat().st_size/1e6:.1f} MB)")
    df_rna = pd.read_csv(rna_file, nrows=1)
    # Gene names are in column headers (format: "GENE (ENTREZ)")
    rna_genes = set()
    for c in df_rna.columns[1:]:
        gene = c.split(" (")[0].split(" ")[0] if " (" in c else c
        rna_genes.add(gene)
    print(f"  Unique genes in RNA-seq: {len(rna_genes)}")
    rescued_rna = missing & rna_genes
    print(f"  Missing genes WITH RNA data: {len(rescued_rna)}/{len(missing)} ({100*len(rescued_rna)/len(missing):.0f}%)")
else:
    rna_genes = set()
    print("No RNA-seq file found!")

# 4. Check what data from Gygi paper we could use
print()
print("=" * 70)
print("DATA GAP ANALYSIS")
print("=" * 70)
print()

# How many missing genes could be resolved by auto-resolving UniProt?
# Use standard gene name -> UniProt mapping via programmatic lookup
print(f"Total unique genes entering protein scorer: {len(GENE_TO_UNIPROT) + len(missing)}")
print(f"  Mapped (full 6-layer scoring): {len(GENE_TO_UNIPROT)} ({100*len(GENE_TO_UNIPROT)/(len(GENE_TO_UNIPROT)+len(missing)):.1f}%)")
print(f"  Fallback (score=0.3 flat): {len(missing)} ({100*len(missing)/(len(GENE_TO_UNIPROT)+len(missing)):.1f}%)")
print()

# The key question: do we need UniProt for all 6 layers?
print("Layer-by-layer data dependency:")
print("  1. Structural (AlphaFold pLDDT, PDB): NEEDS UniProt ID")
print("  2. Abundance (CCLE proteomics): NEEDS Gene_Symbol (NOT UniProt)")
print("  3. Degradability (PROTAC-DB): NEEDS Gene_Symbol")
print("  4. PPI surface (PDB complexes): NEEDS UniProt ID")  
print("  5. RNA expression (DepMap): NEEDS Gene_Symbol (NOT UniProt)")
print("  6. RNA-protein concordance: NEEDS Gene_Symbol")
print()
print("=> Layers 2,3,5,6 can work WITHOUT UniProt ID!")
print("=> Only layers 1 (structural) and 4 (PPI) strictly need UniProt")
print()

# What fraction of proteomics genes have UniProt in standard databases?
if genes_in_prot:
    both_prot_and_missing = genes_in_prot & missing
    print(f"Opportunity: {len(both_prot_and_missing)} genes have proteomics data")
    print(f"  but are scored as flat 0.3 because score_gene() requires UniProt")
    print(f"  for the full pipeline entry point.")
    print()
    
# High-value missing genes (appearing in many cancer pools)
print("Checking high-frequency missing genes in pipeline warnings...")
try:
    log = (ROOT / "results" / "pipeline_run_v2.log").read_text()
    from collections import Counter
    warn_counts = Counter()
    for line in log.split("\n"):
        if "No UniProt ID for" in line:
            gene = line.split("No UniProt ID for ")[1].split(";")[0].strip()
            warn_counts[gene] += 1
    
    print(f"\nTop 20 most frequently warned genes (across cancer types):")
    for gene, count in warn_counts.most_common(20):
        in_prot = "has proteomics" if gene in genes_in_prot else "NO proteomics"
        in_rna = "has RNA" if gene in rna_genes else "NO RNA"
        print(f"  {gene:15s}: {count:3d} warnings  ({in_prot}, {in_rna})")
except Exception as e:
    print(f"  Could not parse log: {e}")
