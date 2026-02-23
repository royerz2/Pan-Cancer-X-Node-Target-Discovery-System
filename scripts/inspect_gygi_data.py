#!/usr/bin/env python3
"""Inspect the new Gygi data files to understand structure and column names."""
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "depmap_data"

def inspect(path, name, nrows=3):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  File: {path.name} ({path.stat().st_size/1e6:.1f} MB)")
    print(f"{'='*70}")
    
    if path.suffix == '.xlsx':
        xls = pd.ExcelFile(path)
        print(f"  Sheets: {xls.sheet_names}")
        for sheet in xls.sheet_names[:3]:
            df = pd.read_excel(path, sheet_name=sheet, nrows=nrows)
            print(f"\n  Sheet '{sheet}': {df.shape[0]}+ rows x {df.shape[1]} cols")
            print(f"  Columns: {list(df.columns)}")
            print(f"  First row:")
            for c in df.columns[:15]:
                print(f"    {c}: {df[c].iloc[0] if len(df) > 0 else 'EMPTY'}")
    else:
        df = pd.read_csv(path, nrows=nrows)
        print(f"  Shape preview: {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns[:20])}")
        if len(df.columns) > 20:
            print(f"    ... and {len(df.columns)-20} more")
        print(f"  First row:")
        for c in list(df.columns[:15]):
            print(f"    {c}: {df[c].iloc[0] if len(df) > 0 else 'EMPTY'}")

# 1. RNA/Protein Correlation
inspect(DATA / "Table_S4_Protein_RNA_Correlation_and_Enrichments.xlsx",
        "RNA/PROTEIN CORRELATION")

# 2. Biological Replicates (normalized)
inspect(DATA / "Table_S3_Biological_Replicates_Protein_Quant_Normalized.xlsx",
        "BIOLOGICAL REPLICATES (Normalized)")

# 3. Biological Replicates (non-normalized)
inspect(DATA / "ccle_biological_replicates_nonnormalized.csv",
        "BIOLOGICAL REPLICATES (Non-Normalized)")

# 4. Mutation Associations
inspect(DATA / "Table_S7_Mutation_Associations.xlsx",
        "MUTATION ASSOCIATIONS")

# 5. Also inspect the Uniprot_Acc column in existing proteomics
print(f"\n{'='*70}")
print(f"  EXISTING PROTEOMICS - Uniprot_Acc column")
print(f"{'='*70}")
df = pd.read_csv(DATA / "protein_quant_current_normalized.csv.gz", 
                 usecols=['Gene_Symbol', 'Uniprot', 'Uniprot_Acc'], nrows=10)
print(f"  Columns: Gene_Symbol, Uniprot, Uniprot_Acc")
for _, row in df.head(5).iterrows():
    print(f"    {row['Gene_Symbol']:15s} | Uniprot: {row['Uniprot']:20s} | Acc: {row['Uniprot_Acc']}")

# Count unique mappings
df_full = pd.read_csv(DATA / "protein_quant_current_normalized.csv.gz",
                      usecols=['Gene_Symbol', 'Uniprot_Acc'])
mapping = df_full.dropna(subset=['Gene_Symbol', 'Uniprot_Acc']).drop_duplicates('Gene_Symbol')
print(f"\n  Total Gene→UniProt_Acc mappings: {len(mapping)}")
# Check how many of the 2296 missing genes we can resolve
missing = set(Path("/tmp/missing_uniprot.txt").read_text().split())
resolved = set(mapping['Gene_Symbol']) & missing
print(f"  Missing genes resolvable: {len(resolved)}/{len(missing)}")
