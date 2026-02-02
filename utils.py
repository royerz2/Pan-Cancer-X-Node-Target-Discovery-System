#!/usr/bin/env python3
"""
Shared utilities for ALIN Framework (Adaptive Lethal Intersection Network).
Keeps common logic DRY across modules.
"""

import re
from pathlib import Path
from typing import Optional, Set

import pandas as pd


def sanitize_cancer_name(name: str, max_len: Optional[int] = None) -> str:
    """
    Sanitize cancer type string for safe use in filenames.
    Replaces non-word chars (except hyphen) with underscore.
    """
    safe = re.sub(r'[^\w\-]', '_', name)
    if max_len is not None:
        safe = safe[:max_len]
    return safe


def load_depmap_crispr_subset(base: Path, genes: Set[str], fallback_genes: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Load DepMap CRISPRGeneEffect.csv subset for given genes.
    Columns are "GENE (entrez_id)" format; returns DataFrame with gene symbols as column names.
    """
    crispr_path = base / "depmap_data" / "CRISPRGeneEffect.csv"
    crispr_cols = pd.read_csv(crispr_path, nrows=0).columns.tolist()
    gene_cols = [c for c in crispr_cols if any(c.startswith(g + ' ') for g in genes)]
    if not gene_cols and fallback_genes:
        gene_cols = [c for c in crispr_cols if any(g in c for g in fallback_genes)]
    load_cols = [crispr_cols[0]] + gene_cols[:15]
    crispr = pd.read_csv(crispr_path, usecols=load_cols, index_col=0)
    col_map = {c: c.split()[0] for c in crispr.columns if ' ' in c}
    return crispr.rename(columns=col_map)
