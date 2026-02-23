#!/usr/bin/env python3
"""
Run the full 6-layer multi-omics druggability scoring pipeline
on all 52 ALIN signaling genes for Pancreatic Adenocarcinoma.
"""

import logging
import sys
sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from alin.protein_scoring import (
    GENE_TO_UNIPROT,
    ProteinDruggabilityScorer,
    generate_protein_scoring_report,
)

# Use canonical druggability from ALIN constants
try:
    from alin.constants import get_druggability_score
except (ImportError, AttributeError):
    print("WARNING: get_druggability_score not found, using flat 0.5 default")
    get_druggability_score = lambda g: 0.5


def main():
    genes = list(GENE_TO_UNIPROT.keys())
    cancer_type = "Pancreatic Adenocarcinoma"

    print(f"Scoring {len(genes)} genes for: {cancer_type}")
    print(f"Genes: {', '.join(genes[:10])}... (+{len(genes)-10} more)")
    print()

    scorer = ProteinDruggabilityScorer(
        genes=genes,
        cancer_type=cancer_type,
        gene_druggability_fn=get_druggability_score,
        cache_dir="./api_cache/protein",
        proteomics_dir="./depmap_data",
        alpha=0.6,
    )

    results = scorer.score_all(progress=True)

    # Generate reports
    csv_path = generate_protein_scoring_report(results, output_dir="./results")

    # Print per-gene detail for top 20
    print(f"\n{'='*90}")
    print(f"DETAILED RESULTS (top 20 by blended score)")
    print(f"{'='*90}")
    sorted_results = sorted(results.values(), key=lambda r: r.blended_score, reverse=True)

    header = (f"{'Gene':<10} {'d_gene':>7} {'p(g)':>7} {'d_new':>7} "
              f"{'Struct':>7} {'Abund':>7} {'Degrad':>7} {'PPI':>7} "
              f"{'RNA_ex':>7} {'Concrd':>7}")
    print(header)
    print('-' * len(header))

    for r in sorted_results[:20]:
        abund = f"{r.abundance.abundance_score:.3f}" if r.abundance else "   N/A"
        rna = f"{r.rna_expression.expression_score:.3f}" if r.rna_expression else "   N/A"
        conc = f"{r.rna_protein_concordance.concordance_score:.3f}" if r.rna_protein_concordance else "   N/A"
        print(
            f"{r.gene:<10} {r.gene_druggability:>7.3f} {r.protein_score:>7.3f} "
            f"{r.blended_score:>7.3f}  {r.structural.structural_score:>6.3f} "
            f"{abund:>7} {r.degradability.degradability_score:>7.3f} "
            f"{r.ppi.ppi_score:>7.3f} {rna:>7} {conc:>7}"
        )

    # Concordance statistics
    concordant = [r for r in results.values() if r.rna_protein_concordance and r.rna_protein_concordance.concordance_tier != 'insufficient']
    if concordant:
        print(f"\n--- RNA/Protein Concordance ---")
        for r in sorted(concordant, key=lambda x: x.rna_protein_concordance.spearman_rho, reverse=True):
            c = r.rna_protein_concordance
            print(f"  {r.gene:<10} ρ={c.spearman_rho:>6.3f}  p={c.spearman_pvalue:.2e}  tier={c.concordance_tier}")


if __name__ == "__main__":
    main()
