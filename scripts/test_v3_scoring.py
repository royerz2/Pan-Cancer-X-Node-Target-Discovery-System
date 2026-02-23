#!/usr/bin/env python3
"""Quick sanity check: score one cancer type with v3 and compare to old fallback."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alin.protein_scoring import ProteinDruggabilityScorer, GENE_TO_UNIPROT
import numpy as np

# Use a mix of genes: some with static UniProt, some without
test_genes = [
    'EGFR', 'STAT3', 'CDK4', 'KRAS',  # Have static UniProt
    'SOX9', 'MEAF6', 'DET1', 'MAEA',   # Previously missing, now from proteomics
    'TP53', 'BCL2',                      # Have static UniProt
]

print(f"Testing v3 scoring on {len(test_genes)} genes...")
scorer = ProteinDruggabilityScorer(
    genes=test_genes,
    cancer_type='Pancreatic Adenocarcinoma',
    gene_druggability_fn=lambda g: 0.5,
)

results = scorer.score_all()

print(f"\n{'Gene':<10} {'UniProt?':<10} {'Struct':>7} {'Abund':>7} {'Degrad':>7} "
      f"{'PPI':>7} {'RNA':>7} {'Conc':>7} {'p(g)':>7} {'blend':>7}")
print("-" * 90)

for gene in test_genes:
    r = results[gene]
    has_up = 'static' if gene in GENE_TO_UNIPROT else ('prot' if r.structural.uniprot_id else 'none')
    abund = f"{r.abundance.abundance_score:.3f}" if r.abundance else "  ---"
    rna = f"{r.rna_expression.expression_score:.3f}" if r.rna_expression else "  ---"
    conc = f"{r.rna_protein_concordance.concordance_score:.3f}" if r.rna_protein_concordance else "  ---"
    print(f"{gene:<10} {has_up:<10} {r.structural.structural_score:7.3f} {abund:>7} "
          f"{r.degradability.degradability_score:7.3f} {r.ppi.ppi_score:7.3f} "
          f"{rna:>7} {conc:>7} {r.protein_score:7.3f} {r.blended_score:7.3f}")

# Summary stats
all_flat = sum(1 for r in results.values() if r.protein_score == 0.3)
print(f"\nFlat-0.3 scores: {all_flat}/{len(results)} ({100*all_flat/len(results):.0f}%)")
print(f"Score range: [{min(r.protein_score for r in results.values()):.3f}, "
      f"{max(r.protein_score for r in results.values()):.3f}]")
