#!/usr/bin/env python3
"""Run weight calibration on protein scoring results."""
import json
import sys
sys.path.insert(0, '.')

from alin.protein_scoring import (
    ProteinDruggabilityScore,
    StructuralDruggability,
    ProteinAbundance,
    DegradabilityScore,
    PPIAccessibility,
    RNAExpression,
    RNAProteinConcordance,
    calibrate_layer_weights,
)

# Load cached results and reconstruct dataclass objects
with open('results/protein_druggability_scores.json') as f:
    raw = json.load(f)

results = {}
for gene, d in raw.items():
    struct = StructuralDruggability(**d['structural'])
    degrad = DegradabilityScore(**d['degradability'])
    ppi = PPIAccessibility(**d['ppi'])
    abundance = ProteinAbundance(**d['abundance']) if 'abundance' in d else None
    rna = RNAExpression(**d['rna_expression']) if 'rna_expression' in d else None
    conc = RNAProteinConcordance(**d['rna_protein_concordance']) if 'rna_protein_concordance' in d else None
    results[gene] = ProteinDruggabilityScore(
        gene=gene,
        structural=struct,
        abundance=abundance,
        degradability=degrad,
        ppi=ppi,
        rna_expression=rna,
        rna_protein_concordance=conc,
        protein_score=d['protein_score'],
        blended_score=d['blended_score'],
        gene_druggability=d['gene_druggability'],
    )

# Run calibration
cal = calibrate_layer_weights(results, n_bootstrap=500)

print("=" * 60)
print("WEIGHT CALIBRATION RESULTS")
print("=" * 60)

if 'error' in cal:
    print(f"\nCalibration error: {cal['error']}")
    sys.exit(1)

print(f"\nNominal weights: {cal['nominal_weights']}")
print(f"Optimized weights: {cal['optimized_weights']}")
print(f"\nNominal AUC: {cal['nominal_auc']}")
print(f"Optimized AUC: {cal['optimized_auc']}")
print(f"AUC improvement: {cal['optimized_auc'] - cal['nominal_auc']:+.4f}")
print(f"\nPositive genes (gene_d >= 0.8): {cal['n_positive_genes']}")
print(f"Negative genes (gene_d <= 0.3): {cal['n_negative_genes']}")

print(f"\nRank stability (500 bootstrap, ±30% perturbation):")
print(f"  Mean Spearman ρ: {cal['rank_stability_mean_rho']}")
print(f"  Std Spearman ρ: {cal['rank_stability_std_rho']}")

print(f"\nPer-layer sensitivity (∂AUC / ∂10%Δw):")
for layer, sens in sorted(cal['weight_sensitivity'].items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {layer:20s}: {sens:+.4f}")

# Save results
with open('results/weight_calibration.json', 'w') as f:
    json.dump(cal, f, indent=2)
print(f"\nSaved to results/weight_calibration.json")
