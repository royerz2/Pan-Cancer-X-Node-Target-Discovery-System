#!/usr/bin/env python3
"""Deep analysis of protein scoring results for discussion."""
import json
import statistics

with open('results/protein_druggability_scores.json') as f:
    data = json.load(f)

# === PDAC Triple Detail ===
print("=" * 70)
print("DETAILED PROTEIN-LEVEL PROFILES: FYN + KRAS + STAT3")
print("=" * 70)
for g in ['FYN', 'KRAS', 'STAT3']:
    d = data[g]
    print(f"\n--- {g} ---")
    print(f"  Protein score: {d['protein_score']:.4f}")
    print(f"  Gene druggability: {d['gene_druggability']}")
    print(f"  Blended score: {d['blended_score']:.4f}")
    s = d['structural']
    print(f"  Structural: {s['structural_score']:.3f} (PDB={s['n_pdb_structures']}, ligand-bound={s['n_ligand_bound']}, pocket={s['has_pocket']})")
    deg = d['degradability']
    print(f"  Degradability: {deg['degradability_score']:.1f} (degrader={deg['has_known_degrader']}, exemplar={deg['degrader_exemplar']})")
    p = d['ppi']
    print(f"  PPI: {p['ppi_score']:.1f} (complexes={p['n_pdb_complexes']}, interface={p['has_interface_data']})")
    a = d['abundance']
    print(f"  Abundance: {a['abundance_score']:.1f} (detected={a['n_detected']}/{a['n_cell_lines']}, mean={a['mean_abundance']:.4f})")
    r = d['rna_expression']
    print(f"  RNA: {r['expression_score']:.1f} (expressed={r['n_expressed']}/{r['n_cell_lines']}, mean_TPM={r['mean_expression']:.2f})")
    c = d['rna_protein_concordance']
    print(f"  Concordance: {c['concordance_score']:.1f} (rho={c['spearman_rho']:.3f}, tier={c['concordance_tier']}, n={c['n_matched_lines']})")

# === Global statistics ===
print("\n" + "=" * 70)
print("GLOBAL STATISTICS ACROSS ALL 79 GENES")
print("=" * 70)

genes = list(data.keys())
protein_scores = [data[g]['protein_score'] for g in genes]
blended_scores = [data[g]['blended_score'] for g in genes]
gene_drug = [data[g]['gene_druggability'] for g in genes]

print(f"\nTotal genes scored: {len(genes)}")
print(f"Protein score: mean={statistics.mean(protein_scores):.3f}, median={statistics.median(protein_scores):.3f}, std={statistics.stdev(protein_scores):.3f}")
print(f"  min={min(protein_scores):.3f} ({genes[protein_scores.index(min(protein_scores))]})")
print(f"  max={max(protein_scores):.3f} ({genes[protein_scores.index(max(protein_scores))]})")
print(f"Blended score: mean={statistics.mean(blended_scores):.3f}, median={statistics.median(blended_scores):.3f}, std={statistics.stdev(blended_scores):.3f}")
print(f"Gene druggability: mean={statistics.mean(gene_drug):.3f}")

# === Layer-by-layer breakdown ===
print("\n" + "=" * 70)
print("LAYER-BY-LAYER BREAKDOWN")
print("=" * 70)

# Structural
struct_scores = [data[g]['structural']['structural_score'] for g in genes]
n_pocket = sum(1 for g in genes if data[g]['structural']['has_pocket'])
n_pdb_any = sum(1 for g in genes if data[g]['structural']['n_pdb_structures'] > 0)
n_ligand = sum(1 for g in genes if data[g]['structural']['n_ligand_bound'] > 0)
print(f"\nStructural (weight=0.25):")
print(f"  Mean score: {statistics.mean(struct_scores):.3f}")
print(f"  Genes with PDB structures: {n_pdb_any}/{len(genes)} ({100*n_pdb_any/len(genes):.0f}%)")
print(f"  Genes with ligand-bound structures: {n_ligand}/{len(genes)} ({100*n_ligand/len(genes):.0f}%)")
print(f"  Genes with druggable pocket: {n_pocket}/{len(genes)} ({100*n_pocket/len(genes):.0f}%)")

# Degradability
deg_scores = [data[g]['degradability']['degradability_score'] for g in genes]
n_degrader = sum(1 for g in genes if data[g]['degradability']['has_known_degrader'])
print(f"\nDegradability (weight=0.15):")
print(f"  Mean score: {statistics.mean(deg_scores):.3f}")
print(f"  Genes with known PROTAC degraders: {n_degrader}/{len(genes)} ({100*n_degrader/len(genes):.0f}%)")
degrader_genes = sorted([g for g in genes if data[g]['degradability']['has_known_degrader']])
print(f"  Degradable targets: {', '.join(degrader_genes)}")

# PPI
ppi_scores = [data[g]['ppi']['ppi_score'] for g in genes]
n_ppi = sum(1 for g in genes if data[g]['ppi']['n_pdb_complexes'] > 0)
print(f"\nPPI Surface (weight=0.15):")
print(f"  Mean score: {statistics.mean(ppi_scores):.3f}")
print(f"  Genes with PDB complexes: {n_ppi}/{len(genes)} ({100*n_ppi/len(genes):.0f}%)")

# Abundance
abd_scores = [data[g]['abundance']['abundance_score'] for g in genes]
n_detected = sum(1 for g in genes if data[g]['abundance']['n_detected'] > 0)
det_fracs = [data[g]['abundance']['n_detected']/max(data[g]['abundance']['n_cell_lines'],1) for g in genes if data[g]['abundance']['n_cell_lines'] > 0]
print(f"\nProtein Abundance - Gygi CCLE (weight=0.20):")
print(f"  Mean score: {statistics.mean(abd_scores):.3f}")
print(f"  Genes detected in >=1 PDAC line: {n_detected}/{len(genes)} ({100*n_detected/len(genes):.0f}%)")
print(f"  Mean detection fraction: {statistics.mean(det_fracs):.3f}")

# RNA
rna_scores = [data[g]['rna_expression']['expression_score'] for g in genes]
n_rna = sum(1 for g in genes if data[g]['rna_expression']['n_expressed'] > 0)
print(f"\nRNA Expression - DepMap (weight=0.15):")
print(f"  Mean score: {statistics.mean(rna_scores):.3f}")
print(f"  Genes expressed in >=1 PDAC line: {n_rna}/{len(genes)} ({100*n_rna/len(genes):.0f}%)")

# Concordance
# Concordance
conc_scores = [data[g]['rna_protein_concordance']['concordance_score'] for g in genes if 'rna_protein_concordance' in data[g]]
rhos = [data[g]['rna_protein_concordance']['spearman_rho'] for g in genes if 'rna_protein_concordance' in data[g] and data[g]['rna_protein_concordance']['n_matched_lines'] > 10]
tiers = [data[g]['rna_protein_concordance']['concordance_tier'] for g in genes if 'rna_protein_concordance' in data[g]]
print(f"\nRNA-Protein Concordance (weight=0.10):")
print(f"  Mean score: {statistics.mean(conc_scores):.3f}")
if rhos:
    print(f"  Mean Spearman rho (n>10 pairs): {statistics.mean(rhos):.3f}")
print(f"  Tier distribution: high={tiers.count('high')}, moderate={tiers.count('moderate')}, low={tiers.count('low')}, insufficient={tiers.count('insufficient')}")

# === Discordance analysis ===
print("\n" + "=" * 70)
print("DISCORDANCE ANALYSIS: Gene vs Protein Druggability")
print("=" * 70)
print("\nGenes where protein score DISAGREES with gene-level score:")
print("(large gap = interesting biology)")
for g in sorted(genes, key=lambda x: abs(data[x]['protein_score'] - data[x]['gene_druggability']), reverse=True)[:15]:
    d = data[g]
    gap = d['protein_score'] - d['gene_druggability']
    direction = "protein >> gene" if gap > 0 else "gene >> protein"
    print(f"  {g:12s}: protein={d['protein_score']:.3f}  gene={d['gene_druggability']:.1f}  gap={gap:+.3f}  ({direction})")

# === Abundance problem ===
print("\n" + "=" * 70)
print("ABUNDANCE LAYER: LOW DETECTION PROBLEM")
print("=" * 70)
low_abd = [(g, data[g]['abundance']['n_detected'], data[g]['abundance']['n_cell_lines'], data[g]['abundance']['mean_abundance']) 
           for g in genes]
low_abd.sort(key=lambda x: x[1])
print("\nBottom 15 by protein detection:")
for g, nd, nl, ma in low_abd[:15]:
    print(f"  {g:12s}: detected in {nd}/{nl} lines (mean abundance={ma:.4f})")
print("\nTop 15 by protein detection:")    
for g, nd, nl, ma in sorted(low_abd, key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {g:12s}: detected in {nd}/{nl} lines (mean abundance={ma:.4f})")

# === PROTAC landscape ===
print("\n" + "=" * 70)
print("PROTAC DEGRADER LANDSCAPE")
print("=" * 70)
for g in sorted(genes):
    deg = data[g]['degradability']
    if deg['has_known_degrader']:
        print(f"  {g:12s}: {deg['degrader_exemplar']:30s} status={deg['degrader_status']}")

# === Top/bottom ranked genes ===
print("\n" + "=" * 70)
print("TOP 15 GENES BY BLENDED SCORE")
print("=" * 70)
for g in sorted(genes, key=lambda x: data[x]['blended_score'], reverse=True)[:15]:
    d = data[g]
    print(f"  {g:12s}: blended={d['blended_score']:.3f}  protein={d['protein_score']:.3f}  gene={d['gene_druggability']}")

print("\n" + "=" * 70)
print("BOTTOM 15 GENES BY BLENDED SCORE")
print("=" * 70)
for g in sorted(genes, key=lambda x: data[x]['blended_score'])[:15]:
    d = data[g]
    print(f"  {g:12s}: blended={d['blended_score']:.3f}  protein={d['protein_score']:.3f}  gene={d['gene_druggability']}")
