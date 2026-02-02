# Manuscript Figure Specification for bioRxiv Preprint

Recommended visualisations for the ALIN Framework manuscript, with bioRxiv attribution format.

---

## bioRxiv Citation (for manuscript and README)

When the preprint is published on bioRxiv, cite it as:

```
Author AN, Author BT. 2025. ALIN Framework (Adaptive Lethal Intersection Network): 
A Systems Biology Pipeline for Triple Drug Combination Prediction. bioRxiv 
doi: 10.1101/YYYY.MM.DD.XXXXXXX.
```

*Replace Author AN, Author BT with actual author names; replace YYYY.MM.DD.XXXXXXX with the assigned bioRxiv DOI after submission.*

---

## Main Text Figures

### Figure 1. Pipeline overview (schematic)

**Purpose:** Orient readers to the workflow before methods.

**Content:** Flowchart showing:
1. DepMap (CRISPR, Model, Subtype) → Cancer type mapping (OncoTree)
2. Viability path inference (co-essentiality, signaling paths)
3. Minimal hitting set + systems biology (X-nodes, synergy, resistance)
4. Triple combination ranking
5. Validation (PubMed, STRING, ClinicalTrials.gov, PRISM) + patient stratification

**Format:** Horizontal flowchart, boxes and arrows. Single column (≈85 mm) or full width (≈178 mm).

**Source:** `generate_figures.py` → `figures/fig1_pipeline_schematic.png`

---

### Figure 2. X-node / minimal hitting set concept

**Purpose:** Explain the core theoretical concept.

**Content:** Diagram showing:
- 3–4 viability paths (e.g., essential modules, signaling cascades) as overlapping sets or network paths
- X-nodes as minimal targets that intersect all paths
- Example: RAF1 + EGFR + STAT3 hitting PDAC viability paths (from Liaki et al., bioRxiv doi: 10.1101/2025.08.04.668325)

**Format:** Venn-style or network diagram. Single column.

**Source:** `generate_figures.py` → `figures/fig2_xnode_concept.png`

---

### Figure 3. Benchmark: recovery of known combinations

**Purpose:** Validate the approach against gold standard.

**Content:** Bar chart comparing recall (%):
- ALIN (ours): 61%
- Random baseline: 21% ± 7%
- Top-genes baseline: 39%

**Format:** Vertical bar chart, error bars for random. Single column.

**Source:** `generate_figures.py` → `figures/benchmark_comparison.png`

**Suggested caption:** *ALIN outperforms random and top-genes baselines in recovering FDA-approved and clinically validated drug combinations (n=23 gold standard).*

---

### Figure 4. Pan-cancer discovery overview

**Purpose:** Show breadth of discovery across cancer types.

**Content:** Heatmap or grouped bar chart:
- Rows: cancer types (or top 15–20)
- Columns: top triple patterns (e.g., KRAS+CDK6+STAT3, BRAF+MAP2K1+STAT3)
- Values: presence/absence or rank

**Alternative:** Bar chart of number of cancer types per triple pattern (current `triple_patterns.png`).

**Format:** Full width or single column.

**Source:** `generate_figures.py` → `figures/triple_patterns.png`; extend with heatmap if desired.

---

### Figure 5. Most frequently predicted targets

**Purpose:** Highlight recurrent targets across cancers.

**Content:** Horizontal bar chart of top 10–12 targets by frequency in top triple combinations (KRAS, STAT3, CDK6, CDK4, BRAF, etc.).

**Format:** Single column.

**Source:** `generate_figures.py` → `figures/target_frequency.png`

---

### Figure 6. Therapeutic potential landscape (synergy vs resistance)

**Purpose:** Show distribution of predicted triples in synergy–resistance space.

**Content:** Scatter plot:
- X: Synergy score
- Y: Resistance score (lower = better)
- Quadrants: high synergy + low resistance = preferred
- Optional: highlight novel combinations (no clinical trials)

**Format:** Single column.

**Source:** `generate_figures.py` → `figures/synergy_resistance.png`

---

### Figure 7. Case study: Pancreatic adenocarcinoma (optional)

**Purpose:** Deep dive on one cancer type.

**Content:**
- Top viability paths (co-essentiality clusters, signaling paths)
- Top triple (e.g., KRAS+CDK6+STAT3) and how it hits paths
- Drug mapping, synergy/resistance scores
- Validation summary (PubMed, STRING, drug sensitivity)

**Format:** Multi-panel (2×2 or 3-panel). Single or double column.

**Source:** Combine outputs from `results_test/`, `priority_pipeline_results/`, and manual network diagram.

---

## Supplementary Figures

### Figure S1. Detailed pipeline flowchart

**Content:** Full pipeline with module names, file I/O, and optional steps.

---

### Figure S2. Co-essentiality clustering schematic

**Content:** Illustration of genes essential together → hierarchical clustering → pathway-like modules.

---

### Figure S3. Network path inference example

**Content:** Driver → intermediate → effector paths in OmniPath network; path scoring by dependency.

---

### Figure S4. Benchmark: rank distribution

**Content:** When ALIN matches a gold-standard combination, where does it rank? (Mean rank = 1.0.)

---

### Figure S5. Novel combinations (no clinical trials)

**Content:** Table or small figure listing the 5 novel combinations with cancer type, targets, drugs, and scores.

---

## Figure Generation Checklist

| Figure | Generated by code? | File |
|--------|--------------------|------|
| Fig 1 (Pipeline) | Yes | `figures/fig1_pipeline_schematic.png` |
| Fig 2 (X-node concept) | Yes | `figures/fig2_xnode_concept.png` |
| Fig 3 (Benchmark) | Yes | `figures/fig3_benchmark_comparison.png` |
| Fig 4 (Pan-cancer) | Yes | `figures/fig4_triple_patterns.png` |
| Fig 5 (Target frequency) | Yes | `figures/fig5_target_frequency.png` |
| Fig 6 (Synergy vs resistance) | Yes | `figures/fig6_synergy_resistance.png` |
| Fig 7 (Case study) | Partial | Combine outputs |
| Fig S1 (Detailed pipeline) | Yes | `figures/figS1_detailed_pipeline.png` |
| Fig S2 (Co-essentiality) | Yes | `figures/figS2_coessentiality.png` |
| Fig S3 (Network paths) | Yes | `figures/figS3_network_paths.png` |
| Fig S4 (Benchmark rank) | Yes | `figures/figS4_benchmark_rank.png` |
| Fig S5 (Novel combinations) | Yes | `figures/figS5_novel_combinations.png` |

Run `python generate_figures.py` to regenerate all figures.

---

## Recommended Figure Order for Manuscript

1. **Fig 1** — Pipeline overview (Introduction/Methods)
2. **Fig 2** — X-node concept (Methods)
3. **Fig 3** — Benchmark (Results)
4. **Fig 4** — Pan-cancer discovery (Results)
5. **Fig 5** — Target frequency (Results)
6. **Fig 6** — Synergy vs resistance (Results)
7. **Fig 7** — Case study (Results, optional)

---

## bioRxiv Attribution (for README / GitHub)

Add to README Citation section after preprint is live:

```
If you use this pipeline, please cite:

Author AN, Author BT. 2025. ALIN Framework (Adaptive Lethal Intersection Network): 
A Systems Biology Pipeline for Triple Drug Combination Prediction. bioRxiv 
doi: 10.1101/YYYY.MM.DD.XXXXXXX.

PDAC combination therapy (source of extrapolated approach): Liaki V, Barrambana S, et al. 2025. A targeted combination therapy achieves effective pancreatic cancer regression and prevents tumor resistance. bioRxiv doi: 10.1101/2025.08.04.668325
DepMap: depmap.org
OmniPath: omnipathdb.org
```
