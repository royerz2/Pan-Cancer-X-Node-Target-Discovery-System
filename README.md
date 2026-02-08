# ALIN Framework

**Adaptive Lethal Intersection Network**

A computational pipeline for discovering optimal triple drug combinations across cancer types using systems biology and minimal hitting set optimization.

**Reference:** Methodology extrapolated from Liaki et al. (2025). A targeted combination therapy achieves effective pancreatic cancer regression and prevents tumor resistance. bioRxiv doi: 10.1101/2025.08.04.668325.

---

## Introduction

### Background and Motivation

Cancer drug resistance remains a major obstacle to durable therapeutic responses. Single-agent therapies often fail due to tumor heterogeneity, adaptive bypass mechanisms, and pre-existing resistant clones. **Combination therapy** addresses this by simultaneously targeting multiple nodes in tumor viability networks, reducing the probability of resistance emergence and improving outcomes.

However, identifying *optimal* combination targets is challenging: the combinatorial space is vast, and empirical screening is costly. Rational design requires integrating (1) tumor-specific dependencies, (2) network topology, (3) known synergy/resistance mechanisms, and (4) druggability.

### X-Node Concept and Theoretical Foundation

The **X-node** term (coined here) formalizes combination target discovery as a **minimal hitting set problem** over tumor viability networks:

- **Viability paths** = functional pathways that support tumor survival (e.g., essential gene modules, signaling cascades).
- **X-nodes** = minimal sets of targets that "hit" (intersect) every viability path.
- **Rationale:** Hitting all paths maximizes tumor kill; minimizing the number of nodes reduces toxicity and side effects.

The approach is extrapolated from Liaki et al. (bioRxiv doi: 10.1101/2025.08.04.668325), who demonstrated that targeting RAF1 + EGFR + STAT3 (downstream, upstream, and orthogonal KRAS signaling) achieved effective pancreatic cancer regression and prevented resistance in preclinical models. This framework **generalizes that methodology to all cancer types** in DepMap, enabling pan-cancer discovery.

### Pan-Cancer Generalization

We extend the PDAC-specific approach from the paper to:

1. **All DepMap cancer types** — Cancer type mapping via OncoTree (OncotreePrimaryDisease).
2. **Triple combinations** — Systems biology scoring for synergy, resistance, and pathway coverage.
3. **Multi-source validation** — PubMed, STRING, ClinicalTrials.gov, PRISM drug sensitivity.
4. **Benchmarking** — Comparison against FDA-approved and clinically validated combinations.

### Key Contributions

- **Integrated pipeline** — End-to-end from DepMap + OmniPath to ranked triple combinations.
- **Reproducible** — Pinned dependencies, data availability documentation, full pipeline script.
- **Validated** — 61% recall vs. gold standard (vs. 21% random, 39% top-genes baseline).
- **Novel discovery** — 5 combinations with no existing clinical trials.

---

## Methods

### 1. Data Sources

| Data | Source | Role |
|------|--------|------|
| **DepMap** | [depmap.org](https://depmap.org) | CRISPR gene dependency (Chronos), cell line metadata (Model.csv), cancer type (OncotreePrimaryDisease) |
| **OmniPath** | Built-in / API | Cancer signaling network (MAPK, PI3K/AKT, JAK/STAT, SRC, cell cycle, apoptosis, etc.) |
| **PRISM** | DepMap Repurposing | Drug sensitivity for validation (primary + secondary screens) |
| **GDSC** | Sanger Institute | Alternative drug sensitivity (optional) |

### 2. Cancer Type Mapping

Cancer types are normalized via **OncoTree** (OncotreePrimaryDisease, OncotreeCode). Common aliases (e.g., PAAD, PDAC → Pancreatic Adenocarcinoma; NSCLC → Non-Small Cell Lung Cancer) are supported. Cell lines are filtered by cancer type for cancer-specific analysis.

### 3. Viability Path Inference

We infer **viability paths** — sets of genes that collectively support tumor survival — using three methods:

1. **Co-essentiality clustering (refined):** Genes essential together across cell lines are clustered into pathway-like modules. Uses hierarchical clustering on co-occurrence (Jaccard-like) matrix. Only genes essential in &gt;30% of cancer cell lines (selectivity filter). Optional expression filter: if CCLE expression data is available, only count essential if expressed in tumor (TPM &gt; threshold).
2. **Consensus essential modules:** Genes consistently essential across cell lines of a cancer type.
3. **Signaling pathway dependencies:** NetworkX `all_simple_paths()` with length limits (2–4 hops). Paths scored by mean dependency in cancer; low-confidence paths (confidence &lt; 0.5) pruned.

Pan-essential genes are filtered to focus on **cancer-specific** dependencies.

### 4. Minimal Hitting Set Optimization

Given viability paths *P*, we find minimal-cost sets *T* such that every path in *P* intersects *T*.

**Cost function** (per gene):

- **Toxicity** — DrugTargetDB (clinical data) + optional OpenTargets API (off-target safety liabilities) + tissue expression weight (GCN portal placeholder) + FDA MedWatch ADRs (placeholder).
- **Tumor specificity** — Reward for stronger dependency in cancer vs. pan-cancer.
- **Druggability** — Reward for approved/clinical-stage drugs.
- **Pan-essential penalty** — Strong penalty if gene is pan-essential.
- **Base penalty** — Per-node cost (fewer nodes preferred).

**Solver:** Greedy (coverage/cost ratio) + exhaustive enumeration for small gene sets (≤25 genes). Solutions are ranked by cardinality and total cost.

### 5. Triple Combination Scoring (Systems Biology)

From hitting set candidates, we enumerate and score **triple combinations** using:

- **Path coverage** — Fraction of viability paths hit (min 0.5–0.7).
- **Total cost** — Sum of gene costs.
- **Synergy score** — Pathway complementarity (hitting independent pathways) + known clinical synergies (e.g., BRAF+MEK, EGFR+MET, SRC+FYN+STAT3).
- **Resistance probability** — Estimated from uncovered bypass mechanisms (e.g., EGFR→MET, BRAF→PIK3CA). Lower is better.
- **Druggability** — Count of targets with approved/clinical drugs.

**Combined score** (lower = better):

```
combined = 0.3×cost + 0.25×(1−synergy) + 0.25×resistance + 0.2×(1−coverage) − 0.15×druggable_count
```

Top triples are ranked by combined score.

### 6. Validation Pipeline

Predicted combinations are validated against:

1. **PubMed** — Literature co-mention of targets + cancer (cached API).
2. **STRING** — Protein–protein interaction and functional enrichment.
3. **ClinicalTrials.gov** — Matching trials by drug names and cancer type.
4. **Drug sensitivity (PRISM/GDSC)** — Gene–drug correlation, Bliss independence for combination effect.

### 7. Patient Stratification

For each combination, we identify **patient subgroups** most likely to benefit:

- **Mutation-based** — KRAS G12C, BRAF V600E, EGFR L858R, etc.
- **Expression biomarkers** — High/low expression thresholds.
- **Companion diagnostic** — Recommended genes for patient selection.

### 8. Benchmarking

We compare predictions against a **gold standard** of 23 FDA-approved and clinically validated combinations (e.g., BRAF+MEK in melanoma, EGFR+MET in NSCLC, CDK4/6 in breast cancer).

- **Recall** — Do our triples contain the known target set (exact or superset)?
- **Gene equivalence** — MAP2K1/MAP2K2 (MEK), CDK4/CDK6 treated as equivalent.
- **Baselines** — Random triple sampling; top-genes (most frequent in DepMap) baseline.

---

## Key Features

- **Pan-cancer analysis** — DepMap CRISPR + OncoTree cancer type mapping
- **Triple combinations** — Network topology (X-nodes), synergy scoring, resistance prediction
- **Multi-source validation** — PubMed, STRING, ClinicalTrials.gov, PRISM drug sensitivity
- **Patient stratification** — Mutation-based subgroups, companion diagnostics
- **Benchmarking** — 61% recall vs. FDA-approved/clinical gold standard
- **Novel discovery** — 5 combinations with no existing clinical trials

---

## Installation

```bash
git clone <repository>
cd "ALIN Framework"
pip install -r requirements.txt
```

---

## Data Requirements

| Data | Source | Required |
|------|--------|----------|
| DepMap | [depmap.org](https://depmap.org) | Yes — Model.csv, CRISPRGeneEffect.csv, SubtypeMatrix.csv |
| PRISM | [depmap.org/repurposing](https://depmap.org/repurposing) | Optional — primary-screen-replicate-collapsed-*.csv, secondary-screen-dose-response-curve-parameters.csv |
| OmniPath | Built-in / API | Built-in cancer signaling network |

Place DepMap files in `depmap_data/`. PRISM in `drug_sensitivity_data/`. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for URLs and licenses.

---

## Quick Start

```bash
# Single cancer (e.g., pancreatic)
python pan_cancer_xnode.py --cancer-type "Pancreatic Adenocarcinoma" --output results/

# Full pan-cancer with triple combinations
python pan_cancer_xnode.py --all-cancers --triples --output results/

# With validation
python pan_cancer_xnode.py --all-cancers --triples --validate --output results/

# Validate existing results (skip re-discovery)
python pan_cancer_xnode.py --validate-only results/

# Benchmark
python benchmarking_module.py --triples results/triple_combinations.csv --output benchmark_results/

# Benchmark with baselines (random, top-genes)
python benchmarking_module.py --triples results/triple_combinations.csv --baselines --n-trials 30 --output benchmark_results/

# Gold standard pipeline (benchmark cancer types only)
python run_pipeline.py

# Full reproducibility pipeline
bash run_full_pipeline.sh
```

---

## Pipeline Overview

```
DepMap (CRISPR, Model, Subtype)
        ↓
Cancer type mapping (OncoTree)
        ↓
Viability path inference (essential genes, signaling paths)
        ↓
Minimal hitting set + Systems biology (X-nodes, synergy, resistance)
        ↓
Triple combination ranking
        ↓
Validation (PubMed, STRING, ClinicalTrials) + Drug sensitivity (PRISM)
        ↓
Patient stratification + Lab protocols
```

---

## Output Structure

```
results/
├── triple_combinations.csv      # All discovered triples
├── triple_target_frequency.csv # Target frequency across cancers
├── pan_cancer_summary.csv       # Per-cancer summary
├── all_findings.json            # Full export
└── *_report.txt                 # Per-cancer reports

benchmark_results/
├── benchmark_results.csv
├── benchmark_metrics.json
└── benchmark_report.txt

priority_pipeline_results/
├── api_validation.csv
├── clinical_trials/
├── stratification/
├── drug_sensitivity/
└── priority_combined_summary.csv
```

---

## Benchmark Results

- **Recall:** 61% (14/23 gold standard combinations recovered)
- **vs Random baseline:** 21% ± 7%
- **vs Top-genes baseline:** 39%
- **Match type:** Superset (our triples contain known pairs)
- **Mean rank when matched:** 1.0 (top prediction)
- **Gold standard:** FDA-approved + clinical trial combinations

---

## Module Overview

| Module | Role |
|--------|------|
| `pan_cancer_xnode.py` | Main discovery engine (DepMap, OmniPath, hitting set, triple finder) |
| `alin/validation.py` | Built-in validation (literature, PPI, drug synergy) |
| `alin/api_validators.py` | PubMed + STRING API validation with caching |
| `alin/drug_sensitivity.py` | PRISM/GDSC drug sensitivity, gene–drug correlation |
| `alin/clinical_trials.py` | ClinicalTrials.gov search |
| `alin/patient_stratification.py` | Patient subgroups, biomarkers, companion diagnostics |
| `alin/toxicity.py` | OpenTargets toxicity, tissue expression (cost function) |
| `alin/utils.py` | Shared utilities (sanitize_cancer_name, load_depmap_crispr_subset) |
| `benchmarking_module.py` | Gold standard comparison, random/top-genes baselines |
| `run_pipeline.py` | Run pipeline for gold standard cancer types |
| `gold_standard.py` | Clinical gold standard entries + benchmark functions |

---

## Citation

If you use this pipeline, please cite:

- **ALIN Framework (bioRxiv preprint):** Author AN, Author BT. 2025. ALIN Framework (Adaptive Lethal Intersection Network): A Systems Biology Pipeline for Triple Drug Combination Prediction. bioRxiv doi: 10.1101/YYYY.MM.DD.XXXXXXX. *(Replace with actual authors and DOI after submission.)*
- PDAC combination therapy (source of extrapolated approach): Liaki V, Barrambana S, et al. 2025. A targeted combination therapy achieves effective pancreatic cancer regression and prevents tumor resistance. bioRxiv doi: [10.1101/2025.08.04.668325](https://doi.org/10.1101/2025.08.04.668325)
- DepMap: [depmap.org](https://depmap.org)
- OmniPath: [omnipathdb.org](https://omnipathdb.org)

---

## License

See LICENSE file.
