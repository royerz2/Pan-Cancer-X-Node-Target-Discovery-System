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

The pipeline extends the PDAC-specific approach to:

1. **All DepMap cancer types** — Cancer type mapping via OncoTree (OncotreePrimaryDisease).
2. **Triple combinations** — Systems biology scoring for synergy, resistance, and pathway coverage.
3. **Multi-source validation** — PubMed, STRING, ClinicalTrials.gov, PRISM drug sensitivity.
4. **Benchmarking** — Comparison against FDA-approved and clinically validated combinations.

### Key Contributions

- **Integrated pipeline** — End-to-end from DepMap + OmniPath to ranked triple combinations.
- **Reproducible** — Pinned dependencies, data availability documentation, full pipeline script.
- **Validated** — 44.2% any-overlap recall vs. 43-entry clinical gold standard (54.3% on 35 testable entries; p < 0.001 vs. random); DrugComb synergy validation (ZIP: Δ = +2.59, p = 0.0001).
- **Novel discovery** — Multiple combinations with no existing clinical trials.

## Repository Layout

- `pan_cancer_xnode.py` — main public discovery CLI
- `run_full_pipeline.sh` / `run_full_pipeline.ps1` — cross-platform wrappers for the latest public strategy-arm workflow
- `scripts/run_pipeline.py` — focused benchmark-cancer runner
- `scripts/pipelines/run_strategy_arm_comparison.py` — fresh actionable vs exploratory arm comparison workflow
- `scripts/pipelines/run_benchmark_viability_audit.py` — focused benchmark audit for one prediction set
- `docs/` — data availability, versioning, and release-oriented documentation
- `outputs/` — runtime home for generated comparisons, audits, and validation reports
- Top-level `*_results/` directories — preserved published analysis artifacts used for benchmarking, calibration, and manuscript support

See [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md) for the release-oriented directory map, [scripts/README.md](scripts/README.md) for the script surface, and [outputs/README.md](outputs/README.md) for the non-root report layout.

---

## Methods

### 1. Data Sources

| Data | Source | Role |
|------|--------|------|
| **DepMap** | [depmap.org](https://depmap.org) | CRISPR gene dependency (Chronos), cell line metadata (Model.csv), cancer type (OncotreePrimaryDisease) |
| **OmniPath** | Built-in / API | Cancer signaling network (MAPK, PI3K/AKT, JAK/STAT, SRC, cell cycle, apoptosis, etc.) |
| **PRISM** | DepMap Repurposing | Drug sensitivity for validation (primary + secondary screens) |
| **DrugComb** | [drugcomb.fimm.fi](https://drugcomb.fimm.fi) | Pairwise drug combination synergy (739,964 measurements, 288 cell lines, 17 tissues) |
| **GDSC** | Sanger Institute | Alternative drug sensitivity (optional) |

### 2. Cancer Type Mapping

Cancer types are normalized via **OncoTree** (OncotreePrimaryDisease, OncotreeCode). Common aliases (e.g., PAAD, PDAC → Pancreatic Adenocarcinoma; NSCLC → Non-Small Cell Lung Cancer) are supported. Cell lines are filtered by cancer type for cancer-specific analysis.

### 3. Viability Path Inference

The pipeline infers **viability paths** — sets of genes that collectively support tumor survival — using three methods:

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
5. **DrugComb synergy** — Pairwise drug combination synergy scores (ZIP, Bliss, Loewe, HSA) compared between ALIN-predicted and non-predicted target pairs in tissue-matched cell lines.

### 7. Patient Stratification

For each combination, we identify **patient subgroups** most likely to benefit:

- **Mutation-based** — KRAS G12C, BRAF V600E, EGFR L858R, etc.
- **Expression biomarkers** — High/low expression thresholds.
- **Companion diagnostic** — Recommended genes for patient selection.

### 8. Benchmarking

Predictions are compared against a **gold standard** of 43 FDA-approved and clinically validated multi-target combinations spanning 25 cancer types (e.g., BRAF+MEK in melanoma, EGFR+MET in NSCLC, CDK4/6+HER2 in breast cancer).

- **Recall** — Do our triples contain the known target set (exact or superset)?
- **Gene equivalence** — MAP2K1/MAP2K2 (MEK), CDK4/CDK6 treated as equivalent.
- **Baselines** — Random triple sampling; top-genes (most frequent in DepMap) baseline.

---

## Key Features

- **Pan-cancer analysis** — DepMap CRISPR + OncoTree cancer type mapping
- **Triple combinations** — Network topology (X-nodes), synergy scoring, resistance prediction
- **Multi-source validation** — PubMed, STRING, ClinicalTrials.gov, PRISM drug sensitivity, DrugComb synergy
- **Patient stratification** — Mutation-based subgroups, companion diagnostics
- **Benchmarking** — 44.2% any-overlap recall vs. 43-entry clinical gold standard
- **Novel discovery** — 5 combinations with no existing clinical trials

---

## Installation

```bash
git clone https://github.com/royerz2/Pan-Cancer-X-Node-Target-Discovery-System.git
cd "Pan-Cancer X-Node Target Discovery System"
pip install -r requirements.txt
```

### Windows (PowerShell)

Use the PowerShell wrapper for data setup:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_data.ps1
# Optional:
.\setup_data.ps1 -lincs
.\setup_data.ps1 -lincsFull
```

If Bash is not available, install one of:

- Git for Windows (Git Bash): https://git-scm.com/download/win
- WSL + a Linux distribution: `wsl --install`

---

## Data Requirements

| Data | Source | Required |
|------|--------|----------|
| DepMap | [depmap.org](https://depmap.org) | Yes — Model.csv, CRISPRGeneEffect.csv, SubtypeMatrix.csv |
| LINCS L1000 | [clue.io](https://clue.io/data/CMap2020) | Recommended — level5_beta_trt_*.gctx, siginfo_beta.txt, geneinfo_beta.txt |
| PRISM | [depmap.org/repurposing](https://depmap.org/repurposing) | Optional — primary-screen-replicate-collapsed-*.csv, secondary-screen-dose-response-curve-parameters.csv |
| OmniPath | Built-in / API | Built-in cancer signaling network |

Place DepMap files in `depmap_data/`, LINCS files in `lincs_data/`, and PRISM in `drug_sensitivity_data/`. LINCS is auto-indexed on first use or can be prepared explicitly with `python build_lincs_index.py`. See [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) for URLs and licenses.

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
python scripts/run_pipeline.py

# Fresh strategy-arm comparison workflow (public-safe defaults)
python scripts/pipelines/run_strategy_arm_comparison.py --skip-historical --no-api --stream-subprocess-output

# Focused benchmark viability audit for one result set
python scripts/pipelines/run_benchmark_viability_audit.py --triples results/triple_combinations.csv

# Cross-platform wrapper for the public strategy-arm workflow
bash run_full_pipeline.sh
```

On Windows PowerShell, run `./run_full_pipeline.ps1`.

### Dual Pipeline Modes

ALIN supports two run modes controlled by `--mode`:

| | Actionable (default) | Exploratory |
|---|---|---|
| Goal | Translational / clinical-trial-ready | Discovery / biology-first |
| Default arm | `liaki_role` | `default` |
| Drug bias | Favours druggable targets | Removed |
| Reward signals | druggability, synergy, coverage | essentiality, mutation, centrality |

```bash
python pan_cancer_xnode.py --all-cancers --triples --output results/ --mode actionable
python pan_cancer_xnode.py --all-cancers --triples --output results_exploratory/ --mode exploratory
python pan_cancer_xnode.py --all-cancers --triples --output results/ --mode actionable --strategy-arm default
python scripts/compare_modes.py --actionable results/ --exploratory results_exploratory/
python scripts/sensitivity_exploratory.py --cancers 10 --max-configs 20
```

When `--strategy-arm` is omitted, actionable defaults to `liaki_role` and exploratory defaults to `default`.

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

results_exploratory/
├── triple_combinations.csv
├── pan_cancer_summary.csv
└── all_findings.json

outputs/
├── comparisons/                 # Strategy-arm comparison runs
├── benchmark_audits/            # Focused benchmark viability audits
└── reports/
        └── validation_reports/
                └── validation_report.json
```

The committed top-level result bundles in this repository are preserved release artifacts. New generated comparisons, audits, and validation reports should be written under `outputs/` instead of the repository root.

`run_full_pipeline.sh` and `run_full_pipeline.ps1` default to `--skip-historical` because the dev-only historical comparison directories are not bundled in this public repo.

---

## Public Workflow Surface

Treat the following as the stable public-facing entry points:

- `pan_cancer_xnode.py` — main discovery CLI
- `run_full_pipeline.sh` and `run_full_pipeline.ps1` — cross-platform wrappers for fresh arm comparisons
- `scripts/run_pipeline.py` — focused benchmark-cancer runner
- `scripts/post_pipeline_validation.py` — post-run validation entry point
- `scripts/compare_modes.py` and `scripts/sensitivity_exploratory.py` — mode analysis tools
- `scripts/pipelines/run_strategy_arm_comparison.py` and `scripts/pipelines/run_benchmark_viability_audit.py` — latest comparison and audit workflows

Other scripts are preserved for manuscript generation, ablations, null models, or exploratory analysis rather than as the primary public CLI surface.

---

## Benchmark Results

Against 43 independently curated FDA-approved and Phase 2/3-validated multi-target combinations:

| Metric | Full (43 entries) | Testable (35 entries) |
|--------|-------------------|----------------------|
| Exact recall | 7.0% (3/43) | 8.6% (3/35) |
| Superset recall | 9.3% (4/43) | 11.4% (4/35) |
| Pair-overlap recall | 30.2% (13/43) | 37.1% (13/35) |
| Any-overlap recall | 44.2% (19/43) | 54.3% (19/35) |
| Cancer-level precision | 47.1% (8/17) | — |

- **vs. Random baseline:** 8.8% any-overlap (p < 0.001)
- **vs. Driver-gene baseline:** 16.3% pair-overlap
- **vs. Candidate-pool random:** 0.2% any-overlap (confirms scoring drives performance)
- **DrugComb synergy validation:** ALIN-predicted target pairs show significantly higher synergy (ZIP: Δ = +2.59, Cohen's d = 0.24, p = 0.0001; Bliss: p = 0.011) across 6 evaluable cancer types
- **Mean rank when matched:** 1.0 (top prediction)

---

## Repository Structure

```
├── build_lincs_index.py         # Optional LINCS index builder
├── pan_cancer_xnode.py          # Main discovery engine
├── gold_standard.py             # Clinical gold standard + benchmark functions
├── benchmarking_module.py       # Gold standard comparison, baselines
├── pharmacological_validation.py # Drug-target validation
├── parameter_tuning.py          # Scoring weight tuning
├── outcome_benchmark.py         # Outcome-based benchmarking
├── conftest.py                  # Pytest configuration
├── run_full_pipeline.sh         # Bash wrapper for the public workflow
├── run_full_pipeline.ps1        # PowerShell wrapper for the public workflow
├── setup_data.ps1               # Windows helper for setup_data.sh
├── alin/                        # Core library package
├── core/                        # Data structures, statistics
├── tests/                       # Unit and integration tests
├── scripts/                     # Stable workflows plus preserved research utilities
├── manuscript/                  # LaTeX source (paper.tex, supplementary.tex)
├── docs/                        # Documentation (DATA_AVAILABILITY, VERSION_INFO)
├── figures/                     # Generated figures (PNG/PDF)
├── results/                     # Primary quick-start pipeline output
└── outputs/                     # Generated comparisons, audits, and validation reports
```

Additional top-level `*_results/` directories are preserved publication artifacts rather than scratch workbench output.

---

## Module Overview

| Module | Role |
|--------|------|
| `pan_cancer_xnode.py` | Main discovery engine (DepMap, OmniPath, hitting set, triple finder) |
| `alin/validation.py` | Built-in validation (literature, PPI, drug synergy) |
| `alin/api_validators.py` | PubMed + STRING API validation with caching |
| `alin/genomic_data.py` | TCGA mutation loading and genomic relevance scoring |
| `alin/drug_sensitivity.py` | PRISM/GDSC drug sensitivity, gene–drug correlation |
| `alin/chembl_data.py` | ChEMBL-backed druggability lookup |
| `alin/run_modes.py` | Actionable vs exploratory configuration presets |
| `alin/strategy_arms.py` | Explicit strategy-arm selection and defaults |
| `alin/structural_triples.py` | Liaki-style structural triple construction |
| `alin/viability_scorecard.py` | Second-layer viability/support scorecard |
| `alin/clinical_trials.py` | ClinicalTrials.gov search |
| `alin/patient_stratification.py` | Patient subgroups, biomarkers, companion diagnostics |
| `alin/toxicity.py` | OpenTargets toxicity, tissue expression (cost function) |
| `alin/utils.py` | Shared utilities (sanitize_cancer_name, load_depmap_crispr_subset) |
| `benchmarking_module.py` | Gold standard comparison, random/top-genes baselines |
| `scripts/run_pipeline.py` | Run pipeline for gold standard cancer types |
| `scripts/pipelines/run_strategy_arm_comparison.py` | Fresh arm comparison and benchmarking workflow |
| `scripts/pipelines/run_benchmark_viability_audit.py` | Focused benchmark viability audit |
| `gold_standard.py` | Clinical gold standard entries + benchmark functions |

---

## Citation

If you use this pipeline, please cite:

- **ALIN Framework:** Erzurumluoğlu R. 2025. ALIN Framework (Adaptive Lethal Intersection Network): A Systems Biology Pipeline for Pan-Cancer Minimal Hitting Set Target Discovery. Zenodo. doi: [10.5281/zenodo.18517646](https://doi.org/10.5281/zenodo.18517646)
- PDAC combination therapy (source of extrapolated approach): Liaki V, Barrambana S, et al. 2025. A targeted combination therapy achieves effective pancreatic cancer regression and prevents tumor resistance. bioRxiv doi: [10.1101/2025.08.04.668325](https://doi.org/10.1101/2025.08.04.668325)
- DepMap: [depmap.org](https://depmap.org)
- OmniPath: [omnipathdb.org](https://omnipathdb.org)

---

## License

See LICENSE file.
