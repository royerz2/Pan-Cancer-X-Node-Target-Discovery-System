# Data Version Information

This document records the exact versions of data sources used in the ALIN Framework analysis.

## DepMap Data

- **Release**: DepMap Public 24Q4 (October 2024)
- **Access Date**: 2026-01-15
- **Download URL**: https://depmap.org/portal/download/
- **Files Used**:
  - `CRISPRGeneEffect.csv` (Chronos dependency scores)
  - `Model.csv` (Cell line metadata with OncoTree annotations)
  - `SubtypeMatrix.csv` (Binary subtype feature matrix)
- **Total Cell Lines**: 1,095 (varies by release)
- **Total Genes**: ~18,000

## OmniPath Network

- **Version**: OmniPath database accessed via API
- **Access Date**: 2026-01-15
- **API Endpoint**: https://omnipathdb.org/interactions
- **Parameters**:
  - `datasets`: omnipath, pathwayextra, kinaseextra
  - `types`: post_translational
- **Total Edges**: ~70,000 (directed signaling interactions)
- **Cache File**: `depmap_data/omnipath_network.csv`

## Validation Data Sources

### PRISM Drug Sensitivity
- **Release**: PRISM Repurposing 19Q4
- **Access Date**: 2026-01-15
- **URL**: https://depmap.org/repurposing/
- **Files**:
  - `primary-screen-replicate-collapsed-logfold-change.csv`
  - `primary-screen-replicate-collapsed-treatment-info.csv`

### Project Score (Sanger)
- **Release**: 2024
- **URL**: https://score.depmap.sanger.ac.uk/downloads
- **File**: `binaryDepScores.tsv`

### O'Neil 2016 Synergy Data
- **Publication**: O'Neil et al., Mol Cancer Ther 2016
- **PubMed ID**: 27397505
- **DOI**: 10.1158/1535-7163.MCT-15-0843

## API Services

### PubMed/NCBI
- **API**: E-utilities (eutils.ncbi.nlm.nih.gov)
- **No API key required**
- **Rate limit**: 3 requests/second without key

### STRING
- **API**: string-db.org
- **Version**: v12.0
- **No API key required**

### ClinicalTrials.gov
- **API**: clinicaltrials.gov/api/v2
- **No API key required**

### OpenTargets
- **API**: api.platform.opentargets.org/api/v4/graphql
- **No API key required**
- **Used for**: Toxicity scores, safety liabilities, tissue expression

## Software Environment

```
Python: 3.13.5
numpy: 2.4.2
pandas: 3.0.0
scipy: 1.17.0
networkx: 3.6.1
scikit-learn: 1.8.0
statsmodels: 0.14.4
```

See `requirements-lock.txt` for complete dependency list.

## Pipeline Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `θ_dep` | −0.5 | Chronos essentiality threshold |
| `θ_sel` | 0.30 | Selectivity fraction |
| Pan-essential cutoff | 0.90 | Fraction of lines for pan-essential exclusion |
| Co-essentiality distance | Jaccard (1 − J) | Binarized co-essentiality |
| Clustering | Ward's, k = 3–15 | Dynamic tree cut by silhouette |
| OmniPath hop limit | ≤ 4 | Maximum directed path length |
| Path pruning threshold | 0.3 | Minimum mean Chronos for path retention |
| Statistical deps | Welch t, BH q < 0.05, d > 0.3 | Cancer-specific gene selection |
| MHS solver | Greedy + ILP (≤500 genes) | Solver hierarchy |
| Hub penalty coefficient | 1.5 | Proportional penalty for hub genes |
| Perturbation bonus (β_pert) | 0.05/gene, max 0.15 | Bonus for feedback coverage |
| Coverage threshold | 0.70 | Minimum path coverage for ranked triples |
| Random seed | 42 | All stochastic components |

## Ablation and Significance Experiments

### Component Ablation (run_ablation.py)
- **Date**: 2026-02-09
- **Conditions**: 6 (full, no_omnipath, no_perturbation, no_coessentiality, no_statistical, no_hub_penalty)
- **Total runtime**: ~75 minutes (Apple M2, 16 GB)
- **Results**: `ablation_results/ablation_summary.json`

### Permutation Test (run_permutation_test.py)
- **Date**: 2026-02-09
- **N permutations**: 1,000
- **Null model**: Random 3-gene sampling from per-cancer candidate pool
- **Median pool size**: 1,717 genes
- **Runtime**: 12.1 seconds
- **Results**: `ablation_results/permutation_test_results.json`

### Degree-Preserving Network Null (run_degree_null_fast.py)
- **Date**: 2026-02-10
- **N permutations**: 20
- **Edge swaps per permutation**: 10 × |E|
- **Test cancers**: NSCLC, Melanoma, CRC, PDAC, Breast
- **Results**: `ablation_results/degree_null_results.json`

## Reproducibility Notes

1. **DepMap releases quarterly** - results may vary slightly with different releases
2. **OmniPath is continuously updated** - we cache the network to ensure reproducibility
3. **API responses may change** - validation scores may vary with database updates
4. **Random seeds** - Set to 42 where applicable for reproducibility

## How to Verify

```bash
# Check DepMap file hashes
md5sum depmap_data/CRISPRGeneEffect.csv
md5sum depmap_data/Model.csv

# Check OmniPath cache date
ls -la depmap_data/omnipath_network.csv

# Run with same random seed
python pan_cancer_xnode.py --all-cancers --seed 42
```
