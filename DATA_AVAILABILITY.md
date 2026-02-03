# Data Availability

## WHERE TO DOWNLOAD (direct links)

### DepMap (required for pipeline)
- **Portal (choose files):** https://depmap.org/portal/download/
- **Full release (Figshare):** https://figshare.com/articles/dataset/DepMap_24Q4_Public/27993248 (or search "DepMap" for latest quarter)
- **You need:** `Model.csv`, `CRISPRGeneEffect.csv`, `SubtypeMatrix.csv` → save into **`depmap_data/`**

### Project Score (optional – Sanger validation)
- **Download page:** https://score.depmap.sanger.ac.uk/downloads
- **Direct zip links (binary matrix, good for validation):**
  - https://cog.sanger.ac.uk/cmp/download/binaryDepScores.tsv.zip
  - https://cog.sanger.ac.uk/cmp/download/essentiality_matrices.zip
- **After download:** Unzip, put `binaryDepScores.tsv` (or build `project_score_crispr.csv` from essentiality matrix) into **`validation_data/`**

### O'Neil 2016 synergy (optional – synergy validation)
- **Article + supplementary data:** https://aacrjournals.org/mct/article/15/6/1155/92159/An-Unbiased-Oncology-Compound-Screen-to-Identify
- **PubMed:** https://pubmed.ncbi.nlm.nih.gov/27397505
- **After download:** Convert supplementary table to CSV with columns `gene1`, `gene2`, `synergy_score` (or `drug1`, `drug2`, `synergy_score`) → save as **`validation_data/oneil_2016_synergy.csv`**

### PRISM (optional – drug sensitivity)
- **Download page:** https://depmap.org/repurposing/
- **Direct file links (Figshare):**
  - Primary (replicate-collapsed): https://ndownloader.figshare.com/files/20237709 (`primary-screen-replicate-collapsed-logfold-change.csv`)
  - Primary (treatment info): https://ndownloader.figshare.com/files/20237715 (`primary-screen-replicate-collapsed-treatment-info.csv`)
  - Secondary (dose-response): https://ndownloader.figshare.com/files/20237739 (`secondary-screen-dose-response-curve-parameters.csv`)
- Save into **`drug_sensitivity_data/`**

---

## What to download NOW (summary)

| Priority | What | Where to put it | Where to get it |
|----------|------|-----------------|-----------------|
| **Required** | DepMap (CRISPR + cell line metadata) | `depmap_data/` | [depmap.org/portal/download](https://depmap.org/portal/download/) → Model.csv, CRISPRGeneEffect.csv, SubtypeMatrix.csv |
| Optional | Project Score (for Sanger validation) | `validation_data/` | [score.depmap.sanger.ac.uk/downloads](https://score.depmap.sanger.ac.uk/downloads) — or direct: [binaryDepScores.tsv.zip](https://cog.sanger.ac.uk/cmp/download/binaryDepScores.tsv.zip) |
| Optional | O'Neil 2016 synergy (for synergy validation) | `validation_data/oneil_2016_synergy.csv` | [AACR article (supplementary)](https://aacrjournals.org/mct/article/15/6/1155/92159/An-Unbiased-Oncology-Compound-Screen-to-Identify) |
| Optional | PRISM (drug sensitivity) | `drug_sensitivity_data/` | [depmap.org/repurposing](https://depmap.org/repurposing/) — or direct Figshare links above |

**No download needed for:** tissue expression weight (OpenTargets API), FDA ADRs (OpenFDA API), cBioPortal validations — all use public APIs.

---

## Required Data

| Dataset | Source | URL | Files |
|---------|--------|-----|-------|
| **DepMap** | Broad Institute | https://depmap.org/portal/download/ | Model.csv, CRISPRGeneEffect.csv, SubtypeMatrix.csv |
| **OmniPath** | Built-in / API | https://omnipathdb.org | Cancer signaling network (built-in) |

## Optional Data (for validation and refinement)

| Dataset | Source | URL | Files |
|---------|--------|-----|-------|
| **PRISM Primary** | DepMap Repurposing | https://depmap.org/repurposing/ | primary-screen-replicate-collapsed-logfold-change.csv, primary-screen-replicate-collapsed-treatment-info.csv |
| **PRISM Secondary** | DepMap Repurposing | https://depmap.org/repurposing/ | secondary-screen-dose-response-curve-parameters.csv |
| **GDSC** | Sanger Institute | https://www.cancerrxgene.org/downloads | GDSC2_fitted_dose_response.xlsx |
| **CCLE Expression** | DepMap | https://depmap.org/portal/download/ | CCLE_expression.csv or OmicsExpressionProteinCodingGenesTPMLogp1.csv (for expression-filtered essentiality) |

## API Access (optional)

- **PubMed:** https://eutils.ncbi.nlm.nih.gov (no key required)
- **STRING:** https://string-db.org (no key required)
- **ClinicalTrials.gov:** https://clinicaltrials.gov/api (no key required)
- **OpenTargets:** https://api.platform.opentargets.org/api/v4/graphql (no key required; for toxicity/safety enhancement)

## License

- DepMap: CC BY 4.0
- PRISM: CC BY 4.0
- OmniPath: CC BY 4.0
