# Data Availability

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
