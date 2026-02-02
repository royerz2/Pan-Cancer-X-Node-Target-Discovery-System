# Manuscript Outline: Pan-Cancer X-Node Target Discovery

## Title (draft)
**Pan-Cancer X-Node Target Discovery: A Systems Biology Pipeline for Triple Drug Combination Prediction**

---

## Abstract (~250 words)
- **Background:** Drug combinations overcome resistance; identifying optimal targets is challenging
- **Methods:** Generalized minimal hitting set + network topology (X-nodes) + synergy/resistance scoring
- **Data:** DepMap CRISPR, OmniPath, PRISM, ClinicalTrials.gov
- **Results:** 61% recall vs gold standard; 5 novel combinations with no existing trials; CDK2+KRAS+STAT3 pan-cancer pattern
- **Conclusion:** Reproducible pipeline for triple combination discovery; ready for experimental validation

---

## Introduction
- Drug resistance in cancer
- Rational combination design (PDAC X-node paper PMID:33753453)
- Gap: pan-cancer generalization, triple combinations
- Our contribution: integrated pipeline, benchmarking, validation

---

## Methods
1. **Data:** DepMap (CRISPR, Model, Subtype), OmniPath, PRISM
2. **Cancer type mapping:** OncoTree
3. **Viability path inference:** Essential genes, signaling paths, differential dependency
4. **Minimal hitting set:** Cost function (toxicity, specificity, druggability)
5. **Systems biology triples:** X-node network analysis, synergy scoring, resistance prediction
6. **Validation:** PubMed, STRING, ClinicalTrials.gov, PRISM drug sensitivity
7. **Benchmarking:** 23 gold standard combinations, random/top-genes baselines

---

## Results
1. **Pan-cancer discovery:** X cancer types, Y triple combinations
2. **Benchmark:** 61% recall, outperforms baselines
3. **Novel combinations:** 5 with no clinical trials
4. **CDK2+KRAS+STAT3:** Recurrent pattern across 6 cancers
5. **Patient stratification:** 75% addressable, companion diagnostic

---

## Discussion
- Strengths: Reproducible, validated, benchmarked
- Limitations: In silico only; experimental validation needed
- Clinical translation: ComboMATCH STAT3 proposal

---

## Supplementary
- Table S1: Full triple combinations
- Table S2: Gold standard benchmark
- Table S3: Novel combinations for lab testing
- Figure S1: Pipeline flowchart
- Data availability, code availability
