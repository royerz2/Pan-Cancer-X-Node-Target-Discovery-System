"""
ALIN Framework — Adaptive Lethal Intersection Network
=====================================================
A computational pipeline for discovering optimal triple drug combinations
across cancer types using systems biology and minimal hitting set optimization.
"""

from alin.utils import sanitize_cancer_name, load_depmap_crispr_subset

__all__ = [
    "sanitize_cancer_name",
    "load_depmap_crispr_subset",
]

# Submodules (lazy import for optional dependencies)
# - alin.validation: ValidationEngine, CombinationValidation, etc.
# - alin.api_validators: CombinedAPIValidator
# - alin.drug_sensitivity: DrugSensitivityValidator
# - alin.clinical_trials: ClinicalTrialMatcher
# - alin.patient_stratification: PatientStratifier
# - alin.toxicity: get_opentargets_toxicity, etc.
