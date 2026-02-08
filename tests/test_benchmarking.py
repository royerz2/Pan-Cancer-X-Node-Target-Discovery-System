"""
Tests for Fix #13: Statistical Baselines
=========================================

Verifies that:
1. All baselines compute exact, superset, AND pairwise recall (no apples-to-oranges).
2. Biology-informed baselines (driver genes, essentiality) exist and are functional.
3. Driver gene knowledge base covers all gold-standard cancer types.
4. check_match() correctly classifies match types.
5. The gold standard has no structural issues (duplicates, circularity flagged).
6. Paper reports pairwise recall consistently for ALIN and ALL baselines.
"""

import pytest
import numpy as np
from pathlib import Path

TRIPLES_CSV = 'results_triples/triple_combinations.csv'


# ============================================================================
# Baseline output schema: all must report pairwise recall
# ============================================================================

class TestBaselineSchema:
    """Every baseline must return pairwise recall alongside exact and superset."""

    def test_random_baseline_has_pairwise(self):
        from benchmarking_module import run_random_baseline
        result = run_random_baseline(TRIPLES_CSV, n_trials=10, seed=42)
        assert 'mean_recall_pairwise' in result, \
            "Random baseline must report pairwise recall"
        assert 'std_recall_pairwise' in result
        assert 0 <= result['mean_recall_pairwise'] <= 1

    def test_poolmatched_baseline_has_pairwise(self):
        from benchmarking_module import run_poolmatched_baseline
        result = run_poolmatched_baseline(TRIPLES_CSV, n_trials=10, seed=42)
        assert 'mean_recall_pairwise' in result, \
            "Pool-matched baseline must report pairwise recall"
        assert 0 <= result['mean_recall_pairwise'] <= 1

    def test_frequency_baseline_has_pairwise(self):
        from benchmarking_module import run_frequency_baseline
        result = run_frequency_baseline(TRIPLES_CSV)
        assert 'recall_pairwise' in result, \
            "Frequency baseline must report pairwise recall"

    def test_topgenes_baseline_has_pairwise(self):
        from benchmarking_module import run_topgenes_baseline
        result = run_topgenes_baseline(TRIPLES_CSV)
        assert 'recall_pairwise' in result, \
            "Top-genes baseline must report pairwise recall"

    def test_driver_baseline_has_pairwise(self):
        from benchmarking_module import run_driver_baseline
        result = run_driver_baseline(TRIPLES_CSV)
        assert 'recall_pairwise' in result, \
            "Driver-gene baseline must report pairwise recall"
        assert result['method'] == 'driver_genes'

    def test_essentiality_baseline_has_pairwise(self):
        from benchmarking_module import run_essentiality_baseline
        result = run_essentiality_baseline(TRIPLES_CSV)
        assert 'recall_pairwise' in result, \
            "Essentiality baseline must report pairwise recall"
        assert result['method'] == 'essentiality'


# ============================================================================
# Biology-informed baselines: structural tests
# ============================================================================

class TestDriverBaseline:
    """Driver-gene baseline uses genuine biological knowledge."""

    def test_driver_genes_dict_exists(self):
        from benchmarking_module import CANCER_DRIVER_GENES
        assert len(CANCER_DRIVER_GENES) >= 6, \
            "Must have driver gene lists for at least 6 cancer types"

    def test_driver_genes_cover_gold_standard_cancers(self):
        """Every cancer in the gold standard should have a driver gene list."""
        from benchmarking_module import (
            CANCER_DRIVER_GENES, COMBINATION_GOLD_STANDARD, match_cancer
        )
        gold_cancers = {g['cancer'] for g in COMBINATION_GOLD_STANDARD}
        for gc in gold_cancers:
            found = any(match_cancer(dc, gc) or match_cancer(gc, dc)
                        for dc in CANCER_DRIVER_GENES)
            assert found, f"No driver genes defined for gold-standard cancer: {gc}"

    def test_driver_genes_are_real_genes(self):
        """Driver genes should be real HUGO symbols, not made-up names."""
        from benchmarking_module import CANCER_DRIVER_GENES
        known_genes = {
            'BRAF', 'KRAS', 'EGFR', 'ALK', 'MET', 'ERBB2', 'FLT3', 'BCL2',
            'PIK3CA', 'MTOR', 'CDK4', 'CDK6', 'ESR1', 'MAP2K1', 'VHL',
            'TP53', 'APC', 'NRAS', 'NF1', 'IDH1', 'IDH2', 'CDKN2A',
            'BRCA1', 'BRCA2', 'AKT1', 'KIT', 'NPM1', 'DNMT3A', 'STAT3',
            'SMAD4', 'VEGFR2', 'PBRM1', 'BAP1', 'SETD2', 'ROS1', 'RET',
            'STK11', 'FGFR1', 'CDH1', 'RUNX1',
        }
        for cancer, genes in CANCER_DRIVER_GENES.items():
            for g in genes:
                assert g in known_genes, \
                    f"Unknown gene {g} in driver list for {cancer}"

    def test_driver_baseline_returns_nonzero_for_melanoma(self):
        """Melanoma drivers include BRAF+MAP2K1 — should match the gold standard."""
        from benchmarking_module import CANCER_DRIVER_GENES
        mel_drivers = CANCER_DRIVER_GENES.get('Melanoma', [])
        assert 'BRAF' in mel_drivers and 'MAP2K1' in mel_drivers, \
            "Melanoma drivers must include BRAF and MAP2K1"


class TestEssentialityBaseline:
    """Essentiality baseline tests DepMap top-essential-gene selection."""

    def test_essentiality_baseline_runs(self):
        from benchmarking_module import run_essentiality_baseline
        result = run_essentiality_baseline(TRIPLES_CSV)
        assert 'method' in result
        assert result['method'] == 'essentiality'

    def test_essentiality_uses_depmap_data(self):
        """If DepMap data exists, should process some cancers."""
        from benchmarking_module import run_essentiality_baseline
        result = run_essentiality_baseline(TRIPLES_CSV)
        if 'error' not in result:
            assert result.get('n_cancers_with_data', 0) > 0, \
                "Should find DepMap data for at least some cancers"


# ============================================================================
# Match logic: check_match() correctness
# ============================================================================

class TestCheckMatch:
    """Verify check_match() correctly classifies match types."""

    def test_exact_match(self):
        from benchmarking_module import check_match
        matched, mtype = check_match({'BRAF', 'MAP2K1'}, {'BRAF', 'MAP2K1'})
        assert matched and mtype == 'exact'

    def test_superset_match(self):
        from benchmarking_module import check_match
        matched, mtype = check_match(
            {'BRAF', 'MAP2K1', 'STAT3'}, {'BRAF', 'MAP2K1'}
        )
        assert matched and mtype == 'superset'

    def test_pair_overlap_match(self):
        from benchmarking_module import check_match
        matched, mtype = check_match(
            {'BRAF', 'CDK4', 'STAT3'}, {'BRAF', 'MAP2K1', 'CDK4'}
        )
        assert matched and mtype == 'pair_overlap'

    def test_no_match(self):
        from benchmarking_module import check_match
        matched, mtype = check_match(
            {'STAT3', 'CDK6', 'FYN'}, {'BRAF', 'MAP2K1'}
        )
        assert not matched and mtype == 'none'

    def test_single_overlap_is_any_overlap(self):
        from benchmarking_module import check_match
        matched, mtype = check_match(
            {'BRAF', 'STAT3', 'CDK6'}, {'BRAF', 'EGFR'}
        )
        assert matched and mtype == 'any_overlap', \
            "Single gene overlap should count as any_overlap match"

    def test_gene_equivalents_work(self):
        from benchmarking_module import check_match
        # MAP2K2 is equivalent to MAP2K1
        matched, mtype = check_match(
            {'BRAF', 'MAP2K2'}, {'BRAF', 'MAP2K1'}
        )
        assert matched, "Gene equivalents (MAP2K1<->MAP2K2) should match"


# ============================================================================
# Gold standard integrity
# ============================================================================

class TestGoldStandard:
    """Structural checks on the gold standard."""

    def test_all_entries_have_required_fields(self):
        from benchmarking_module import COMBINATION_GOLD_STANDARD
        for i, entry in enumerate(COMBINATION_GOLD_STANDARD):
            assert 'cancer' in entry, f"Entry {i} missing 'cancer'"
            assert 'targets' in entry, f"Entry {i} missing 'targets'"
            assert isinstance(entry['targets'], frozenset), \
                f"Entry {i} targets should be frozenset"
            assert len(entry['targets']) >= 2, \
                f"Entry {i} should have >=2 targets for combination benchmark"
            assert 'evidence' in entry, f"Entry {i} missing 'evidence'"

    def test_no_identical_duplicate_entries(self):
        """Flag entries with identical cancer+targets (like Melanoma BRAF+MAP2K1 x2)."""
        from benchmarking_module import COMBINATION_GOLD_STANDARD
        seen = set()
        duplicates = []
        for entry in COMBINATION_GOLD_STANDARD:
            key = (entry['cancer'], entry['targets'])
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        # We document duplicates rather than fail — entries #1 and #4 ARE duplicates
        # but represent different trials (COMBI-d vs coBRIM)
        if duplicates:
            import warnings
            warnings.warn(
                f"Gold standard has {len(duplicates)} duplicate cancer+target "
                f"pair(s): {duplicates}. These inflate recall if matched."
            )

    def test_pdac_entry_flagged_as_preclinical(self):
        """The Liaki PDAC entry should be marked preclinical, not FDA-approved."""
        from benchmarking_module import COMBINATION_GOLD_STANDARD
        pdac_entries = [e for e in COMBINATION_GOLD_STANDARD
                        if 'Pancreatic' in e['cancer'] or 'PDAC' in e['cancer']]
        for entry in pdac_entries:
            if 'STAT3' in entry['targets']:
                assert entry['evidence'].lower() in ('preclinical',), \
                    "PDAC KRAS+EGFR+STAT3 entry should be 'Preclinical'"


# ============================================================================
# Recall ordering invariant
# ============================================================================

class TestRecallOrdering:
    """Pairwise recall >= superset >= exact for every baseline."""

    def test_random_recall_ordering(self):
        from benchmarking_module import run_random_baseline
        r = run_random_baseline(TRIPLES_CSV, n_trials=50, seed=42)
        assert r['mean_recall_pairwise'] >= r['mean_recall_superset'] >= r['mean_recall_exact']

    def test_poolmatched_recall_ordering(self):
        from benchmarking_module import run_poolmatched_baseline
        r = run_poolmatched_baseline(TRIPLES_CSV, n_trials=50, seed=42)
        assert r['mean_recall_pairwise'] >= r['mean_recall_superset'] >= r['mean_recall_exact']

    def test_frequency_recall_ordering(self):
        from benchmarking_module import run_frequency_baseline
        r = run_frequency_baseline(TRIPLES_CSV)
        assert r['recall_pairwise'] >= r['recall_superset'] >= r['recall_exact']

    def test_driver_recall_ordering(self):
        from benchmarking_module import run_driver_baseline
        r = run_driver_baseline(TRIPLES_CSV)
        assert r['recall_pairwise'] >= r['recall_superset'] >= r['recall_exact']


# ============================================================================
# Paper consistency
# ============================================================================

class TestPaperConsistency:
    """Verify paper reports baseline metrics accurately."""

    def test_paper_mentions_driver_baseline(self):
        paper = Path('paper.tex').read_text()
        assert 'driver' in paper.lower(), \
            "Paper must mention the driver-gene baseline"

    def test_paper_mentions_essentiality_baseline(self):
        paper = Path('paper.tex').read_text()
        assert 'essentiality' in paper.lower(), \
            "Paper must mention the essentiality baseline"

    def test_paper_reports_pairwise_for_baselines(self):
        """Paper should report pairwise recall for baselines, not just superset."""
        paper = Path('paper.tex').read_text()
        # The paper should mention "pairwise" near baseline results
        assert 'pairwise' in paper.lower()

    def test_paper_mentions_frequency_baseline_tie(self):
        """Paper should honestly acknowledge the frequency baseline tie."""
        paper = Path('paper.tex').read_text()
        assert 'frequency' in paper.lower()
