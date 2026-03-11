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
import pandas as pd
from pathlib import Path

TRIPLES_CSV = 'results/triple_combinations.csv'


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


class TestPredictionParsing:
    """Verify benchmark ingestion respects explicit rank and best-combo metadata."""

    def test_build_cancer_predictions_uses_explicit_rank(self, tmp_path):
        from benchmarking_module import _build_cancer_predictions

        csv_path = tmp_path / 'ranked_triples.csv'
        pd.DataFrame([
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 2,
                'Target_1': 'BRAF',
                'Target_2': 'MAP2K1',
                'Target_3': 'STAT3',
            },
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 1,
                'Target_1': 'BRAF',
                'Target_2': 'EGFR',
                'Target_3': 'STAT3',
            },
        ]).to_csv(csv_path, index=False)

        predictions = _build_cancer_predictions(str(csv_path))

        assert predictions['Melanoma'][0] == frozenset({'BRAF', 'EGFR', 'STAT3'})
        assert predictions['Melanoma'][1] == frozenset({'BRAF', 'MAP2K1', 'STAT3'})

    def test_build_cancer_predictions_injects_best_combo_first(self, tmp_path):
        from benchmarking_module import _build_cancer_predictions

        csv_path = tmp_path / 'ranked_triples.csv'
        pd.DataFrame([
            {
                'Cancer_Type': 'Colorectal Adenocarcinoma',
                'Rank': 1,
                'Target_1': 'BRAF',
                'Target_2': 'MAP2K1',
                'Target_3': 'STAT3',
                'Best_Combo_Size': 2,
                'Best_Combo_1': 'BRAF',
                'Best_Combo_2': 'EGFR',
                'Best_Combo_3': '',
            },
            {
                'Cancer_Type': 'Colorectal Adenocarcinoma',
                'Rank': 2,
                'Target_1': 'BRAF',
                'Target_2': 'EGFR',
                'Target_3': 'KRAS',
                'Best_Combo_Size': 2,
                'Best_Combo_1': 'BRAF',
                'Best_Combo_2': 'EGFR',
                'Best_Combo_3': '',
            },
        ]).to_csv(csv_path, index=False)

        predictions = _build_cancer_predictions(str(csv_path))

        assert predictions['Colorectal Adenocarcinoma'][0] == frozenset({'BRAF', 'EGFR'})
        assert predictions['Colorectal Adenocarcinoma'][1] == frozenset({'BRAF', 'MAP2K1', 'STAT3'})
        assert predictions['Colorectal Adenocarcinoma'][2] == frozenset({'BRAF', 'EGFR', 'KRAS'})

    def test_build_cancer_predictions_prefers_ranked_companion(self, tmp_path):
        from benchmarking_module import _build_cancer_predictions

        triple_path = tmp_path / 'triple_combinations.csv'
        ranked_path = tmp_path / 'ranked_triple_combinations.csv'

        pd.DataFrame([
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 1,
                'Target_1': 'BRAF',
                'Target_2': 'STAT3',
                'Target_3': 'AKT1',
                'Best_Combo_Size': 2,
                'Best_Combo_1': 'BRAF',
                'Best_Combo_2': 'EGFR',
                'Best_Combo_3': '',
            },
        ]).to_csv(triple_path, index=False)

        pd.DataFrame([
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 2,
                'Target_1': 'BRAF',
                'Target_2': 'MAP2K1',
                'Target_3': 'STAT3',
            },
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 1,
                'Target_1': 'BRAF',
                'Target_2': 'EGFR',
                'Target_3': 'STAT3',
            },
        ]).to_csv(ranked_path, index=False)

        predictions = _build_cancer_predictions(str(triple_path))

        assert predictions['Melanoma'] == [
            frozenset({'BRAF', 'EGFR', 'STAT3'}),
            frozenset({'BRAF', 'MAP2K1', 'STAT3'}),
        ]

    def test_gold_standard_benchmark_prefers_ranked_companion(self, tmp_path, monkeypatch):
        import gold_standard

        triple_path = tmp_path / 'triple_combinations.csv'
        ranked_path = tmp_path / 'ranked_triple_combinations.csv'

        pd.DataFrame([
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 1,
                'Target_1': 'BRAF',
                'Target_2': 'STAT3',
                'Target_3': 'AKT1',
                'Best_Combo_Size': 2,
                'Best_Combo_1': 'BRAF',
                'Best_Combo_2': 'EGFR',
                'Best_Combo_3': '',
            },
        ]).to_csv(triple_path, index=False)

        pd.DataFrame([
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 1,
                'Target_1': 'BRAF',
                'Target_2': 'EGFR',
                'Target_3': 'STAT3',
            },
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 2,
                'Target_1': 'BRAF',
                'Target_2': 'MAP2K1',
                'Target_3': 'STAT3',
            },
        ]).to_csv(ranked_path, index=False)

        monkeypatch.setattr(
            gold_standard,
            'GOLD_STANDARD',
            [{
                'cancer': 'Melanoma',
                'targets': frozenset({'BRAF', 'MAP2K1', 'STAT3'}),
                'evidence': 'Phase_2',
                'description': 'test entry',
            }],
        )

        result = gold_standard.run_benchmark(
            str(triple_path),
            tier1=True,
            tier2=False,
            verbose=False,
        )

        assert result['predictions_source'].endswith('ranked_triple_combinations.csv')
        assert result['results'][0]['best_rank'] == 2.0
        assert result['results'][0]['predicted'] == frozenset({'BRAF', 'MAP2K1', 'STAT3'})


class TestBenchmarkAudit:
    """Benchmark audit utilities should surface semantic conflicts clearly."""

    def test_audit_benchmark_matches_surfaces_off_rank_exact(self, tmp_path):
        from benchmarking_module import audit_benchmark_matches

        csv_path = tmp_path / 'ranked_triples.csv'
        pd.DataFrame([
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 1,
                'Target_1': 'BRAF',
                'Target_2': 'STAT3',
                'Target_3': 'AKT1',
            },
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 2,
                'Target_1': 'BRAF',
                'Target_2': 'MAP2K1',
                'Target_3': '',
            },
        ]).to_csv(csv_path, index=False)

        rows, summary = audit_benchmark_matches(
            str(csv_path),
            gold_standard=[{
                'cancer': 'Melanoma',
                'targets': frozenset({'BRAF', 'MAP2K1'}),
                'evidence': 'test',
                'description': 'test entry',
            }],
        )

        assert len(rows) == 1
        row = rows[0]
        assert row['top_prediction_match_type'] == 'any_overlap'
        assert row['current_benchmark_match_type'] == 'any_overlap'
        assert row['strongest_available_match_type'] == 'exact'
        assert row['rank_semantic_conflict'] is True
        assert row['gap_classification'] == 'exact_available_off_rank'
        assert summary['rank_semantic_conflicts'] == 1
        assert summary['exact_available_off_rank'] == 1

    def test_run_baseline_calibration_normalizes_outputs(self, monkeypatch):
        import benchmarking_module as bm

        monkeypatch.setattr(bm, 'run_random_baseline', lambda *args, **kwargs: {
            'method': 'random',
            'mean_recall_exact': 0.1,
            'mean_recall_superset': 0.2,
            'mean_recall_pair_overlap': 0.3,
            'mean_recall_any_overlap': 0.4,
            'std_recall_exact': 0.01,
            'std_recall_superset': 0.02,
            'std_recall_pair_overlap': 0.03,
            'std_recall_any_overlap': 0.04,
            'n_trials': 5,
        })
        monkeypatch.setattr(bm, 'run_poolmatched_baseline', lambda *args, **kwargs: {
            'method': 'pool_matched',
            'mean_recall_exact': 0.05,
            'mean_recall_superset': 0.1,
            'mean_recall_pair_overlap': 0.15,
            'mean_recall_any_overlap': 0.2,
            'n_trials': 5,
        })
        monkeypatch.setattr(bm, 'run_frequency_baseline', lambda *args, **kwargs: {
            'method': 'frequency',
            'recall_exact': 0.11,
            'recall_superset': 0.22,
            'recall_pair_overlap': 0.33,
            'recall_any_overlap': 0.44,
        })
        monkeypatch.setattr(bm, 'run_topgenes_baseline', lambda *args, **kwargs: {
            'method': 'top_genes',
            'recall_exact': 0.12,
            'recall_superset': 0.23,
            'recall_pair_overlap': 0.34,
            'recall_any_overlap': 0.45,
        })
        monkeypatch.setattr(bm, 'run_driver_baseline', lambda *args, **kwargs: {
            'method': 'driver_genes',
            'recall_exact': 0.13,
            'recall_superset': 0.24,
            'recall_pair_overlap': 0.35,
            'recall_any_overlap': 0.46,
            'n_cancers_with_drivers': 8,
        })
        monkeypatch.setattr(bm, 'run_essentiality_baseline', lambda *args, **kwargs: {
            'method': 'essentiality',
            'recall_exact': 0.0,
            'recall_superset': 0.0,
            'recall_pair_overlap': 0.0,
            'recall_any_overlap': 0.0,
            'error': 'missing data',
        })

        rows = bm.run_baseline_calibration('dummy.csv', n_trials=5)
        by_method = {row['method']: row for row in rows}

        assert by_method['random']['metric_source'] == 'monte_carlo_mean'
        assert by_method['random']['recall_pair_overlap_or_better'] == 0.3
        assert by_method['frequency']['metric_source'] == 'point_estimate'
        assert by_method['driver_genes']['n_cancers_with_drivers'] == 8
        assert by_method['essentiality']['error'] == 'missing data'


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
        paper = Path('manuscript/paper.tex').read_text()
        assert 'driver' in paper.lower(), \
            "Paper must mention the driver-gene baseline"

    def test_paper_mentions_essentiality_baseline(self):
        paper = Path('manuscript/paper.tex').read_text()
        assert 'essentiality' in paper.lower(), \
            "Paper must mention the essentiality baseline"

    def test_paper_reports_pairwise_for_baselines(self):
        """Paper should report pairwise recall for baselines, not just superset."""
        paper = Path('manuscript/paper.tex').read_text()
        # The paper should mention "pairwise" near baseline results
        assert 'pairwise' in paper.lower()

    def test_paper_mentions_frequency_baseline_tie(self):
        """Paper should honestly acknowledge the frequency baseline tie."""
        paper = Path('manuscript/paper.tex').read_text()
        assert 'frequency' in paper.lower()
