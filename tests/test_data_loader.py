"""
Unit Tests for Data Loader
==========================
Tests for DepMap and OmniPath data loading utilities.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Set

from core.data_structures import ViabilityPath


class TestDepMapLoaderMock:
    """Tests for DepMapLoader with mocked data"""
    
    def create_mock_model_df(self) -> pd.DataFrame:
        """Create mock Model.csv data"""
        return pd.DataFrame({
            'ModelID': ['ACH-001', 'ACH-002', 'ACH-003', 'ACH-004', 'ACH-005'],
            'OncotreePrimaryDisease': [
                'Pancreatic Adenocarcinoma',
                'Pancreatic Adenocarcinoma',
                'Non-Small Cell Lung Cancer',
                'Non-Small Cell Lung Cancer',
                'Melanoma'
            ],
            'OncotreeLineage': ['Pancreas', 'Pancreas', 'Lung', 'Lung', 'Skin'],
            'CellLineName': ['PANC1', 'MIAPACA2', 'A549', 'H1975', 'A375'],
        }).set_index('ModelID')
    
    def create_mock_crispr_df(self) -> pd.DataFrame:
        """Create mock CRISPRGeneEffect data"""
        np.random.seed(42)
        genes = ['KRAS', 'BRAF', 'STAT3', 'CDK4', 'MYC', 'RPL11']  # RPL11 = pan-essential
        cell_lines = ['ACH-001', 'ACH-002', 'ACH-003', 'ACH-004', 'ACH-005']
        
        # Create dependency scores (negative = essential)
        data = np.random.normal(-0.3, 0.3, (len(cell_lines), len(genes)))
        
        # Make KRAS essential in pancreatic lines
        data[0:2, 0] = -0.8  # KRAS in PAAD
        
        # Make RPL11 pan-essential
        data[:, 5] = -1.0  # RPL11 in all lines
        
        df = pd.DataFrame(data, index=cell_lines, columns=genes)
        return df
    
    def test_cancer_type_normalization(self):
        """Test that cancer type aliases are normalized"""
        from pan_cancer_xnode import normalize_cancer_type, CANCER_TYPE_ALIASES
        
        assert normalize_cancer_type('PAAD') == 'Pancreatic Adenocarcinoma'
        assert normalize_cancer_type('PDAC') == 'Pancreatic Adenocarcinoma'
        assert normalize_cancer_type('NSCLC') == 'Non-Small Cell Lung Cancer'
        assert normalize_cancer_type('MEL') == 'Melanoma'
        
        # Unknown should return as-is
        assert normalize_cancer_type('Unknown Cancer') == 'Unknown Cancer'
    
    def test_get_cell_lines_for_cancer(self):
        """Test filtering cell lines by cancer type"""
        model_df = self.create_mock_model_df()
        
        # Simulate get_cell_lines_for_cancer
        cancer_type = 'Pancreatic Adenocarcinoma'
        matches = model_df[model_df['OncotreePrimaryDisease'] == cancer_type].index.tolist()
        
        assert len(matches) == 2
        assert 'ACH-001' in matches
        assert 'ACH-002' in matches
    
    def test_pan_essential_gene_detection(self):
        """Test identification of pan-essential genes"""
        crispr_df = self.create_mock_crispr_df()
        
        # Pan-essential: essential in >90% of lines
        threshold = -0.5
        essential_fraction = (crispr_df < threshold).mean(axis=0)
        pan_essential = set(essential_fraction[essential_fraction > 0.9].index)
        
        # RPL11 should be pan-essential (we set it to -1.0 in all lines)
        assert 'RPL11' in pan_essential
        
        # KRAS should NOT be pan-essential (only essential in some lines)
        assert 'KRAS' not in pan_essential
    
    def test_gene_essentiality_in_cancer(self):
        """Test computing gene essentiality in specific cancer"""
        crispr_df = self.create_mock_crispr_df()
        model_df = self.create_mock_model_df()
        
        # Get PAAD cell lines
        paad_lines = model_df[model_df['OncotreePrimaryDisease'] == 'Pancreatic Adenocarcinoma'].index
        paad_lines = [l for l in paad_lines if l in crispr_df.index]
        
        # Compute mean dependency for KRAS in PAAD
        kras_paad_dep = crispr_df.loc[paad_lines, 'KRAS'].mean()
        
        # Should be highly essential (negative score)
        assert kras_paad_dep < -0.5


class TestOmniPathLoaderMock:
    """Tests for OmniPath network loading"""
    
    def create_mock_network(self) -> pd.DataFrame:
        """Create mock signaling network"""
        edges = [
            ('EGFR', 'KRAS', 'activation', 'KEGG'),
            ('KRAS', 'BRAF', 'activation', 'KEGG'),
            ('BRAF', 'MAP2K1', 'activation', 'KEGG'),
            ('MAP2K1', 'MAPK1', 'activation', 'KEGG'),
            ('JAK1', 'STAT3', 'activation', 'KEGG'),
            ('STAT3', 'MYC', 'activation', 'Reactome'),
            ('SRC', 'STAT3', 'activation', 'PhosphoSite'),
        ]
        return pd.DataFrame(edges, columns=['source', 'target', 'interaction_type', 'database'])
    
    def test_downstream_targets(self):
        """Test finding downstream targets"""
        network_df = self.create_mock_network()
        
        # Simple 1-hop downstream
        egfr_direct = set(network_df[network_df['source'] == 'EGFR']['target'])
        assert 'KRAS' in egfr_direct
        
        # 2-hop downstream
        def get_downstream(gene: str, depth: int = 2) -> Set[str]:
            visited = {gene}
            frontier = {gene}
            
            for _ in range(depth):
                new_frontier = set()
                for g in frontier:
                    targets = network_df[network_df['source'] == g]['target'].tolist()
                    new_frontier.update(targets)
                new_frontier -= visited
                visited.update(new_frontier)
                frontier = new_frontier
            
            visited.discard(gene)
            return visited
        
        egfr_2hop = get_downstream('EGFR', depth=2)
        assert 'KRAS' in egfr_2hop
        assert 'BRAF' in egfr_2hop
    
    def test_upstream_regulators(self):
        """Test finding upstream regulators"""
        network_df = self.create_mock_network()
        
        # STAT3 upstream
        stat3_upstream = set(network_df[network_df['target'] == 'STAT3']['source'])
        
        assert 'JAK1' in stat3_upstream
        assert 'SRC' in stat3_upstream
    
    def test_path_exists(self):
        """Test checking if path exists between genes"""
        network_df = self.create_mock_network()
        
        # Build simple path check
        def path_exists(source: str, target: str, max_depth: int = 4) -> bool:
            if source == target:
                return True
            
            visited = {source}
            frontier = {source}
            
            for _ in range(max_depth):
                new_frontier = set()
                for g in frontier:
                    targets = network_df[network_df['source'] == g]['target'].tolist()
                    if target in targets:
                        return True
                    new_frontier.update(targets)
                new_frontier -= visited
                visited.update(new_frontier)
                frontier = new_frontier
            
            return False
        
        assert path_exists('EGFR', 'MAPK1')  # EGFR -> KRAS -> BRAF -> MAP2K1 -> MAPK1
        assert path_exists('JAK1', 'MYC')    # JAK1 -> STAT3 -> MYC
        assert not path_exists('MAPK1', 'EGFR')  # No reverse path


class TestViabilityPathInference:
    """Tests for viability path inference"""
    
    def test_path_confidence_calculation(self):
        """Test path confidence based on dependency scores"""
        # Simulate path confidence calculation
        # conf(p) = max(0, min(1, 0.5 - mean_dependency))
        
        # Highly essential path (mean dep = -0.8)
        dep_scores_essential = [-0.8, -0.9, -0.7]
        mean_dep = np.mean(dep_scores_essential)
        conf_essential = max(0, min(1, 0.5 - mean_dep))
        
        assert conf_essential > 0.5  # High confidence
        
        # Weakly essential path (mean dep = -0.2)
        dep_scores_weak = [-0.2, -0.1, -0.3]
        mean_dep_weak = np.mean(dep_scores_weak)
        conf_weak = max(0, min(1, 0.5 - mean_dep_weak))
        
        assert conf_weak < conf_essential  # Lower confidence
    
    def test_selectivity_filtering(self):
        """Test selectivity threshold for gene inclusion"""
        np.random.seed(42)
        
        # Create mock data: gene essential in varying fractions of lines
        n_lines = 10
        genes = ['KRAS', 'BRAF', 'RARE_GENE']
        
        # KRAS essential in 5/10 lines (50%)
        kras_essential = [True, True, True, True, True, False, False, False, False, False]
        
        # BRAF essential in 2/10 lines (20%)
        braf_essential = [True, True, False, False, False, False, False, False, False, False]
        
        # RARE_GENE essential in 1/10 lines (10%)
        rare_essential = [True, False, False, False, False, False, False, False, False, False]
        
        # Selectivity threshold = 30%
        theta = 0.3
        
        def is_selective(essential_counts: list) -> bool:
            return sum(essential_counts) / len(essential_counts) >= theta
        
        assert is_selective(kras_essential)  # 50% >= 30%
        assert not is_selective(braf_essential)  # 20% < 30%
        assert not is_selective(rare_essential)  # 10% < 30%


class TestDataIntegrity:
    """Tests for data integrity checks"""
    
    def test_missing_values_handling(self):
        """Test handling of missing values in dependency data"""
        # Create data with NaN
        df = pd.DataFrame({
            'KRAS': [np.nan, -0.5, -0.6],
            'BRAF': [-0.4, -0.5, np.nan],
        }, index=['ACH-001', 'ACH-002', 'ACH-003'])
        
        # Compute mean ignoring NaN
        kras_mean = df['KRAS'].mean()  # Should be mean of -0.5, -0.6
        
        assert np.isclose(kras_mean, -0.55)
    
    def test_duplicate_gene_handling(self):
        """Test handling of duplicate gene names"""
        # Simulate duplicate columns
        df = pd.DataFrame({
            'KRAS (123)': [1, 2, 3],
            'KRAS (456)': [4, 5, 6],  # Duplicate gene, different Entrez
        })
        
        # Parse and deduplicate
        def parse_gene(col: str) -> str:
            import re
            match = re.match(r'^([A-Z0-9\-]+)\s*\(\d+\)$', col)
            return match.group(1) if match else col
        
        gene_names = [parse_gene(c) for c in df.columns]
        
        # Should have duplicates
        assert gene_names[0] == gene_names[1] == 'KRAS'
        
        # Deduplicate by keeping first
        seen = set()
        unique_cols = []
        for col in df.columns:
            gene = parse_gene(col)
            if gene not in seen:
                seen.add(gene)
                unique_cols.append(col)
        
        assert len(unique_cols) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
