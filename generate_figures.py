#!/usr/bin/env python3
"""
Generate publication figures for ALIN Framework (Adaptive Lethal Intersection Network)
"""

import pandas as pd
from pathlib import Path
import json

def main():
    base = Path(__file__).parent
    fig_dir = base / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib seaborn")
        return
    
    # Figure 1: Benchmark comparison (our method vs baselines)
    metrics_file = base / "benchmark_results" / "benchmark_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        methods = ['ALIN\n(ours)', 'Random\nbaseline', 'Top-genes\nbaseline']
        recalls = [
            metrics.get('recall_any', 0),
            metrics.get('random_baseline_mean', 0),
            metrics.get('topgenes_baseline', 0)
        ]
        errs = [0, metrics.get('random_baseline_std', 0), 0]
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        bars = ax.bar(methods, [r*100 for r in recalls], yerr=[e*100 for e in errs], 
                     color=colors, capsize=5, edgecolor='black', linewidth=1.2)
        ax.set_ylabel('Recall (%)', fontsize=12)
        ax.set_title('Benchmark: Recovery of Known Drug Combinations', fontsize=14)
        ax.set_ylim(0, 80)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / "benchmark_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {fig_dir}/benchmark_comparison.png")
    
    # Figure 2: Target frequency in triples
    triples_file = base / "results_triples" / "triple_target_frequency.csv"
    if triples_file.exists():
        df = pd.read_csv(triples_file)
        count_col = 'Appearances_in_Triples' if 'Appearances_in_Triples' in df.columns else 'Count'
        target_col = 'Target_Gene' if 'Target_Gene' in df.columns else 'Target'
        if target_col in df.columns and count_col in df.columns:
            top = df.nlargest(12, count_col)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top[target_col].iloc[::-1], top[count_col].iloc[::-1], color='#3498db', edgecolor='black')
            ax.set_xlabel('Frequency in top triple combinations', fontsize=12)
            ax.set_title('Most Frequently Predicted Targets (Pan-Cancer)', fontsize=14)
            plt.tight_layout()
            plt.savefig(fig_dir / "target_frequency.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {fig_dir}/target_frequency.png")
    
    # Figure 3: Cancer type coverage (number of cancers per triple pattern)
    triples_file = base / "results_triples" / "triple_combinations.csv"
    if triples_file.exists():
        df = pd.read_csv(triples_file)
        if 'Triple_Targets' in df.columns:
            pattern_counts = df['Triple_Targets'].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(range(len(pattern_counts)), pattern_counts.values, color='#9b59b6', edgecolor='black')
            ax.set_yticks(range(len(pattern_counts)))
            ax.set_yticklabels(pattern_counts.index, fontsize=9)
            ax.set_xlabel('Number of cancer types', fontsize=12)
            ax.set_title('Top Triple Combination Patterns (Pan-Cancer)', fontsize=14)
            plt.tight_layout()
            plt.savefig(fig_dir / "triple_patterns.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {fig_dir}/triple_patterns.png")
    
    # Figure 4: Synergy vs resistance (scatter)
    if triples_file.exists():
        df = pd.read_csv(triples_file)
        if 'Synergy_Score' in df.columns and 'Resistance_Score' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(df['Synergy_Score'], df['Resistance_Score'], alpha=0.6, s=30, c='#e67e22')
            ax.set_xlabel('Synergy Score', fontsize=12)
            ax.set_ylabel('Resistance Score (lower = better)', fontsize=12)
            ax.set_title('Triple Combinations: Synergy vs Resistance', fontsize=14)
            ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='High synergy')
            ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Low resistance')
            plt.tight_layout()
            plt.savefig(fig_dir / "synergy_resistance.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {fig_dir}/synergy_resistance.png")
    
    print(f"\nFigures saved to {fig_dir}/")

if __name__ == "__main__":
    main()
