#!/usr/bin/env python3
"""
Generate updated publication-quality figures for ALIN Framework.
Reflects expanded 43-entry gold standard, hub-gene penalty, and new benchmark numbers.

Data sources:
  - results_triples_expanded/triple_combinations.csv (17 cancers, hub-penalty predictions)
  - results_triples_expanded/triple_combinations_merged.csv (77 cancers, merged)
  - Hardcoded benchmark numbers from expanded_gold_standard.py run

Reference: Liaki V, Barrambana S, et al. 2025. bioRxiv doi: 10.1101/2025.08.04.668325.
"""

import pandas as pd
from pathlib import Path
import json
import shutil
from collections import Counter

# Benchmark results from latest expanded gold standard run
BENCHMARK = {
    "alin_any_overlap": 44.2,
    "alin_pair_overlap": 30.2,
    "alin_superset": 9.3,
    "alin_exact": 7.0,
    "global_freq_any_overlap": 48.8,
    "global_freq_pair_overlap": 27.9,
    "global_freq_superset": 0.0,
    "global_freq_exact": 0.0,
    "driver_any_overlap": 44.2,
    "driver_pair_overlap": 16.3,
    "driver_superset": 14.0,
    "driver_exact": 0.0,
    "random_any_overlap": 22.0,
    "random_pair_overlap": 8.8,
    "random_superset": 1.1,
    "random_exact": 0.0,
    "gold_standard_size": 43,
    "gold_standard_cancers": 25,
    "alin_any_overlap_matches": 19,
    "alin_pair_overlap_matches": 13,
    "alin_superset_matches": 4,
    "alin_exact_matches": 3,
    # Testable-only (25/43 entries where cancer+target modality are DepMap-compatible)
    "n_testable": 25,
    "n_untestable": 18,
    "alin_testable_any_overlap": 64.0,
    "alin_testable_pair_overlap": 44.0,
    "driver_testable_any_overlap": 60.0,
    "driver_testable_pair_overlap": 20.0,
    "global_freq_testable_any_overlap": 76.0,
    "global_freq_testable_pair_overlap": 40.0,
    "random_testable_any_overlap": 32.2,
    "random_testable_pair_overlap": 12.1,
    # Cancer-level precision
    "alin_precision": 47.1,
    "alin_precision_testable": 58.3,
    "alin_precision_evaluable": 17,
    "alin_precision_hits": 8,
    "alin_precision_evaluable_testable": 12,
    "alin_precision_hits_testable": 7,
    "global_freq_precision": 58.8,
    "global_freq_precision_testable": 83.3,
    "driver_precision": 100.0,
    "driver_precision_testable": 100.0,
    "random_precision": 34.1,
    "random_precision_testable": 43.3,
    # Candidate-pool random baseline (per-cancer selective-essential pool)
    "cpool_random_any_overlap": 0.2,
    "cpool_random_pair_overlap": 0.0,
    "cpool_random_superset": 0.0,
    "cpool_random_exact": 0.0,
    "cpool_random_testable_any_overlap": 0.3,
    "cpool_random_testable_pair_overlap": 0.0,
    "cpool_random_precision": 0.3,
    "cpool_random_precision_testable": 0.4,
    # KNOWN_SYNERGIES ablation (all metrics identical to with-KS)
    "noks_any_overlap": 44.2,
    "noks_pair_overlap": 30.2,
    "noks_superset": 9.3,
    "noks_exact": 7.0,
    "noks_precision": 47.1,
    "noks_precision_testable": 58.3,
    "noks_testable_any_overlap": 64.0,
    "noks_delta_any_overlap": 0.0,  # no change
}

# Publication-quality color palette (distinct, colorblind-friendly)
PALETTE = {
    "primary": "#2E86AB",    # Deep blue
    "secondary": "#A23B72",   # Magenta
    "accent": "#F18F01",     # Amber
    "success": "#3B7A57",    # Forest green
    "neutral": "#5C5C5C",    # Charcoal
    "light": "#E8E8E8",
    "gradient": ["#2E86AB", "#A23B72", "#F18F01", "#3B7A57", "#5C5C5C"],
}


def _setup_style():
    """Apply publication-quality style. Fallback to seaborn if SciencePlots unavailable."""
    try:
        import scienceplots
        import matplotlib.pyplot as plt
        plt.style.use(["science", "nature", "no-latex"])
        return plt
    except (ImportError, OSError):
        pass
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="husl", font="sans-serif",
                      rc={"axes.facecolor": "#FAFAFA", "figure.facecolor": "white",
                          "axes.edgecolor": ".15", "grid.color": ".9",
                          "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
        return plt
    except ImportError:
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.family": "sans-serif", "font.size": 11,
            "axes.titlesize": 13, "axes.labelsize": 11,
            "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
        })
        return plt


def _save_fig(plt, path: Path, dpi=300):
    """Save figure as both PNG and PDF with tight layout."""
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    pdf_path = path.with_suffix(".pdf")
    plt.savefig(str(pdf_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path.name} + {pdf_path.name}")


def fig3_benchmark(fig_dir: Path):
    """Figure 3: Benchmark comparison — 2-panel grouped bar chart."""
    plt = _setup_style()
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]})

    # --- Panel A: Grouped bar comparison across methods ---
    methods = ["ALIN", "Driver\ngenes", "Global\nfrequency", "Random\nglobal", "Random\ncandidate"]
    any_ov = [
        BENCHMARK["alin_any_overlap"],
        BENCHMARK["driver_any_overlap"],
        BENCHMARK["global_freq_any_overlap"],
        BENCHMARK["random_any_overlap"],
        BENCHMARK["cpool_random_any_overlap"],
    ]
    pair_ov = [
        BENCHMARK["alin_pair_overlap"],
        BENCHMARK["driver_pair_overlap"],
        BENCHMARK["global_freq_pair_overlap"],
        BENCHMARK["random_pair_overlap"],
        BENCHMARK["cpool_random_pair_overlap"],
    ]

    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax1.bar(x - w/2, any_ov, w, label="Any-overlap", color=PALETTE["primary"], edgecolor="white", linewidth=1.2)
    bars2 = ax1.bar(x + w/2, pair_ov, w, label="Pair-overlap", color=PALETTE["success"], edgecolor="white", linewidth=1.2)

    # Value labels
    for bar, val in zip(bars1, any_ov):
        if val > 2:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=PALETTE["primary"])
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color="gray")
    for bar, val in zip(bars2, pair_ov):
        if val > 2:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=PALETTE["success"])
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color="gray")

    # p-value annotation next to ALIN bar (no line)
    ax1.text(0, any_ov[0] + 5, "$p < 0.001$\nvs. random", ha="center", fontsize=8,
             fontstyle="italic", color="#555", linespacing=1.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylabel("Recall (%) — n = 43 gold-standard entries", fontsize=11)
    ax1.set_ylim(0, 60)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("A  Baseline comparison", fontsize=12, fontweight="bold", loc="left")

    # --- Panel B: ALIN recall breakdown (stacked horizontal) ---
    categories = ["Any-overlap\n(|G∩T|≥1)", "Pair-overlap\n(|G∩T|≥2)", "Superset\n(G⊆T)", "Exact\n(G=T)"]
    values = [
        BENCHMARK["alin_any_overlap"],
        BENCHMARK["alin_pair_overlap"],
        BENCHMARK["alin_superset"],
        BENCHMARK["alin_exact"],
    ]
    counts = [
        BENCHMARK["alin_any_overlap_matches"],
        BENCHMARK["alin_pair_overlap_matches"],
        BENCHMARK["alin_superset_matches"],
        BENCHMARK["alin_exact_matches"],
    ]
    bar_colors = ["#2E86AB", "#3B7A57", "#F18F01", "#A23B72"]

    y = np.arange(len(categories))
    hbars = ax2.barh(y, values, color=bar_colors, edgecolor="white", linewidth=1.2, height=0.6)

    for bar, val, cnt in zip(hbars, values, counts):
        ax2.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}% ({int(cnt)}/43)", va="center", fontsize=10, fontweight="bold")

    ax2.set_yticks(y)
    ax2.set_yticklabels(categories, fontsize=10)
    ax2.set_xlabel("Recall (%)", fontsize=11)
    ax2.set_xlim(0, 65)
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("B  ALIN recall by stringency", fontsize=12, fontweight="bold", loc="left")

    fig.suptitle(f"Benchmark: Recovery of validated drug combinations\n"
                 f"(n = {BENCHMARK['gold_standard_size']} gold-standard entries, "
                 f"{BENCHMARK['gold_standard_cancers']} cancer types)",
                 fontsize=13, fontweight="bold", y=1.02)

    _save_fig(plt, fig_dir / "fig3_benchmark_comparison.png")


def fig4_triple_patterns(base: Path, fig_dir: Path):
    """Figure 4: Pan-cancer triple patterns from merged results (77 cancers)."""
    plt = _setup_style()

    # Use merged file (expanded + original = 77 cancers)
    triples_file = base / "results_triples_expanded" / "triple_combinations_merged.csv"
    if not triples_file.exists():
        triples_file = base / "results_triples_expanded" / "triple_combinations.csv"
    if not triples_file.exists():
        print("  Skipping fig4: no expanded triple data found")
        return

    df = pd.read_csv(triples_file)

    # Determine triple column name
    if "Triple_Targets" in df.columns:
        triple_col = "Triple_Targets"
    elif all(c in df.columns for c in ["Target_1", "Target_2", "Target_3"]):
        # Build triple string from individual targets
        df["Triple_Targets"] = df.apply(
            lambda r: " + ".join(sorted([str(r["Target_1"]), str(r["Target_2"]), str(r["Target_3"])])),
            axis=1)
        triple_col = "Triple_Targets"
    elif "Target 1" in df.columns:
        df["Triple_Targets"] = df.apply(
            lambda r: " + ".join(sorted([str(r["Target 1"]), str(r["Target 2"]), str(r["Target 3"])])),
            axis=1)
        triple_col = "Triple_Targets"
    else:
        print("  Skipping fig4: no target columns found")
        return

    pattern_counts = df[triple_col].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [PALETTE["primary"] if i == 0 else PALETTE["secondary"] if i == 1
              else PALETTE["accent"] for i in range(len(pattern_counts))]
    bars = ax.barh(range(len(pattern_counts)), pattern_counts.values, color=colors,
                   edgecolor="white", linewidth=1.5, height=0.7)
    ax.set_yticks(range(len(pattern_counts)))
    ax.set_yticklabels(pattern_counts.index, fontsize=9, fontweight="medium")
    ax.set_xlabel("Number of cancer types", fontsize=12, fontweight="medium")
    ax.set_title("Top Triple Combination Patterns (Pan-Cancer)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    _save_fig(plt, fig_dir / "fig4_triple_patterns.png")


def fig5_target_frequency(base: Path, fig_dir: Path):
    """Figure 5: Most frequently predicted targets across expanded predictions."""
    plt = _setup_style()

    # Use merged file for full picture
    triples_file = base / "results_triples_expanded" / "triple_combinations_merged.csv"
    if not triples_file.exists():
        triples_file = base / "results_triples_expanded" / "triple_combinations.csv"
    if not triples_file.exists():
        print("  Skipping fig5: no expanded triple data found")
        return

    df = pd.read_csv(triples_file)

    # Collect all target genes
    targets = []
    for col in ["Target_1", "Target_2", "Target_3", "Target 1", "Target 2", "Target 3"]:
        if col in df.columns:
            targets.extend(df[col].dropna().tolist())

    if not targets:
        print("  Skipping fig5: no target columns found")
        return

    counts = Counter(targets)
    top = pd.DataFrame([{"Gene": k, "Count": v} for k, v in counts.most_common(15)])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors_arr = plt.cm.Blues(0.4 + 0.5 * (top["Count"].values / top["Count"].max()))
    ax.barh(top["Gene"].iloc[::-1], top["Count"].iloc[::-1], color=colors_arr[::-1],
            edgecolor="white", linewidth=1.5, height=0.7)
    ax.set_xlabel("Frequency in top triple combinations", fontsize=12, fontweight="medium")
    ax.set_title("Most Frequently Predicted Targets (Pan-Cancer, n=77)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(plt, fig_dir / "fig5_target_frequency.png")


def fig6_synergy_resistance(base: Path, fig_dir: Path):
    """Figure 6: Synergy vs resistance landscape from expanded triples."""
    plt = _setup_style()

    # First try merged, then expanded, then original
    for path in [
        base / "results_triples_expanded" / "triple_combinations_merged.csv",
        base / "results_triples_expanded" / "triple_combinations.csv",
        base / "results_triples" / "triple_combinations.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            syn_col = "Synergy_Score" if "Synergy_Score" in df.columns else "Synergy"
            res_col = "Resistance_Score" if "Resistance_Score" in df.columns else "Resistance"
            if syn_col in df.columns and res_col in df.columns:
                break
    else:
        print("  Skipping fig6: no synergy/resistance data found")
        return

    # Convert percentage strings if needed
    for col in [syn_col, res_col]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce') / 100.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(df[syn_col], df[res_col], alpha=0.7, s=60, c=PALETTE["accent"],
               edgecolors="white", linewidths=0.5)
    ax.axvline(x=0.9, color=PALETTE["success"], linestyle="--", alpha=0.7,
               linewidth=1.5, label="High synergy")
    ax.axhline(y=0.4, color="#C73E1D", linestyle="--", alpha=0.7,
               linewidth=1.5, label="Low resistance")
    ax.set_xlabel("Synergy Score", fontsize=12, fontweight="medium")
    ax.set_ylabel("Resistance Score (lower = better)", fontsize=12, fontweight="medium")
    ax.set_title("Triple Combinations: Synergy vs Resistance Landscape",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, fancybox=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(plt, fig_dir / "fig6_synergy_resistance.png")


def figS4_benchmark_detail(fig_dir: Path):
    """Figure S4: Benchmark detail — stacked bar showing match types + baseline comparison."""
    plt = _setup_style()

    # Panel A: Match type breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: match breakdown
    categories = ["Any-overlap\nonly", "Pair-overlap", "Superset", "Exact", "Unmatched"]
    n_any_only = BENCHMARK["alin_any_overlap_matches"] - BENCHMARK["alin_pair_overlap_matches"]
    n_pair_only = BENCHMARK["alin_pair_overlap_matches"] - BENCHMARK["alin_superset_matches"]
    values = [
        n_any_only,  # any-overlap only (|G∩T|=1)
        n_pair_only,  # pair-overlap (|G∩T|≥2 but not superset)
        BENCHMARK["alin_superset_matches"],  # superset
        0,  # exact
        BENCHMARK["gold_standard_size"] - BENCHMARK["alin_any_overlap_matches"],  # unmatched
    ]
    colors = [PALETTE["neutral"], PALETTE["primary"], PALETTE["success"], PALETTE["accent"], PALETTE["light"]]

    ax1.bar(categories, values, color=colors, edgecolor="white", linewidth=2, width=0.55)
    for i, v in enumerate(values):
        if v > 0:
            ax1.text(i, v + 0.3, str(v), ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of gold-standard entries", fontsize=11)
    ax1.set_title(f"Match Breakdown (n={BENCHMARK['gold_standard_size']})",
                  fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right panel: method comparison
    methods = ["ALIN", "Driver\ngenes", "Random", "Global\nfreq."]
    any_overlap = [
        BENCHMARK["alin_any_overlap"],
        BENCHMARK["driver_any_overlap"],
        BENCHMARK["random_any_overlap"],
        BENCHMARK["global_freq_any_overlap"],
    ]
    bar_colors = [PALETTE["success"], PALETTE["accent"], PALETTE["neutral"], PALETTE["secondary"]]

    ax2.bar(methods, any_overlap, color=bar_colors, edgecolor="white", linewidth=2, width=0.6)
    for i, v in enumerate(any_overlap):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Any-Overlap Recall (%) — |G∩T|≥1", fontsize=11)
    ax2.set_title("Method Comparison", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 55)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("Benchmark Performance Against 43 Gold-Standard Combinations",
                 fontsize=13, fontweight="bold", y=1.02)
    _save_fig(plt, fig_dir / "figS4_benchmark_rank.png")


def figS5_novel_combinations(base: Path, fig_dir: Path):
    """Figure S5: Top novel combinations by therapeutic potential."""
    plt = _setup_style()

    triples_file = base / "results_triples_expanded" / "triple_combinations_merged.csv"
    if not triples_file.exists():
        triples_file = base / "results_triples_expanded" / "triple_combinations.csv"
    if not triples_file.exists():
        print("  Skipping figS5: no expanded triple data found")
        return

    df = pd.read_csv(triples_file)
    syn_col = "Synergy_Score" if "Synergy_Score" in df.columns else "Synergy"
    res_col = "Resistance_Score" if "Resistance_Score" in df.columns else "Resistance"
    cancer_col = "Cancer_Type" if "Cancer_Type" in df.columns else "Cancer Type"
    triple_col = "Triple_Targets" if "Triple_Targets" in df.columns else None

    if syn_col not in df.columns or res_col not in df.columns:
        print("  Skipping figS5: missing synergy/resistance data")
        return

    # Convert percentage strings if needed
    for col in [syn_col, res_col]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce') / 100.0

    df["potential"] = df[syn_col] - df[res_col]
    top5 = df.nlargest(5, "potential")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    rows = [["Cancer Type", "Triple", "Synergy", "Resistance", "Drugs"]]
    for _, r in top5.iterrows():
        cancer = str(r.get(cancer_col, ""))[:28]
        if triple_col and triple_col in r:
            triple = str(r[triple_col])[:28]
        else:
            t1 = r.get("Target_1", r.get("Target 1", ""))
            t2 = r.get("Target_2", r.get("Target 2", ""))
            t3 = r.get("Target_3", r.get("Target 3", ""))
            triple = f"{t1}+{t2}+{t3}"[:28]

        d1 = r.get("Drug_1", r.get("Drug 1", ""))
        d2 = r.get("Drug_2", r.get("Drug 2", ""))
        d3 = r.get("Drug_3", r.get("Drug 3", ""))
        drugs = f"{d1}+{d2}+{d3}"[:45]

        rows.append([
            cancer,
            triple,
            f"{r[syn_col]:.2f}",
            f"{r[res_col]:.2f}",
            drugs,
        ])

    table = ax.table(cellText=rows, loc="center", cellLoc="center",
                     colWidths=[0.22, 0.22, 0.12, 0.12, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)
    for (i, j), cell in table.get_celld().items():
        cell.set_facecolor("#F5F5F5" if i == 0 else "white")
        cell.set_text_props(fontweight="bold" if i == 0 else "normal")
        cell.set_edgecolor(PALETTE["light"])
    ax.set_title("Top 5 Combinations by Therapeutic Potential (Synergy − Resistance)",
                 fontsize=12, fontweight="bold", pad=20)
    _save_fig(plt, fig_dir / "figS5_novel_combinations.png")


def update_benchmark_metrics(base: Path):
    """Update benchmark_results/benchmark_metrics.json with new numbers."""
    metrics = {
        "total_gold_standard": BENCHMARK["gold_standard_size"],
        "gold_standard_cancers": BENCHMARK["gold_standard_cancers"],
        "exact_matches": 0,
        "superset_matches": BENCHMARK["alin_superset_matches"],
        "pair_overlap_matches": BENCHMARK["alin_pair_overlap_matches"] - BENCHMARK["alin_superset_matches"],
        "any_overlap_matches": BENCHMARK["alin_any_overlap_matches"] - BENCHMARK["alin_pair_overlap_matches"],
        "no_match": BENCHMARK["gold_standard_size"] - BENCHMARK["alin_any_overlap_matches"],
        "recall_exact": BENCHMARK["alin_exact"] / 100,
        "recall_superset_or_better": BENCHMARK["alin_superset"] / 100,
        "recall_pair_overlap_or_better": BENCHMARK["alin_pair_overlap"] / 100,
        "recall_any_overlap_or_better": BENCHMARK["alin_any_overlap"] / 100,
        "global_frequency_baseline_any_overlap": BENCHMARK["global_freq_any_overlap"] / 100,
        "global_frequency_baseline_pair_overlap": BENCHMARK["global_freq_pair_overlap"] / 100,
        "driver_baseline_any_overlap": BENCHMARK["driver_any_overlap"] / 100,
        "driver_baseline_pair_overlap": BENCHMARK["driver_pair_overlap"] / 100,
        "random_baseline_any_overlap": BENCHMARK["random_any_overlap"] / 100,
        "random_baseline_pair_overlap": BENCHMARK["random_pair_overlap"] / 100,
        "hub_penalty_applied": True,
        "hub_penalty_effect": "STAT3 removed from 17/17 benchmarked cancer predictions",
        "alin_vs_global_freq_ratio_any_overlap": round(BENCHMARK["alin_any_overlap"] / BENCHMARK["global_freq_any_overlap"], 1),
        "alin_vs_global_freq_ratio_pair_overlap": round(BENCHMARK["alin_pair_overlap"] / BENCHMARK["global_freq_pair_overlap"], 1),
        # Testable-only metrics
        "n_testable": BENCHMARK["n_testable"],
        "n_untestable": BENCHMARK["n_untestable"],
        "testable_recall_any_overlap": BENCHMARK["alin_testable_any_overlap"] / 100,
        "testable_recall_pair_overlap": BENCHMARK["alin_testable_pair_overlap"] / 100,
        "testable_driver_any_overlap": BENCHMARK["driver_testable_any_overlap"] / 100,
        "testable_driver_pair_overlap": BENCHMARK["driver_testable_pair_overlap"] / 100,
        "testable_global_freq_any_overlap": BENCHMARK["global_freq_testable_any_overlap"] / 100,
        "testable_random_any_overlap": BENCHMARK["random_testable_any_overlap"] / 100,
        # Cancer-level precision
        "precision_alin": BENCHMARK["alin_precision"] / 100,
        "precision_alin_evaluable": BENCHMARK["alin_precision_evaluable"],
        "precision_alin_hits": BENCHMARK["alin_precision_hits"],
        "precision_alin_testable": BENCHMARK["alin_precision_testable"] / 100,
        "precision_global_freq": BENCHMARK["global_freq_precision"] / 100,
        "precision_driver": BENCHMARK["driver_precision"] / 100,
        "precision_random": BENCHMARK["random_precision"] / 100,
    }

    out = base / "benchmark_results" / "benchmark_metrics.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Updated: {out}")


def main():
    base = Path(__file__).parent
    fig_dir = base / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("Generating updated figures for ALIN Framework...")
    print(f"  Gold standard: {BENCHMARK['gold_standard_size']} entries, "
          f"{BENCHMARK['gold_standard_cancers']} cancer types")
    print(f"  ALIN any-overlap: {BENCHMARK['alin_any_overlap']}% | "
          f"pair-overlap: {BENCHMARK['alin_pair_overlap']}%")
    print(f"  Driver: {BENCHMARK['driver_any_overlap']}% / {BENCHMARK['driver_pair_overlap']}% | "
          f"Global freq: {BENCHMARK['global_freq_any_overlap']}% / {BENCHMARK['global_freq_pair_overlap']}% | "
          f"Random: {BENCHMARK['random_any_overlap']}% / {BENCHMARK['random_pair_overlap']}%")
    print()

    # Data-driven figures (need updating)
    print("[Fig 3] Benchmark comparison...")
    fig3_benchmark(fig_dir)

    print("[Fig 4] Triple patterns...")
    fig4_triple_patterns(base, fig_dir)

    print("[Fig 5] Target frequency...")
    fig5_target_frequency(base, fig_dir)

    print("[Fig 6] Synergy vs resistance...")
    fig6_synergy_resistance(base, fig_dir)

    print("[Fig S4] Benchmark detail...")
    figS4_benchmark_detail(fig_dir)

    print("[Fig S5] Novel combinations...")
    figS5_novel_combinations(base, fig_dir)

    # Update benchmark metrics JSON
    print("\n[Metrics] Updating benchmark_metrics.json...")
    update_benchmark_metrics(base)

    # Legacy filenames
    for src, dst in [
        ("fig3_benchmark_comparison.png", "benchmark_comparison.png"),
        ("fig4_triple_patterns.png", "triple_patterns.png"),
        ("fig5_target_frequency.png", "target_frequency.png"),
        ("fig6_synergy_resistance.png", "synergy_resistance.png"),
    ]:
        s, d = fig_dir / src, fig_dir / dst
        if s.exists():
            shutil.copy(s, d)
            shutil.copy(s.with_suffix(".pdf"), d.replace(".png", ".pdf") if isinstance(d, str) else d.with_suffix(".pdf"))

    print(f"\nAll figures saved to {fig_dir}/")
    print("Static figures (fig1, fig2, figS1, figS2, figS3) unchanged — they are methodological schematics.")


if __name__ == "__main__":
    main()
