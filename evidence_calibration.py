#!/usr/bin/env python3
"""
Evidence Calibration & De-correlation Analysis
===============================================

Addresses the concern that the 8-component combined_score in
TripleCombinationFinder sums correlated "evidence" with hand-tuned
weights.  This script:

  1. Collects ALL scored triple/doublet combinations across cancers
     with their raw features.
  2. Labels each combination as positive (overlaps gold-standard) or
     negative.
  3. Reports pairwise feature correlations and flags |r| > 0.7 pairs.
  4. Fits a calibrated logistic-regression model predicting "known
     combo" from features (with isotonic calibration).
  5. Computes per-feature SHAP values and leave-one-feature-out
     ablation ΔAUROCs.
  6. Generates a publication-quality 4-panel figure.

Usage:
    python evidence_calibration.py
"""

import os, sys, json, warnings, logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────

OUTDIR = Path("calibration_results")
OUTDIR.mkdir(exist_ok=True)

# Feature names as they appear in the scoring closure
FEATURE_NAMES = [
    'total_cost',
    'synergy_score',
    'resistance_score',
    'coverage',
    'combo_tox_score',
    'hub_penalty',
    'druggable_count',
    'perturbation_bonus',
]

# Pretty display names
FEATURE_LABELS = {
    'total_cost': 'Cost',
    'synergy_score': 'Synergy',
    'resistance_score': 'Resistance',
    'coverage': 'Coverage',
    'combo_tox_score': 'Combo Tox',
    'hub_penalty': 'Hub Penalty',
    'druggable_count': 'Druggable #',
    'perturbation_bonus': 'Perturbation',
}

# ── Step 1: collect all scored combos ────────────────────────────────

def collect_all_scored_combos() -> pd.DataFrame:
    """
    Re-run the analysis pipeline for each cancer type and extract ALL
    scored TripleCombination objects with their raw features.

    We call analyze_cancer_type() to build paths and internal state,
    then re-call triple_finder.find_triple_combinations() with a huge
    top_n to capture the full ranking (not just top-10).
    """
    sys.path.insert(0, '.')
    from pan_cancer_xnode import PanCancerXNodeAnalyzer

    analyzer = PanCancerXNodeAnalyzer()

    # Import gold standard for labelling
    from gold_standard import GOLD_STANDARD, check_match, CANCER_ALIASES

    # Build cancer→gold lookup
    cancer_gold = defaultdict(list)
    for entry in GOLD_STANDARD:
        cancer_gold[entry['cancer']].append(entry['targets'])

    # Resolve pipeline cancer names → gold-standard cancer names
    # (reverse CANCER_ALIASES mapping)
    pipeline_to_gold = {}
    for gold_name, aliases in CANCER_ALIASES.items():
        for alias in aliases:
            pipeline_to_gold[alias] = gold_name
        pipeline_to_gold[gold_name] = gold_name

    rows = []
    cancer_type_tuples = analyzer.depmap.get_available_cancer_types()
    cancer_types = [ct for ct, count in cancer_type_tuples if count >= 5]
    logger.info(f"Processing {len(cancer_types)} cancer types")

    for cancer in sorted(cancer_types):
        logger.info(f"  {cancer}...")
        try:
            result = analyzer.analyze_cancer_type(cancer)
        except Exception as e:
            logger.warning(f"  SKIP {cancer}: {e}")
            continue

        if result is None or not result.viability_paths:
            continue

        # Re-run triple_finder with huge top_n to get ALL scored combos
        # (analyze_cancer_type only keeps top-10)
        all_triples = analyzer.triple_finder.find_triple_combinations(
            result.viability_paths, result.cancer_type,
            top_n=100000, min_coverage=0.3  # lower min_cov for more negatives
        )
        # Also get doublets
        all_doublets = getattr(analyzer.triple_finder,
                               '_last_doublet_combinations', [])
        all_combos = list(all_triples) + list(all_doublets)

        if not all_combos:
            # Fallback to whatever was stored
            all_combos = list(result.triple_combinations or [])

        # Determine gold-standard targets for this cancer
        gold_name = pipeline_to_gold.get(cancer, cancer)
        gold_target_sets = cancer_gold.get(gold_name, [])

        for combo in all_combos:
            pred_targets = frozenset(combo.targets)
            # Label: any overlap with ANY gold entry for this cancer
            label = 0
            best_match = 'none'
            for gs_targets in gold_target_sets:
                mt = check_match(pred_targets, gs_targets)
                match_rank = {'exact': 4, 'superset': 3, 'pair_overlap': 2,
                              'any_overlap': 1, 'none': 0}
                if match_rank[mt] > match_rank[best_match]:
                    best_match = mt
            if best_match in ('exact', 'superset', 'pair_overlap', 'any_overlap'):
                label = 1

            rows.append({
                'cancer_type': cancer,
                'targets': '+'.join(sorted(combo.targets)),
                'n_targets': len(combo.targets),
                'total_cost': combo.total_cost,
                'synergy_score': combo.synergy_score,
                'resistance_score': combo.resistance_score,
                'coverage': combo.coverage,
                'combo_tox_score': combo.combo_tox_score,
                'druggable_count': combo.druggable_count,
                'combined_score': combo.combined_score,
                'label': label,
                'match_type': best_match,
            })

        logger.info(f"    {len(all_combos)} combos, "
                     f"{sum(1 for r in rows[-len(all_combos):] if r['label']==1)} positive")

    df = pd.DataFrame(rows)
    logger.info(f"Total: {len(df)} combos, {df['label'].sum()} positives "
                f"across {df['cancer_type'].nunique()} cancers")
    return df


def collect_combos_with_hub_penalty(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    If hub_penalty and perturbation_bonus aren't in the data (they aren't
    stored in TripleCombination), we reconstruct them from the combined_score
    formula:

    combined = cost*0.22 + (1-synergy)*0.18 + resistance*0.18
             + (1-coverage)*0.14 + tox*0.18 + hub - druggable*0.1 - pert

    Solve for hub - pert:
      hub - pert = combined - cost*0.22 - (1-syn)*0.18 - res*0.18
                 - (1-cov)*0.14 - tox*0.18 + drug*0.1
    
    We can't separate hub and pert from just combined_score; so we 
    re-derive them from the pipeline internals. For the calibration model,
    we'll use the features we CAN extract.
    """
    # hub_penalty and perturbation_bonus are not stored in TripleCombination.
    # We compute the "residual" = combined_score - weighted known components
    # This residual = hub_penalty - perturbation_bonus
    known_part = (
        df_base['total_cost'] * 0.22 +
        (1 - df_base['synergy_score']) * 0.18 +
        df_base['resistance_score'] * 0.18 +
        (1 - df_base['coverage']) * 0.14 +
        df_base['combo_tox_score'] * 0.18 -
        df_base['druggable_count'] * 0.1
    )
    df_base['hub_minus_pert'] = df_base['combined_score'] - known_part
    # Approximate: perturbation_bonus is small (max 0.1*feedback_cov ~ 0.01-0.05)
    # hub_penalty can be larger. For correlation / model we use this combined residual.
    df_base['hub_penalty'] = df_base['hub_minus_pert'].clip(lower=0)
    df_base['perturbation_bonus'] = (-df_base['hub_minus_pert']).clip(lower=0)
    return df_base


# ── Step 2: feature correlations ─────────────────────────────────────

def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson and Spearman correlations among features."""
    features = [f for f in FEATURE_NAMES if f in df.columns]
    pearson = df[features].corr(method='pearson')
    spearman = df[features].corr(method='spearman')

    # Flag highly correlated pairs (|r| > 0.7)
    flagged = []
    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            if i < j:
                rp = pearson.loc[fi, fj]
                rs = spearman.loc[fi, fj]
                if abs(rp) > 0.7 or abs(rs) > 0.7:
                    flagged.append({
                        'feature_i': fi, 'feature_j': fj,
                        'pearson': round(rp, 3), 'spearman': round(rs, 3),
                        'action': 'drop_one' if abs(rp) > 0.7 else 'monitor',
                    })

    flagged_df = pd.DataFrame(flagged) if flagged else pd.DataFrame(
        columns=['feature_i', 'feature_j', 'pearson', 'spearman', 'action'])

    # Save
    pearson.to_csv(OUTDIR / 'pearson_correlation.csv')
    spearman.to_csv(OUTDIR / 'spearman_correlation.csv')
    flagged_df.to_csv(OUTDIR / 'flagged_correlated_pairs.csv', index=False)

    logger.info(f"Feature correlations: {len(flagged)} pairs with |r|>0.7")
    return pearson


# ── Step 3: de-correlated feature set ────────────────────────────────

def select_decorrelated_features(corr_matrix: pd.DataFrame,
                                 threshold: float = 0.7) -> list:
    """
    Greedy removal: for each |r|>threshold pair, drop the feature with
    lower absolute mean-correlation with the label (if known) or just
    the second one.
    """
    features = list(corr_matrix.columns)
    drop = set()
    for i, fi in enumerate(features):
        if fi in drop:
            continue
        for j, fj in enumerate(features):
            if j <= i or fj in drop:
                continue
            if abs(corr_matrix.loc[fi, fj]) > threshold:
                # Drop the one less informative — we'll keep the one with
                # higher variance (as a heuristic before model fitting)
                drop.add(fj)
                logger.info(f"  Dropping '{fj}' (corr with '{fi}' = "
                            f"{corr_matrix.loc[fi, fj]:.3f})")

    kept = [f for f in features if f not in drop]
    logger.info(f"De-correlated feature set: {kept} (dropped {drop})")
    return kept


# ── Step 4: calibrated logistic regression ───────────────────────────

def fit_calibrated_model(df: pd.DataFrame, features: list):
    """
    Fit logistic regression with isotonic calibration predicting
    gold-standard overlap from combo features.

    Uses leave-one-cancer-out (LOCO) cross-validation to avoid
    information leakage.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss, log_loss)

    X = df[features].values
    y = df['label'].values
    cancers = df['cancer_type'].values
    unique_cancers = sorted(set(cancers))

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LOCO cross-validation
    y_prob_loco = np.full(len(y), np.nan)
    coef_matrix = []  # cancer → coefficients

    for held_out in unique_cancers:
        train_mask = cancers != held_out
        test_mask = cancers == held_out

        if y[train_mask].sum() < 2 or y[train_mask].sum() == len(y[train_mask]):
            # Too few positives — use prior
            y_prob_loco[test_mask] = y[train_mask].mean()
            continue

        clf = LogisticRegression(
            C=1.0, penalty='l2', solver='lbfgs',
            max_iter=10000, class_weight='balanced'
        )
        clf.fit(X_scaled[train_mask], y[train_mask])
        coef_matrix.append(clf.coef_[0].copy())

        # Isotonic calibration on training fold
        try:
            cal = CalibratedClassifierCV(clf, cv=3, method='isotonic')
            cal.fit(X_scaled[train_mask], y[train_mask])
            y_prob_loco[test_mask] = cal.predict_proba(X_scaled[test_mask])[:, 1]
        except Exception:
            y_prob_loco[test_mask] = clf.predict_proba(X_scaled[test_mask])[:, 1]

    # Metrics on LOCO predictions
    valid = ~np.isnan(y_prob_loco)
    metrics = {}
    if valid.sum() > 10 and y[valid].sum() > 0:
        metrics['auroc'] = roc_auc_score(y[valid], y_prob_loco[valid])
        metrics['auprc'] = average_precision_score(y[valid], y_prob_loco[valid])
        metrics['brier'] = brier_score_loss(y[valid], y_prob_loco[valid])
        metrics['log_loss'] = log_loss(y[valid], y_prob_loco[valid].clip(1e-6, 1-1e-6))

    # Compare with heuristic combined_score ranking
    # (negate because lower combined_score = better, but AUC expects higher=positive)
    heuristic_score = -df['combined_score'].values
    if valid.sum() > 10 and y[valid].sum() > 0:
        metrics['heuristic_auroc'] = roc_auc_score(y[valid], heuristic_score[valid])
        metrics['heuristic_auprc'] = average_precision_score(y[valid], heuristic_score[valid])

    # Fit full model for coefficients
    clf_full = LogisticRegression(
        C=1.0, penalty='l2', solver='lbfgs',
        max_iter=10000, class_weight='balanced'
    )
    clf_full.fit(X_scaled, y)

    coef_df = pd.DataFrame({
        'feature': features,
        'coefficient': clf_full.coef_[0],
        'abs_coefficient': np.abs(clf_full.coef_[0]),
        'odds_ratio': np.exp(clf_full.coef_[0]),
    }).sort_values('abs_coefficient', ascending=False)

    # Mean LOCO coefficients (stability)
    if coef_matrix:
        coef_arr = np.array(coef_matrix)
        coef_df['mean_loco_coef'] = coef_arr.mean(axis=0)
        coef_df['std_loco_coef'] = coef_arr.std(axis=0)

    logger.info(f"Calibrated LR — LOCO AUROC: {metrics.get('auroc', 'N/A'):.3f}  "
                f"vs heuristic: {metrics.get('heuristic_auroc', 'N/A'):.3f}")

    return {
        'metrics': metrics,
        'coef_df': coef_df,
        'y_prob_loco': y_prob_loco,
        'clf_full': clf_full,
        'scaler': scaler,
    }


# ── Step 5: SHAP values ─────────────────────────────────────────────

def compute_shap_values(df: pd.DataFrame, features: list, model_result: dict):
    """Compute SHAP values for the full logistic regression."""
    import shap

    clf = model_result['clf_full']
    scaler = model_result['scaler']
    X = df[features].values
    X_scaled = scaler.transform(X)

    # Use LinearExplainer (exact for linear models)
    explainer = shap.LinearExplainer(clf, X_scaled, feature_names=features)
    shap_values = explainer.shap_values(X_scaled)

    # Mean |SHAP| per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_shap,
    }).sort_values('mean_abs_shap', ascending=False)

    shap_df.to_csv(OUTDIR / 'shap_importance.csv', index=False)
    logger.info(f"SHAP importance:\n{shap_df.to_string(index=False)}")

    return shap_values, shap_df


# ── Step 6: leave-one-feature-out ablation ───────────────────────────

def feature_ablation(df: pd.DataFrame, features: list):
    """
    Leave-one-feature-out: fit LR on all-but-one feature, compute
    LOCO AUROC. The delta from full-model tells per-feature contribution.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    X_full = df[features].values
    y = df['label'].values
    cancers = df['cancer_type'].values
    unique_cancers = sorted(set(cancers))

    def loco_auroc(X):
        """LOCO AUROC for given feature matrix."""
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        y_prob = np.full(len(y), np.nan)
        for held_out in unique_cancers:
            train = cancers != held_out
            test = cancers == held_out
            if y[train].sum() < 2 or y[train].sum() == len(y[train]):
                y_prob[test] = y[train].mean()
                continue
            clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                                     max_iter=10000, class_weight='balanced')
            clf.fit(Xs[train], y[train])
            y_prob[test] = clf.predict_proba(Xs[test])[:, 1]
        valid = ~np.isnan(y_prob)
        if valid.sum() > 10 and y[valid].sum() > 0:
            return roc_auc_score(y[valid], y_prob[valid])
        return 0.5

    # Full model AUROC
    full_auroc = loco_auroc(X_full)

    # Ablate each feature
    ablation_rows = []
    for i, feat in enumerate(features):
        X_ablated = np.delete(X_full, i, axis=1)
        abl_auroc = loco_auroc(X_ablated)
        delta = full_auroc - abl_auroc
        ablation_rows.append({
            'feature': feat,
            'full_auroc': round(full_auroc, 4),
            'ablated_auroc': round(abl_auroc, 4),
            'delta_auroc': round(delta, 4),
        })
        logger.info(f"  Ablate {feat}: AUROC {abl_auroc:.4f} (Δ={delta:+.4f})")

    ablation_df = pd.DataFrame(ablation_rows).sort_values('delta_auroc', ascending=False)
    ablation_df.to_csv(OUTDIR / 'feature_ablation.csv', index=False)
    return ablation_df


# ── Step 7: figure ───────────────────────────────────────────────────

def generate_figure(df, pearson_corr, features, model_result,
                    shap_df, ablation_df):
    """4-panel publication figure."""

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # --- Panel A: Feature correlation heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    feat_labels = [FEATURE_LABELS.get(f, f) for f in features]
    corr_sub = pearson_corr.loc[features, features].values
    im = ax1.imshow(corr_sub, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(features)))
    ax1.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(feat_labels, fontsize=8)
    # Annotate with values
    for i in range(len(features)):
        for j in range(len(features)):
            val = corr_sub[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=6, color=color)
    plt.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title('A. Feature correlation matrix', fontsize=11, fontweight='bold')

    # --- Panel B: SHAP importance ---
    ax2 = fig.add_subplot(gs[0, 1])
    shap_sorted = shap_df.sort_values('mean_abs_shap', ascending=True)
    y_pos = range(len(shap_sorted))
    bars = ax2.barh(y_pos, shap_sorted['mean_abs_shap'], color='steelblue', edgecolor='navy')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([FEATURE_LABELS.get(f, f) for f in shap_sorted['feature']],
                         fontsize=9)
    ax2.set_xlabel('Mean |SHAP value|', fontsize=10)
    ax2.set_title('B. SHAP feature importance', fontsize=11, fontweight='bold')
    # Annotate bars
    for bar, val in zip(bars, shap_sorted['mean_abs_shap']):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=7)

    # --- Panel C: Leave-one-feature-out ablation ---
    ax3 = fig.add_subplot(gs[1, 0])
    abl_sorted = ablation_df.sort_values('delta_auroc', ascending=True)
    y_pos3 = range(len(abl_sorted))
    colors = ['#d62728' if d > 0 else '#2ca02c' for d in abl_sorted['delta_auroc']]
    bars3 = ax3.barh(y_pos3, abl_sorted['delta_auroc'], color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(y_pos3)
    ax3.set_yticklabels([FEATURE_LABELS.get(f, f) for f in abl_sorted['feature']],
                         fontsize=9)
    ax3.set_xlabel('ΔAUROC when removed', fontsize=10)
    ax3.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax3.set_title('C. Feature ablation (LOCO)', fontsize=11, fontweight='bold')
    # Reference line for full AUROC
    full_auroc = ablation_df['full_auroc'].iloc[0]
    ax3.text(0.98, 0.02, f'Full AUROC: {full_auroc:.3f}',
             transform=ax3.transAxes, ha='right', va='bottom', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    # --- Panel D: Calibrated vs heuristic ---
    ax4 = fig.add_subplot(gs[1, 1])
    metrics = model_result['metrics']
    metric_names = ['AUROC', 'AUPRC', 'Brier', 'Log loss']
    calibrated_vals = [
        metrics.get('auroc', 0),
        metrics.get('auprc', 0),
        metrics.get('brier', 1),
        metrics.get('log_loss', 5),
    ]
    heuristic_vals = [
        metrics.get('heuristic_auroc', 0),
        metrics.get('heuristic_auprc', 0),
        np.nan,  # no Brier for ranker
        np.nan,  # no log-loss for ranker
    ]

    x_pos = np.arange(len(metric_names))
    width = 0.35
    bars_c = ax4.bar(x_pos - width/2, calibrated_vals, width, label='Calibrated LR',
                     color='steelblue', edgecolor='navy')
    bars_h = ax4.bar(x_pos + width/2, heuristic_vals, width, label='Heuristic score',
                     color='coral', edgecolor='darkred')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metric_names, fontsize=9)
    ax4.set_ylabel('Score', fontsize=10)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_title('D. Calibrated LR vs heuristic', fontsize=11, fontweight='bold')
    # Annotate
    for bar, val in zip(bars_c, calibrated_vals):
        if not np.isnan(val):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=7)
    for bar, val in zip(bars_h, heuristic_vals):
        if not np.isnan(val):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=7)

    plt.savefig(OUTDIR / 'evidence_calibration.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR / 'evidence_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Figure saved to {OUTDIR}/evidence_calibration.{{pdf,png}}")


# ── Step 8: logistic-regression re-weighted score comparison ─────────

def compare_reweighted_scoring(df, model_result, features):
    """
    Compare the heuristic rank ordering with the calibrated LR
    rank ordering. Report how many top-1 triples change per cancer,
    and concordance (Kendall τ).
    """
    from scipy.stats import kendalltau

    y_prob = model_result['y_prob_loco']
    cancers = df['cancer_type'].unique()

    results = []
    for cancer in sorted(cancers):
        mask = df['cancer_type'] == cancer
        sub = df[mask].copy()
        if len(sub) < 2:
            continue

        # Heuristic rank (lower combined_score = rank 1)
        sub = sub.sort_values('combined_score')
        heuristic_top = sub.iloc[0]['targets']

        # Calibrated rank (higher prob = rank 1)
        sub_probs = y_prob[mask.values]
        if np.any(np.isnan(sub_probs)):
            continue
        sub = sub.copy()
        sub['lr_prob'] = sub_probs
        sub_lr = sub.sort_values('lr_prob', ascending=False)
        calibrated_top = sub_lr.iloc[0]['targets']

        # Kendall tau between rankings
        rank_h = sub['combined_score'].rank().values
        rank_c = (-sub['lr_prob']).values  # negate for rank direction
        tau, pval = kendalltau(rank_h, rank_c)

        results.append({
            'cancer_type': cancer,
            'n_combos': len(sub),
            'heuristic_top1': heuristic_top,
            'calibrated_top1': calibrated_top,
            'top1_changed': heuristic_top != calibrated_top,
            'kendall_tau': round(tau, 3),
            'kendall_pval': round(pval, 4),
        })

    results_df = pd.DataFrame(results)
    n_changed = results_df['top1_changed'].sum()
    n_total = len(results_df)
    results_df.to_csv(OUTDIR / 'rank_comparison.csv', index=False)
    logger.info(f"Top-1 changed in {n_changed}/{n_total} cancers "
                f"({100*n_changed/max(n_total,1):.1f}%)")
    logger.info(f"Mean Kendall τ: {results_df['kendall_tau'].mean():.3f}")
    return results_df


# ── main ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Evidence Calibration & De-correlation Analysis")
    logger.info("=" * 60)

    # Step 1: collect data
    logger.info("\n[1/7] Collecting all scored combinations...")
    df = collect_all_scored_combos()

    if len(df) == 0:
        logger.error("No combinations collected — aborting")
        return

    # Reconstruct hub_penalty and perturbation_bonus
    df = collect_combos_with_hub_penalty(df)
    df.to_csv(OUTDIR / 'all_combos_features.csv', index=False)
    logger.info(f"Saved {len(df)} combos to {OUTDIR}/all_combos_features.csv")

    # Step 2: correlations
    logger.info("\n[2/7] Analyzing feature correlations...")
    avail_features = [f for f in FEATURE_NAMES if f in df.columns]
    pearson_corr = analyze_correlations(df)

    # Step 3: de-correlate
    logger.info("\n[3/7] Selecting de-correlated feature set...")
    decorr_features = select_decorrelated_features(pearson_corr, threshold=0.7)

    # Step 4: calibrated model (using FULL feature set for comparison)
    logger.info("\n[4/7] Fitting calibrated logistic regression (all features)...")
    model_all = fit_calibrated_model(df, avail_features)
    model_all['coef_df'].to_csv(OUTDIR / 'lr_coefficients_all.csv', index=False)

    logger.info("\n[4b/7] Fitting with de-correlated features...")
    model_decorr = fit_calibrated_model(df, decorr_features)
    model_decorr['coef_df'].to_csv(OUTDIR / 'lr_coefficients_decorrelated.csv', index=False)

    # Step 5: SHAP
    logger.info("\n[5/7] Computing SHAP values...")
    shap_values, shap_df = compute_shap_values(df, avail_features, model_all)

    # Step 6: feature ablation
    logger.info("\n[6/7] Leave-one-feature-out ablation...")
    ablation_df = feature_ablation(df, avail_features)

    # Step 6b: rank comparison
    logger.info("\n[6b/7] Comparing heuristic vs calibrated rankings...")
    rank_df = compare_reweighted_scoring(df, model_all, avail_features)

    # Step 7: figure
    logger.info("\n[7/7] Generating figure...")
    generate_figure(df, pearson_corr, avail_features, model_all,
                    shap_df, ablation_df)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    m = model_all['metrics']
    logger.info(f"Total combos: {len(df)}  |  Positives: {df['label'].sum()}")
    logger.info(f"Calibrated LR AUROC (LOCO): {m.get('auroc', 'N/A'):.3f}")
    logger.info(f"Heuristic AUROC:            {m.get('heuristic_auroc', 'N/A'):.3f}")
    logger.info(f"Calibrated AUPRC:           {m.get('auprc', 'N/A'):.3f}")
    logger.info(f"Heuristic AUPRC:            {m.get('heuristic_auprc', 'N/A'):.3f}")
    logger.info(f"Brier score:                {m.get('brier', 'N/A'):.3f}")

    # Feature importance summary
    logger.info("\nFeature importance (SHAP):")
    for _, row in shap_df.iterrows():
        logger.info(f"  {FEATURE_LABELS.get(row['feature'], row['feature']):20s} "
                     f"SHAP={row['mean_abs_shap']:.4f}")

    logger.info(f"\nCorrelated pairs (|r|>0.7): "
                f"{len(pd.read_csv(OUTDIR / 'flagged_correlated_pairs.csv'))}")
    logger.info(f"De-correlated features: {decorr_features}")

    md = model_decorr['metrics']
    logger.info(f"\nDe-correlated model AUROC: {md.get('auroc', 'N/A'):.3f}")

    if rank_df is not None and len(rank_df) > 0:
        n_changed = rank_df['top1_changed'].sum()
        logger.info(f"Top-1 changed (calibrated vs heuristic): "
                     f"{n_changed}/{len(rank_df)} cancers")

    # Save full summary JSON
    summary = {
        'total_combos': int(len(df)),
        'positive_combos': int(df['label'].sum()),
        'n_cancers': int(df['cancer_type'].nunique()),
        'calibrated_auroc': round(m.get('auroc', 0), 4),
        'heuristic_auroc': round(m.get('heuristic_auroc', 0), 4),
        'calibrated_auprc': round(m.get('auprc', 0), 4),
        'heuristic_auprc': round(m.get('heuristic_auprc', 0), 4),
        'brier_score': round(m.get('brier', 1), 4),
        'decorrelated_auroc': round(md.get('auroc', 0), 4),
        'decorrelated_features': decorr_features,
        'n_correlated_pairs': len(pd.read_csv(OUTDIR / 'flagged_correlated_pairs.csv')),
        'top1_changed': int(rank_df['top1_changed'].sum()) if rank_df is not None else 0,
        'top1_total': len(rank_df) if rank_df is not None else 0,
        'shap_ranking': shap_df.to_dict('records'),
        'ablation_ranking': ablation_df.to_dict('records'),
        'lr_coefficients': model_all['coef_df'].to_dict('records'),
    }
    with open(OUTDIR / 'calibration_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {OUTDIR}/")
    logger.info("Done.")


if __name__ == '__main__':
    main()
