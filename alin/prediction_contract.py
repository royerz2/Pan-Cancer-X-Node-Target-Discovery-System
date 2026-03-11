"""Shared helpers for ranked prediction CSV contracts used by benchmarks."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

RANKED_TRIPLES_FILENAME = 'ranked_triple_combinations.csv'
TOP_TRIPLES_FILENAME = 'triple_combinations.csv'


@dataclass(frozen=True)
class RankedPrediction:
    targets: frozenset[str]
    rank: float
    source: str = 'triple'


@dataclass(frozen=True)
class LoadedPredictions:
    rows: pd.DataFrame
    predictions_by_cancer: Dict[str, List[RankedPrediction]]
    resolved_path: Path
    used_legacy_best_combo: bool


def read_prediction_rows(path) -> pd.DataFrame:
    """Read a prediction CSV, normalizing spaced target columns."""
    df = pd.read_csv(path)
    df.columns = [column.replace(' ', '_') for column in df.columns]
    return df


def clean_target_set(values: Iterable[object]) -> frozenset[str]:
    """Normalize a sequence of CSV values into a frozenset of gene symbols."""
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == 'nan':
            continue
        cleaned.append(text)
    return frozenset(cleaned)


def extract_primary_targets(row) -> frozenset[str]:
    """Extract the main predicted triple from a normalized row."""
    return clean_target_set([
        row.get('Target_1', ''),
        row.get('Target_2', ''),
        row.get('Target_3', ''),
    ])


def extract_best_combo_targets(row) -> frozenset[str]:
    """Extract best-of-any-size prediction metadata when present."""
    targets = clean_target_set([
        row.get('Best_Combo_1', ''),
        row.get('Best_Combo_2', ''),
        row.get('Best_Combo_3', ''),
    ])
    return targets if len(targets) >= 2 else frozenset()


def prepare_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Sort prediction rows by explicit rank when available, else preserve file order."""
    if df.empty:
        return df.copy()

    prepared = df.copy()
    prepared['_row_order'] = np.arange(len(prepared))
    if 'Rank' in prepared.columns:
        prepared['_explicit_rank'] = pd.to_numeric(prepared['Rank'], errors='coerce')
    else:
        prepared['_explicit_rank'] = np.nan

    prepared = prepared.sort_values(
        ['Cancer_Type', '_explicit_rank', '_row_order'],
        ascending=[True, True, True],
        na_position='last',
    )
    prepared['_prediction_rank'] = prepared.groupby('Cancer_Type').cumcount() + 1
    has_explicit_rank = prepared['_explicit_rank'].notna()
    prepared.loc[has_explicit_rank, '_prediction_rank'] = prepared.loc[
        has_explicit_rank, '_explicit_rank'
    ]
    return prepared


def resolve_benchmark_predictions_path(predictions_csv) -> Tuple[Path, bool]:
    """Resolve the ranked companion file used by evaluation when available."""
    path = Path(predictions_csv)

    if path.is_dir():
        ranked_path = path / RANKED_TRIPLES_FILENAME
        if ranked_path.exists():
            return ranked_path, True
        top_triples_path = path / TOP_TRIPLES_FILENAME
        if top_triples_path.exists():
            return top_triples_path, False
        return path, False

    if path.name.lower() == TOP_TRIPLES_FILENAME:
        ranked_path = path.with_name(RANKED_TRIPLES_FILENAME)
        if ranked_path.exists():
            return ranked_path, True

    return path, False


def uses_ranked_prediction_contract(resolved_path, used_ranked_companion: bool = False) -> bool:
    """Return whether evaluation should treat the source as explicit ranked predictions."""
    return used_ranked_companion or Path(resolved_path).name.lower() == RANKED_TRIPLES_FILENAME


def load_benchmark_prediction_rows(predictions_csv) -> Tuple[pd.DataFrame, Path, bool]:
    """Load the prediction rows actually consumed by benchmark evaluation."""
    resolved_path, used_ranked_companion = resolve_benchmark_predictions_path(predictions_csv)
    rows = prepare_prediction_rows(read_prediction_rows(resolved_path))
    return rows, resolved_path, used_ranked_companion


def load_ranked_predictions(
    predictions_csv,
    include_legacy_best_combo: bool = True,
) -> LoadedPredictions:
    """Load per-cancer ranked predictions with backward-compatible legacy fallback."""
    rows, resolved_path, used_ranked_companion = load_benchmark_prediction_rows(predictions_csv)
    use_legacy_best_combo = include_legacy_best_combo and not uses_ranked_prediction_contract(
        resolved_path,
        used_ranked_companion,
    )

    raw_predictions = defaultdict(list)
    best_combo_injected = set()

    for sequence, (_, row) in enumerate(rows.iterrows()):
        cancer_type = row['Cancer_Type']
        prediction_rank = row.get('_prediction_rank', np.nan)
        if pd.isna(prediction_rank):
            prediction_rank = float('inf')
        else:
            prediction_rank = float(prediction_rank)

        primary_targets = extract_primary_targets(row)
        if primary_targets:
            raw_predictions[cancer_type].append(
                (prediction_rank, sequence, primary_targets, 'triple')
            )

        if use_legacy_best_combo and cancer_type not in best_combo_injected:
            best_combo_targets = extract_best_combo_targets(row)
            if best_combo_targets and best_combo_targets != primary_targets:
                raw_predictions[cancer_type].append(
                    (prediction_rank - 0.5, sequence, best_combo_targets, 'best_combo')
                )
            best_combo_injected.add(cancer_type)

    predictions_by_cancer = {}
    for cancer_type, predictions in raw_predictions.items():
        seen = set()
        ordered_predictions = []
        for rank, sequence, targets, source in sorted(predictions, key=lambda item: (item[0], item[1])):
            if targets in seen:
                continue
            seen.add(targets)
            ordered_predictions.append(RankedPrediction(targets=targets, rank=rank, source=source))
        predictions_by_cancer[cancer_type] = ordered_predictions

    return LoadedPredictions(
        rows=rows,
        predictions_by_cancer=predictions_by_cancer,
        resolved_path=resolved_path,
        used_legacy_best_combo=use_legacy_best_combo,
    )


__all__ = [
    'LoadedPredictions',
    'RANKED_TRIPLES_FILENAME',
    'TOP_TRIPLES_FILENAME',
    'RankedPrediction',
    'clean_target_set',
    'extract_best_combo_targets',
    'extract_primary_targets',
    'load_benchmark_prediction_rows',
    'load_ranked_predictions',
    'prepare_prediction_rows',
    'read_prediction_rows',
    'resolve_benchmark_predictions_path',
    'uses_ranked_prediction_contract',
]