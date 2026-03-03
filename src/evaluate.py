from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn import metrics


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
) -> Dict[str, float]:
    """Compute a set of standard binary classification metrics."""
    results: Dict[str, float] = {}
    results["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    results["precision"] = metrics.precision_score(y_true, y_pred, zero_division=0)
    results["recall"] = metrics.recall_score(y_true, y_pred, zero_division=0)
    results["f1"] = metrics.f1_score(y_true, y_pred, zero_division=0)

    if y_proba is not None:
        results["roc_auc"] = metrics.roc_auc_score(y_true, y_proba)
    else:
        results["roc_auc"] = np.nan

    return results


def aggregate_cv_results(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-fold metrics into mean and std for each metric."""
    if not fold_metrics:
        raise ValueError("No fold metrics provided.")

    metric_names = fold_metrics[0].keys()
    aggregated: Dict[str, Dict[str, float]] = {}

    for metric_name in metric_names:
        values = np.array([m[metric_name] for m in fold_metrics], dtype=float)
        aggregated[metric_name] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }

    return aggregated


__all__ = ["compute_classification_metrics", "aggregate_cv_results"]

