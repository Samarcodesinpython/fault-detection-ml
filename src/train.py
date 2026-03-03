from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config
from .data_loader import load_train_data
from .evaluate import aggregate_cv_results, compute_classification_metrics
from .model import get_models
from .preprocessing import Preprocessor
from .utils import seed_everything, setup_logging


logger = logging.getLogger(__name__)


def _compute_class_imbalance(y) -> Tuple[float, float]:
    """Return (pos_ratio, scale_pos_weight = n_neg / n_pos)."""
    y = np.asarray(y)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0:
        raise ValueError("No positive samples found in training data.")
    pos_ratio = n_pos / (n_pos + n_neg)
    scale_pos_weight = n_neg / n_pos
    return float(pos_ratio), float(scale_pos_weight)


def cross_validate_models():
    """Run StratifiedKFold CV for all baseline models and select the best one."""
    X, y = load_train_data()
    _, scale_pos_weight = _compute_class_imbalance(y)

    models = get_models(class_weight="balanced", scale_pos_weight=scale_pos_weight)

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_name, model in models.items():
        logger.info("Starting cross-validation for model: %s", model_name)
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "decision_function"):
                # Calibrate to probabilities via sigmoid-like mapping if needed
                scores = model.decision_function(X_val)
                y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            else:
                y_proba = None

            fold_result = compute_classification_metrics(y_val, y_pred, y_proba)
            fold_metrics.append(fold_result)

            logger.info(
                "Model %s - Fold %d/%d - metrics: %s",
                model_name,
                fold_idx,
                config.N_FOLDS,
                fold_result,
            )

        aggregated = aggregate_cv_results(fold_metrics)
        all_results[model_name] = aggregated

        logger.info("Model %s - CV aggregated metrics:", model_name)
        for metric_name, stats in aggregated.items():
            logger.info(
                "  %s: mean=%.4f, std=%.4f",
                metric_name,
                stats["mean"],
                stats["std"],
            )

    # Select best model by ROC-AUC
    best_model_name = None
    best_auc = -np.inf
    for model_name, metrics_dict in all_results.items():
        roc_auc_mean = metrics_dict.get("roc_auc", {}).get("mean", np.nan)
        if roc_auc_mean > best_auc:
            best_auc = roc_auc_mean
            best_model_name = model_name

    if best_model_name is None:
        raise RuntimeError("Failed to select best model based on ROC-AUC.")

    logger.info("Best baseline model by ROC-AUC: %s (ROC-AUC=%.4f)", best_model_name, best_auc)
    return best_model_name, all_results


def _get_xgboost_param_distributions(scale_pos_weight: float) -> Dict[str, list]:
    return {
        "clf__n_estimators": list(range(100, 501, 50)),
        "clf__max_depth": [3, 4, 5, 6, 7, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__scale_pos_weight": [scale_pos_weight],
    }


def _get_lightgbm_param_distributions(scale_pos_weight: float) -> Dict[str, list]:
    return {
        "clf__n_estimators": list(range(100, 501, 50)),
        "clf__num_leaves": [15, 31, 63, 127],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__feature_fraction": [0.6, 0.8, 1.0],
        "clf__bagging_fraction": [0.6, 0.8, 1.0],
        "clf__scale_pos_weight": [scale_pos_weight],
    }


def hyperparameter_tuning():
    """Perform RandomizedSearchCV for XGBoost and LightGBM and return best model."""
    X, y = load_train_data()
    _, scale_pos_weight = _compute_class_imbalance(y)

    from .model import get_models

    base_models = get_models(class_weight="balanced", scale_pos_weight=scale_pos_weight)
    xgb_pipeline = base_models["xgboost"]
    lgbm_pipeline = base_models["lightgbm"]

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    xgb_search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=_get_xgboost_param_distributions(scale_pos_weight),
        n_iter=25,
        scoring="roc_auc",
        n_jobs=-1,
        cv=skf,
        verbose=1,
        random_state=config.RANDOM_STATE,
        refit=True,
    )

    lgbm_search = RandomizedSearchCV(
        estimator=lgbm_pipeline,
        param_distributions=_get_lightgbm_param_distributions(scale_pos_weight),
        n_iter=25,
        scoring="roc_auc",
        n_jobs=-1,
        cv=skf,
        verbose=1,
        random_state=config.RANDOM_STATE,
        refit=True,
    )

    logger.info("Starting hyperparameter tuning for XGBoost.")
    xgb_search.fit(X, y)
    logger.info(
        "XGBoost best ROC-AUC: %.4f with params: %s",
        xgb_search.best_score_,
        xgb_search.best_params_,
    )

    logger.info("Starting hyperparameter tuning for LightGBM.")
    lgbm_search.fit(X, y)
    logger.info(
        "LightGBM best ROC-AUC: %.4f with params: %s",
        lgbm_search.best_score_,
        lgbm_search.best_params_,
    )

    # Choose the best boosting model
    if xgb_search.best_score_ >= lgbm_search.best_score_:
        best_search = xgb_search
        best_name = "xgboost"
        best_score = xgb_search.best_score_
    else:
        best_search = lgbm_search
        best_name = "lightgbm"
        best_score = lgbm_search.best_score_

    logger.info("Best boosting model: %s (ROC-AUC=%.4f)", best_name, best_score)
    return best_name, best_search


def plot_feature_importance(best_estimator, output_path: Path) -> None:
    """Plot and save feature importance for the best boosting model."""
    preprocess: Preprocessor = best_estimator.named_steps["preprocess"]
    clf = best_estimator.named_steps["clf"]

    if not hasattr(clf, "feature_importances_"):
        logger.warning("Classifier does not expose feature_importances_. Skipping plot.")
        return

    importances = clf.feature_importances_
    feature_names = preprocess.get_feature_names_out()

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 8))
    top_n = min(30, len(indices))
    selected_idx = indices[:top_n]
    plt.barh(
        [feature_names[i] for i in selected_idx][::-1],
        importances[selected_idx][::-1],
    )
    plt.xlabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logger.info("Feature importance plot saved to %s", output_path)


def train_final_model(best_boosting_estimator) -> None:
    """Train final model on full training data and save it."""
    from .data_loader import load_train_data

    X, y = load_train_data()
    best_boosting_estimator.fit(X, y)

    config.ensure_directories()
    model_path = config.MODELS_DIR / "final_model.pkl"
    joblib.dump(best_boosting_estimator, model_path)
    logger.info("Final model saved to %s", model_path)


def main() -> None:
    setup_logging()
    seed_everything()
    config.ensure_directories()

    logger.info("=== Starting cross-validation for baseline models ===")
    best_baseline_name, all_results = cross_validate_models()
    logger.info("Best baseline model: %s", best_baseline_name)

    logger.info("=== Starting hyperparameter tuning for boosting models ===")
    best_boosting_name, best_search = hyperparameter_tuning()
    best_estimator = best_search.best_estimator_

    # Plot feature importance
    fi_path = config.MODELS_DIR / "feature_importance.png"
    plot_feature_importance(best_estimator, fi_path)

    logger.info("=== Training final model on full training data ===")
    train_final_model(best_estimator)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()

