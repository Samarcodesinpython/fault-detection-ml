from __future__ import annotations

import logging
from typing import Dict

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import Preprocessor

logger = logging.getLogger(__name__)


def _build_pipeline(estimator) -> Pipeline:
    """Helper to build a Pipeline with shared preprocessing."""
    preprocessor = Preprocessor(remove_high_corr=True, corr_threshold=config.CORR_THRESHOLD)
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", estimator)])


def get_models(class_weight: str = "balanced", scale_pos_weight: float = 1.0) -> Dict[str, Pipeline]:
    """Create baseline model pipelines for comparison.

    Parameters
    ----------
    class_weight:
        Passed to models that support class weights (Logistic Regression, Random Forest).
    scale_pos_weight:
        Positive class weight for XGBoost and LightGBM to handle imbalance.
        Typically computed as n_negative / n_positive.
    """
    logger.info("Building model pipelines with class_weight=%s, scale_pos_weight=%s", class_weight, scale_pos_weight)

    log_reg = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=class_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        class_weight=class_weight,
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
    )

    lgbm = LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        class_weight=None,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    models = {
        "logistic_regression": _build_pipeline(log_reg),
        "random_forest": _build_pipeline(rf),
        "xgboost": _build_pipeline(xgb),
        "lightgbm": _build_pipeline(lgbm),
    }

    return models


__all__ = ["get_models"]

