from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

from . import config
from .utils import get_feature_columns

logger = logging.getLogger(__name__)


def load_train_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the training data and return features X and target y."""
    logger.info("Loading training data from %s", config.TRAIN_FILE)
    df = pd.read_csv(config.TRAIN_FILE)

    feature_cols = get_feature_columns()
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature columns in TRAIN.csv: {missing_features}")

    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COL}' not found in TRAIN.csv")

    X = df[feature_cols]
    y = df[config.TARGET_COL].astype(int)
    logger.info("Training data loaded: X shape=%s, y shape=%s", X.shape, y.shape)
    return X, y


def load_test_data() -> pd.DataFrame:
    """Load the test data including ID and feature columns."""
    logger.info("Loading test data from %s", config.TEST_FILE)
    df = pd.read_csv(config.TEST_FILE)

    if config.ID_COL not in df.columns:
        raise ValueError(f"ID column '{config.ID_COL}' not found in TEST.csv")

    feature_cols = get_feature_columns()
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature columns in TEST.csv: {missing_features}")

    logger.info("Test data loaded: shape=%s", df.shape)
    return df


__all__ = ["load_train_data", "load_test_data"]

