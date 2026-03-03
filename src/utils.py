from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np

from . import config


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for the project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def seed_everything(seed: Optional[int] = None) -> int:
    """Seed Python, NumPy and relevant libraries for reproducibility."""
    if seed is None:
        seed = config.RANDOM_STATE

    random.seed(seed)
    np.random.seed(seed)

    # Some libraries (e.g. XGBoost, LightGBM) accept seed via their own params.
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names F01..F47."""
    return [f"{config.FEATURE_PREFIX}{i:02d}" for i in range(1, config.N_FEATURES + 1)]


__all__ = ["setup_logging", "seed_everything", "get_feature_columns"]

