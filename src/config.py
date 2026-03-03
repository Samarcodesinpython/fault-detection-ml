from __future__ import annotations

from pathlib import Path


# Project paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
SUBMISSIONS_DIR: Path = BASE_DIR / "submissions"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"

TRAIN_FILE: Path = DATA_DIR / "TRAIN.csv"
TEST_FILE: Path = DATA_DIR / "TEST.csv"

# Data schema
TARGET_COL: str = "Class"
ID_COL: str = "ID"
FEATURE_PREFIX: str = "F"
N_FEATURES: int = 47

# Modeling configuration
RANDOM_STATE: int = 42
N_FOLDS: int = 5
TEST_SIZE: float = 0.2
CORR_THRESHOLD: float = 0.95


def ensure_directories() -> None:
    """Ensure that key output directories exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "SUBMISSIONS_DIR",
    "NOTEBOOKS_DIR",
    "TRAIN_FILE",
    "TEST_FILE",
    "TARGET_COL",
    "ID_COL",
    "FEATURE_PREFIX",
    "N_FEATURES",
    "RANDOM_STATE",
    "N_FOLDS",
    "TEST_SIZE",
    "CORR_THRESHOLD",
    "ensure_directories",
]

