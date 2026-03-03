from __future__ import annotations

import logging

import joblib
import pandas as pd

from . import config
from .data_loader import load_test_data
from .utils import get_feature_columns, seed_everything, setup_logging


logger = logging.getLogger(__name__)


def generate_predictions() -> None:
    """Load the trained model, run predictions on TEST.csv, and save submission."""
    seed_everything()
    config.ensure_directories()

    model_path = config.MODELS_DIR / "final_model.pkl"
    logger.info("Loading final model from %s", model_path)
    model = joblib.load(model_path)

    test_df = load_test_data()
    feature_cols = get_feature_columns()
    X_test = test_df[feature_cols]

    logger.info("Generating predictions for test data.")
    y_pred = model.predict(X_test)

    submission = pd.DataFrame(
        {
            config.ID_COL: test_df[config.ID_COL],
            "CLASS": y_pred.astype(int),
        }
    )

    submission_path = config.SUBMISSIONS_DIR / "final_submission.csv"
    submission.to_csv(submission_path, index=False)
    logger.info("Submission file saved to %s", submission_path)


def main() -> None:
    setup_logging()
    generate_predictions()


if __name__ == "__main__":
    main()

