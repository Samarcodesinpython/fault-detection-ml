from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

logger = logging.getLogger(__name__)


class Preprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessing transformer.

    Steps:
    - Median imputation
    - Remove constant features (zero variance)
    - Optionally remove highly correlated features
    - Standard scaling
    """

    def __init__(self, remove_high_corr: bool = True, corr_threshold: float = config.CORR_THRESHOLD):
        self.remove_high_corr = remove_high_corr
        self.corr_threshold = corr_threshold

        self.imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.constant_feature_indices_: Optional[np.ndarray] = None
        self.correlated_feature_indices_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.output_feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[Sequence] = None):
        """Fit preprocessing on features X."""
        X_arr, feature_names = self._to_array_and_feature_names(X)
        self.feature_names_in_ = feature_names

        # Impute missing values
        self.imputer_ = SimpleImputer(strategy="median")
        X_imputed = self.imputer_.fit_transform(X_arr)

        # Remove constant (zero-variance) features
        variances = np.var(X_imputed, axis=0)
        constant_idx = np.where(variances == 0.0)[0]
        self.constant_feature_indices_ = constant_idx.astype(int)
        non_constant_mask = variances > 0.0
        X_nc = X_imputed[:, non_constant_mask]
        nc_feature_names = [name for name, keep in zip(feature_names, non_constant_mask) if keep]

        # Optionally remove highly correlated features
        if self.remove_high_corr and X_nc.shape[1] > 1:
            corr_matrix = np.corrcoef(X_nc, rowvar=False)
            upper = np.triu_indices_from(corr_matrix, k=1)
            to_drop = set()
            for i, j in zip(*upper):
                if abs(corr_matrix[i, j]) > self.corr_threshold:
                    # Drop the later feature (j)
                    to_drop.add(j)
            self.correlated_feature_indices_ = np.array(sorted(to_drop), dtype=int) if to_drop else np.array([], dtype=int)

            keep_mask = np.ones(X_nc.shape[1], dtype=bool)
            keep_mask[self.correlated_feature_indices_] = False
            X_sel = X_nc[:, keep_mask]
            selected_feature_names = [name for name, keep in zip(nc_feature_names, keep_mask) if keep]
        else:
            self.correlated_feature_indices_ = np.array([], dtype=int)
            X_sel = X_nc
            selected_feature_names = nc_feature_names

        # Indices of selected features relative to original feature order
        self.selected_indices_ = np.array(
            [self.feature_names_in_.index(name) for name in selected_feature_names],
            dtype=int,
        )
        self.output_feature_names_ = selected_feature_names

        # Standard scaling
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_sel)

        logger.info(
            "Preprocessor fitted: %d original features, %d constant removed, %d correlated removed, %d final features",
            len(feature_names),
            len(self.constant_feature_indices_),
            len(self.correlated_feature_indices_),
            len(self.output_feature_names_),
        )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply preprocessing to features X."""
        if self.imputer_ is None or self.scaler_ is None or self.selected_indices_ is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        X_arr, _ = self._to_array_and_feature_names(X, expect_names=self.feature_names_in_)
        X_imputed = self.imputer_.transform(X_arr)

        # Remove constant and correlated features using selected_indices_
        X_sel = X_imputed[:, self.selected_indices_]
        X_scaled = self.scaler_.transform(X_sel)
        return X_scaled

    def get_feature_names_out(self) -> List[str]:
        if self.output_feature_names_ is None:
            raise RuntimeError("Preprocessor must be fitted before getting feature names.")
        return self.output_feature_names_

    @staticmethod
    def _to_array_and_feature_names(
        X: pd.DataFrame | np.ndarray,
        expect_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if isinstance(X, pd.DataFrame):
            arr = X.values
            feature_names = list(X.columns)
        else:
            arr = np.asarray(X)
            if expect_names is not None:
                feature_names = list(expect_names)
            else:
                feature_names = [f"feature_{i}" for i in range(arr.shape[1])]
        return arr, feature_names


def build_preprocessing_pipeline(
    remove_high_corr: bool = True,
    corr_threshold: float = config.CORR_THRESHOLD,
) -> Pipeline:
    """Return a scikit-learn Pipeline that applies the Preprocessor."""
    preprocessor = Preprocessor(remove_high_corr=remove_high_corr, corr_threshold=corr_threshold)
    return Pipeline(steps=[("preprocess", preprocessor)])


__all__ = ["Preprocessor", "build_preprocessing_pipeline"]

