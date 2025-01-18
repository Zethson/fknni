from __future__ import annotations

from typing import Literal

import faiss
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class FaissImputer(BaseEstimator, TransformerMixin):
    """Imputer for completing missing values using Faiss, incorporating weighted averages based on distance."""

    def __init__(
        self,
        missing_values: int | float | str | None = np.nan,
        n_neighbors: int = 5,
        *,
        metric: Literal["l2", "ip"] = "l2",
        strategy: Literal["mean", "median", "weighted"] = "mean",
        index_factory: str = "Flat",
    ):
        """Initializes FaissImputer with specified parameters that are used for the imputation.

        Args:
            missing_values: The missing value to impute. Defaults to np.nan.
            n_neighbors: Number of neighbors to use for imputation. Defaults to 5.
            metric: Distance metric to use for neighbor search. Defaults to 'l2'.
            strategy: Method to compute imputed values among neighbors.
                      The weighted strategy is similar to scikt-learn's implementation,
                      where closer neighbors have a higher influence on the imputation.
            index_factory: Description of the Faiss index type to build. Defaults to 'Flat'.
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if strategy not in {"mean", "median", "weighted"}:
            raise ValueError("Unknown strategy. Choose one of 'mean', 'median', 'weighted'")

        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.strategy = strategy
        self.index_factory = index_factory
        super().__init__()

    def fit(self, X: np.ndarray | pd.DataFrame, *, y: np.ndarray | None = None) -> FaissImputer:
        """Fits the FaissImputer to the provided data.

        Args:
            X: Input data with potential missing values.
            y: Ignored, present for compatibility with sklearn's TransformerMixin.
        """
        X = np.asarray(X, dtype=np.float32)
        if np.isnan(X).all(axis=0).any():
            raise ValueError("Features with all values missing cannot be handled.")

        self.global_fallbacks_ = (
            np.nanmean(X, axis=0) if self.strategy in ["mean", "weighted"] else np.nanmedian(X, axis=0)
        )
        non_missing = ~np.isnan(X).any(axis=1)

        if non_missing.any():
            X_complete = X[non_missing]
            index = faiss.index_factory(X.shape[1], self.index_factory)
            index.metric_type = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
            index.train(X_complete)
            index.add(X_complete)
            self.index_ = index
            self.X_train_ = X_complete
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Imputes missing values in the data using the fitted Faiss index.

        Args:
            X: Data with missing values to impute.

        Returns:
            Data with imputed values as a NumPy array of the original data type.
        """
        X = check_array(X, ensure_all_finite="allow-nan")
        check_is_fitted(self)
        X_imputed = X.copy()
        is_missing = np.isnan(X_imputed)

        for idx in np.where(is_missing.any(axis=1))[0]:
            missing = is_missing[idx]
            sample = X_imputed[idx].copy()
            sample[missing] = self.global_fallbacks_[missing]

            # We were able to build a FAISS index so we perform actual KNN imputation
            if hasattr(self, "index_"):
                distances, indices = self.index_.search(sample.reshape(1, -1), self.n_neighbors)
                neighbors = self.X_train_[indices[0]]
                valid_neighbors = ~np.isnan(neighbors[:, missing]).all(axis=0)

                if valid_neighbors.any():
                    missing_cols = np.where(missing)[0][valid_neighbors]
                    if self.strategy == "weighted":
                        weights = 1 / (distances[0] + 1e-10)[:, np.newaxis]
                        neighbor_vals = neighbors[:, missing_cols]
                        weighted_sum = np.nansum(neighbor_vals * weights, axis=0)
                        weight_sum = np.nansum(weights, axis=0)
                        X_imputed[idx, missing_cols] = weighted_sum / weight_sum
                    else:
                        func = np.nanmean if self.strategy == "mean" else np.nanmedian
                        X_imputed[idx, missing_cols] = func(neighbors[:, missing_cols], axis=0)

                fallback_cols = np.where(missing)[0][~valid_neighbors]
                if len(fallback_cols):
                    X_imputed[idx, fallback_cols] = self.global_fallbacks_[fallback_cols]
            # We were not able to build a FAISS index and therefore perform a fallback strategy
            else:
                X_imputed[idx, missing] = self.global_fallbacks_[missing]

        return X_imputed
