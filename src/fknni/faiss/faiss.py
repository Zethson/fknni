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

        super().__init__()
        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.strategy = strategy
        self.index_factory = index_factory

    def fit(self, X: np.ndarray | pd.DataFrame, *, y: np.ndarray | None = None) -> FaissImputer:
        """Fits the FaissImputer to the provided data.

        Args:
            X: Input data with potential missing values.
            y: Ignored, present for compatibility with sklearn's TransformerMixin.
        """
        X = np.asarray(X, dtype=np.float32)
        if isinstance(X, pd.DataFrame):
            X = X.replace(self.missing_values, np.nan).values
        else:
            X = np.where(X == self.missing_values, np.nan, X)

        if np.isnan(X).all(axis=0).any():
            raise ValueError("Features with all values missing cannot be handled.")

        self.global_fallbacks_ = (
            np.nanmean(X, axis=0) if self.strategy in ["mean", "weighted"] else np.nanmedian(X, axis=0)
        )

        non_missing_mask = ~np.isnan(X).any(axis=1)
        if non_missing_mask.sum() > max(2, X.shape[1] // 2):  # Normal case
            X_complete = X[non_missing_mask]
            index = faiss.index_factory(X.shape[1], self.index_factory)
            index.metric_type = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
            index.train(X_complete)
            index.add(X_complete)
            self.index_ = index
            self.X_train_ = X_complete
            self.is_extremely_sparse_ = False
        else:  # Extremely sparse case
            self.is_extremely_sparse_ = True
            self.X_train_ = X

        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Imputes missing values in the data using the fitted Faiss index.

        Args:
            X: Data with missing values to impute.

        Returns:
            Data with imputed values as a NumPy array of the original data type.
        """
        X = check_array(X, dtype=np.float32, ensure_all_finite="allow-nan")
        check_is_fitted(self)

        if self.is_extremely_sparse_:
            return self._transform_sparse(X)
        return self._transform_normal(X)

    def _transform_sparse(self, X):
        X_imputed = X.copy()
        is_missing = np.isnan(X_imputed)
        X_imputed[is_missing] = np.take(self.global_fallbacks_, np.where(is_missing)[1])
        return X_imputed

    def _transform_normal(self, X):
        X_imputed = X.copy()
        is_value_missing = np.isnan(X)

        for sample_index in np.where(is_value_missing.any(axis=1))[0]:
            sample_data = X[sample_index]
            missing_value_mask = is_value_missing[sample_index]
            missing_columns = np.where(missing_value_mask)[0]
            sample_data[missing_value_mask] = self.global_fallbacks_[missing_value_mask]

            distances, neighbor_indices = self.index_.search(sample_data.reshape(1, -1), self.n_neighbors)
            neighbors_data = self.X_train_[neighbor_indices[0]]
            neighbor_values = neighbors_data[:, missing_columns]

            if self.strategy == "mean":
                imputed_values = np.nanmean(neighbor_values, axis=0)
            elif self.strategy == "median":
                imputed_values = np.nanmedian(neighbor_values, axis=0)
            elif self.strategy == "weighted":
                weights = 1 / (distances[0] + 1e-10)
                adjusted_weights = weights[:, np.newaxis]
                weighted_totals = np.nansum(neighbor_values * adjusted_weights, axis=0)
                total_weights = np.nansum(adjusted_weights, axis=0)
                imputed_values = weighted_totals / total_weights

                no_weight_conditions = total_weights == 0
                if no_weight_conditions.any():
                    fallback_values = self.global_fallbacks_[missing_columns][no_weight_conditions]
                    imputed_values[no_weight_conditions] = fallback_values

            X_imputed[sample_index, missing_columns] = imputed_values

        return X_imputed
