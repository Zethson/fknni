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
        n_neighbors: int = 5,
        metric: Literal["l2", "ip"] = "l2",
        strategy: Literal["mean", "median", "weighted"] = "weighted",
        index_factory: str = "Flat",
    ):
        """Initializes FaissImputer with specified parameters that are used for the imputation.

        Args:
            n_neighbors: Number of neighbors to use for imputation. Defaults to 5.
            metric: Distance metric to use for neighbor search. Defaults to 'l2'.
            strategy: Method to compute imputed values among neighbors.
                      The weighted strategy is similar to scikt-learn's implementation,
                      where closer neighbors have a higher influence on the imputation.
            index_factory: Description of the Faiss index type to build. Defaults to 'Flat'.
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.strategy = strategy
        self.index_factory = index_factory

    def fit(self, X: np.ndarray | pd.DataFrame, *, y: np.ndarray | None = None) -> "FaissImputer":
        """Fits the FaissImputer to the provided data.

        Args:
            X: Input data with potential missing values.
            y: Ignored, present for compatibility with sklearn's TransformerMixin.

        Raises:
          ValueError: If any parameters are set to an invalid value.
        """
        X = np.asarray(X, dtype=np.float32)
        if np.isnan(X).all(axis=0).any():
            raise ValueError("Features with all values missing cannot be handled.")

        mask = ~np.isnan(X).any(axis=1)
        X_non_missing = X[mask]

        index = faiss.index_factory(X_non_missing.shape[1], self.index_factory)
        index.metric_type = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        index.train(X_non_missing)
        index.add(X_non_missing)

        self.index_ = index
        self.global_fallbacks_ = (
            np.nanmean(X, axis=0) if self.strategy in ["mean", "weighted"] else np.nanmedian(X, axis=0)
        )

        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Imputes missing values in the data using the fitted Faiss index.

        Args:
            X: Data with missing values to impute. Expected to be either a NumPy array or a pandas DataFrame.

        Returns:
            Data with imputed values as a NumPy array of the original data type.
        """
        X = check_array(X, dtype=np.float32, force_all_finite="allow-nan")
        check_is_fitted(self)
        missing_mask = np.isnan(X)

        for sample_idx in np.where(missing_mask.any(axis=1))[0]:
            sample_row = X[sample_idx]
            sample_missing_mask = missing_mask[sample_idx]
            sample_missing_cols = np.where(sample_missing_mask)[0]
            sample_row[sample_missing_mask] = self.global_fallbacks_[sample_missing_mask]

            distances, neighbor_indices = self.index_.search(sample_row.reshape(1, -1), self.n_neighbors)
            selected_vectors = X[neighbor_indices[0]]
            selected_values = selected_vectors[:, sample_missing_cols]

            if self.strategy in ["mean", "median"]:
                if self.strategy == "mean":
                    imputed_values = np.nanmean(selected_values, axis=0)
                else:
                    imputed_values = np.nanmedian(selected_values, axis=0)
            elif self.strategy == "weighted":
                weights = 1 / (distances[0] + 1e-10)
                valid_weights = weights[:, np.newaxis]
                weighted_sums = np.nansum(selected_values * valid_weights, axis=0)
                total_weights = np.nansum(valid_weights, axis=0)
                imputed_values = weighted_sums / total_weights
                fallback_condition = total_weights == 0
                if fallback_condition.any():
                    fallback_values = self.global_fallbacks_[sample_missing_cols][fallback_condition]
                    imputed_values[fallback_condition] = fallback_values

            sample_row[sample_missing_cols] = imputed_values
            X[sample_idx] = sample_row

        return X
