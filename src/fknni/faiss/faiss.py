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
        is_value_missing = np.isnan(X)

        for sample_index in np.where(is_value_missing.any(axis=1))[0]:
            sample_data = X[sample_index]
            missing_value_mask = is_value_missing[sample_index]
            missing_columns = np.where(missing_value_mask)[0]
            sample_data[missing_value_mask] = self.global_fallbacks_[missing_value_mask]

            distances, neighbor_indices = self.index_.search(sample_data.reshape(1, -1), self.n_neighbors)
            neighbors_data = X[neighbor_indices[0]]
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

            sample_data[missing_columns] = imputed_values
            X[sample_index] = sample_data

        return X
