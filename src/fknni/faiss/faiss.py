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
        X = check_array(X, force_all_finite="allow-nan")
        self.input_dtype_ = X.dtype

        # Handle missing values for indexing
        self.means_ = np.nanmean(X, axis=0)  # Store means for missing value handling
        X_non_missing = np.where(np.isnan(X), self.means_, X).astype(np.float32)

        index = faiss.index_factory(
            X_non_missing.shape[1],
            self.index_factory,
            faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT,
        )
        index.train(X_non_missing)
        index.add(X_non_missing)
        self.index_ = index

        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Imputes missing values in the data using the fitted Faiss index.

        Args:
            X: Data with missing values to impute. Expected to be either a NumPy array or a pandas DataFrame.

        Returns:
            Data with imputed values as a NumPy array of the original data type.
        """
        X = check_array(X, force_all_finite="allow-nan")
        check_is_fitted(self, "index_")
        X_imputed = np.array(X, dtype=np.float32)  # Use float32 for processing
        missing_mask = np.isnan(X_imputed)

        X_filled = np.where(missing_mask, self.means_, X_imputed)

        for sample_idx in np.where(missing_mask.any(axis=1))[0]:
            sample_row_filled = X_filled[sample_idx]
            sample_missing_cols = np.where(missing_mask[sample_idx])[0]

            distances, neighbor_indices = self.index_.search(sample_row_filled.reshape(1, -1), self.n_neighbors)
            neighbors = X_filled[neighbor_indices[0]]

            for col in sample_missing_cols:
                valid_neighbors = neighbors[:, col][~np.isnan(neighbors[:, col])]
                valid_distances = distances[0, : len(valid_neighbors)]

                if len(valid_neighbors) < self.n_neighbors:
                    if len(valid_neighbors) == 0:
                        imputed_value = self.means_[col]
                    else:
                        if self.strategy in {"mean", "weighted"}:
                            weights = (
                                1 / (1 + valid_distances)
                                if self.strategy == "weighted"
                                else np.ones_like(valid_distances)
                            )
                            imputed_value = np.average(valid_neighbors, weights=weights)
                        elif self.strategy == "median":
                            imputed_value = np.median(valid_neighbors)
                else:
                    if self.strategy == "mean":
                        imputed_value = np.mean(valid_neighbors)
                    elif self.strategy == "median":
                        imputed_value = np.median(valid_neighbors)
                    elif self.strategy == "weighted":
                        small_constant = 1e-10  # Small constant to prevent division by zero
                        weights = 1 / (valid_distances + small_constant)
                        imputed_value = np.average(valid_neighbors, weights=weights)

                X_imputed[sample_idx, col] = imputed_value

        return X_imputed.astype(self.input_dtype_)  # Cast back to the original input dtype
