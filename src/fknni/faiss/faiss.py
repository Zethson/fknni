from typing import Literal

import faiss
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class FaissImputer(BaseEstimator, TransformerMixin):
    """Imputer for completing missing values using Faiss."""

    def __init__(
        self,
        n_neighbors: int = 3,
        metric: Literal["l2", "ip"] = "l2",
        strategy: Literal["mean", "median"] = "mean",
        index_factory: str = "Flat",
    ):
        """Initializes FaissImputer with specified parameters.

        Args:
            n_neighbors: Number of neighbors to use for imputation.
            metric: Distance metric to use for neighbor search.
            strategy: Method to compute imputed values.
            index_factory: Description of the Faiss index type to build.
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
        X = check_array(X, dtype=np.float32, force_all_finite="allow-nan")

        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        if self.metric not in {"l2", "ip"}:
            raise ValueError("metric must be either 'l2' or 'ip'")
        if self.strategy not in {"mean", "median"}:
            raise ValueError("strategy must be either 'mean' or 'median'")

        mask = ~np.isnan(X).any(axis=1)
        X_non_missing = X[mask]

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
            Data with imputed values as a NumPy array.
        """
        X = check_array(X, dtype=np.float32, force_all_finite="allow-nan")
        check_is_fitted(self, "index_")
        X_imputed = np.array(X, copy=True)
        missing_mask = np.isnan(X_imputed)

        placeholder_values = (
            np.nanmean(X_imputed, axis=0) if self.strategy == "mean" else np.nanmedian(X_imputed, axis=0)
        )

        for sample_idx in np.where(missing_mask.any(axis=1))[0]:
            sample_row = X_imputed[sample_idx, :]
            sample_missing_cols = np.where(missing_mask[sample_idx])[0]
            sample_row[sample_missing_cols] = placeholder_values[sample_missing_cols]

            _, neighbor_indices = self.index_.search(sample_row.reshape(1, -1), self.n_neighbors)
            selected_values = X_imputed[neighbor_indices[0], :][:, sample_missing_cols]

            sample_row[sample_missing_cols] = (
                np.mean(selected_values, axis=0) if self.strategy == "mean" else np.median(selected_values, axis=0)
            )
            X_imputed[sample_idx, :] = sample_row

        return X_imputed
