from __future__ import annotations

from typing import Literal, Any, Iterable, Tuple, Callable

import faiss
import numpy as np
from numpy import ndarray, dtype, floating
from overrides import overrides
from sklearn.base import BaseEstimator, TransformerMixin

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
        self.min_data_ratio = 0
        self.X_full = None
        self.features_nan = None
        self.min_data_ratio = 0.25
        super().__init__()

    @overrides
    def fit_transform(self, X: np.ndarray, y=None, **fit_params) -> ndarray[Any, dtype[Any]] | None:
        """Imputes missing values in the data using the fitted Faiss index.

        Args:
            X: Input data with potential missing values.
            y: Ignored, present for compatibility with sklearn's TransformerMixin.

        Returns:
            Data with imputed values as a NumPy array of the original data type.
        """
        self.X_full = np.asarray(X, dtype=np.float32)
        if not np.isnan(self.X_full).any():
            return self.X_full
        if np.isnan(self.X_full).all(axis=0).any():
            raise ValueError("Features with all values missing cannot be handled.")

        # Prepare fallback values, used to prefill the query vectors nan´s
        # or as an imputation fallback if we can't build an index
        global_fallbacks_ = (
            np.nanmean(X, axis=0) if self.strategy in ["mean", "weighted"] else np.nanmedian(X, axis=0)
        )

        # We will need to impute all features having nan´s
        feature_indices_to_impute = [i for i in range(X.shape[1]) if np.isnan(X[:, i]).any()]

        # Now impute iteratively
        while feature_indices_to_impute:
            feature_indices_being_imputed, training_indices, training_data, index = self._fit_train_imputer(feature_indices_to_impute)

            # Can we proceed with a FAISS imputation?
            assert index is not None # Todo: Fallback here

            # Extract the features from X that was used to train FAISS, and compute the sparseness matrix
            X_imputed = self.X_full[::, training_indices]
            X_imputed_missing_mask = np.isnan(X_imputed)

            # Iterate through the data to impute, ignoring already imputed rows
            for idx in np.where(X_imputed_missing_mask.any(axis=1))[0]:
                # Extract the sample
                sample_missing_mask = X_imputed_missing_mask[idx]
                sample = X_imputed[idx]

                # We need to prefill the query vector as FAISS doesn't accept nan´s
                try:
                    sample[sample_missing_mask] = global_fallbacks_[feature_indices_being_imputed][sample_missing_mask]
                except:
                    print("ERROR")

                # Call FAISS and retrieve data
                distances, indices = index.search(sample.reshape(1, -1), self.n_neighbors)
                valid_indices = indices[0][indices[0] >= 0] # Filter out negative indices because it's a FAISS error code
                assert len(valid_indices) == self.n_neighbors # Todo: Assert or warning or exception?
                neighbors = training_data[valid_indices]

                # Apply strategy on neighbors
                missing_cols = np.where(sample_missing_mask)[0]
                if self.strategy == "weighted":
                    weights = 1 / (distances[0] + 1e-10)[:, np.newaxis]
                    neighbor_vals = neighbors[:, missing_cols]
                    weighted_sum = np.nansum(neighbor_vals * weights, axis=0)
                    weight_sum = np.nansum(weights, axis=0)
                    X_imputed[idx, missing_cols] = weighted_sum / weight_sum
                else:
                    func = np.nanmean if self.strategy == "mean" else np.nanmedian
                    X_imputed[idx, missing_cols] = func(neighbors[:, missing_cols], axis=0)

            # Transfer to X and remove the imputed features from the to-do list
            self.X_full[::, feature_indices_being_imputed] = X_imputed
            feature_indices_to_impute = [x for x in feature_indices_to_impute if x not in feature_indices_being_imputed]

        assert not np.isnan(self.X_full).any()  # TODO: What to do here? It shouldn't happen.
        return self.X_full

    def _fit_train_imputer(self, features_indices: list[int]) -> (list[int],
                                                                  list[int] | None,
                                                                  np.ndarray | None,
                                                                  faiss.Index | None):
        features_indices_to_impute = features_indices.copy()

        # See what features are already imputed
        already_imputed_features_indices = [i for i in range(self.X_full.shape[1])
                                            if not np.isnan(self.X_full[:, i]).any()]

        while True:
            # Do we have only one feature left? Then it's no point trying to build an index
            if len(features_indices_to_impute) <= 1:
                return features_indices_to_impute, None, None

            # Train data features are those indexed by features_indices AND those already fully imputed in
            # the full X
            train_indices = features_indices_to_impute + already_imputed_features_indices

            # Filter X with the features indices provided in column and not containing nan´s for rows
            X_subset = self.X_full[::, train_indices]
            X_non_missing = X_subset[~np.isnan(X_subset).any(axis=1)]

            # Check if we have enough data
            if X_non_missing.shape[0] >= self.X_full.shape[0] * self.min_data_ratio:
                # Yes, we have our list of features to impute
                return features_indices_to_impute, train_indices, X_non_missing, self._train(X_non_missing)
            else:
                # No, remove the feature containing the largest amount of nan´s and iterate again
                for value in self._features_indices_sorted_descending_on_nan():
                    if value in features_indices_to_impute:
                        features_indices_to_impute.remove(value)
                        break

    def _features_indices_sorted_descending_on_nan(self) -> list[int]:
        if self.features_nan is None:
            self.features_nan = sorted(
                (i for i in range(self.X_full.shape[1]) if np.isnan(self.X_full[:, i]).sum() > 0),
                key=lambda i: np.isnan(self.X_full[:, i]).sum(),
                reverse=True
            )

        return self.features_nan

    def _train(self, X_train: np.ndarray) -> faiss.Index:
        index = faiss.index_factory(X_train.shape[1], self.index_factory)
        index.metric_type = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        index.train(X_train)
        index.add(X_train)
        return index


