from __future__ import annotations

from typing import Literal, Any
from lamin_utils import logger

import faiss
import numpy as np
from numpy import ndarray, dtype
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
        min_data_ratio: float = 0.25,
    ):
        """Initializes FaissImputer with specified parameters that are used for the imputation.

        Args:
            missing_values: The missing value to impute.
            n_neighbors: Number of neighbors to use for imputation.
            metric: Distance metric to use for neighbor search.
            strategy: Method to compute imputed values among neighbors.
                      The weighted strategy is similar to scikt-learn's implementation,
                      where closer neighbors have a higher influence on the imputation.
            index_factory: Description of the Faiss index type to build.
            min_data_ratio: The minimum (dimension 0) size of the FAISS index relative to the (dimension 0) size of the
                            dataset that will be used to train FAISS. Defaults to 0.25. See also `fit_transform`.
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
        self.X_full = None
        self.features_nan = None
        self.min_data_ratio = min_data_ratio
        self.warned_fallback = False
        self.warned_unsufficient_neighbors = False
        super().__init__()

    # @override
    def fit_transform(self, X: np.ndarray, y=None, **fit_params) -> ndarray[Any, dtype[Any]] | None:
        """Imputes missing values in the data using the fitted Faiss index. This imputation will be performed in place.
        This imputation will use self.min_data_ratio to check if the index is of sufficient (dimension 0) size to
        perform a qualitative KNN lookup. If not, it will temporarily exclude enough features to reach this threshold
        and try again. If an index still can't be built, it will use fallbacks values as defined by self.strategy.

        Args:
            X: Input data with potential missing values.
            y: Ignored, present for compatibility with sklearn's TransformerMixin.

        Returns:
            Data with imputed values as a NumPy array of the original data type.
        """
        self.X_full = np.asarray(X, dtype=np.float64) if not np.issubdtype(X.dtype, np.floating) else X
        if np.isnan(self.X_full).all(axis=0).any():
            raise ValueError("Features with only missing values cannot be handled.")

        # Prepare fallback values, used to prefill the query vectors nan´s
        # or as an imputation fallback if we can't build an index
        global_fallbacks_ = (
            np.nanmean(self.X_full, axis=0) if self.strategy in ["mean", "weighted"] else np.nanmedian(self.X_full, axis=0)
        )

        # We will need to impute all features having nan´s
        feature_indices_to_impute = [i for i in range(self.X_full.shape[1]) if np.isnan(self.X_full[:, i]).any()]

        # Now impute iteratively
        while feature_indices_to_impute:
            feature_indices_being_imputed, training_indices, training_data, index = self._fit_train_imputer(feature_indices_to_impute)

            # Use fallback data if we can't build an index and iterate again
            if index is None:
                self._warn_fallback()
                self.X_full[:, feature_indices_being_imputed] =  global_fallbacks_[feature_indices_being_imputed]
                continue

            # Extract the features from X that was used to train FAISS, and compute the sparseness matrix
            x_imputed = self.X_full[:, training_indices]
            x_imputed_missing_mask = np.isnan(x_imputed)

            # Iterate through the data to impute, ignoring already imputed rows
            sample_missing_mask = []
            for idx in np.where(x_imputed_missing_mask.any(axis=1))[0]:
                # Extract the sample
                sample_missing_mask = x_imputed_missing_mask[idx]
                sample = x_imputed[idx]

                # We need to prefill the query vector as FAISS doesn't accept nan´s
                sample[sample_missing_mask] = global_fallbacks_[training_indices][sample_missing_mask]
                missing_cols = np.where(sample_missing_mask)[0]

                # Call FAISS and retrieve data
                distances, indices = index.search(sample.reshape(1, -1), self.n_neighbors)
                assert len(indices[0]) == self.n_neighbors
                valid_indices = indices[0][indices[0] >= 0] # Filter out negative indices because they are FAISS error codes

                # FAISS couldn't find any neighbor, use fallback values, and go to next row
                if len(valid_indices) == 0:
                    self._warn_fallback()
                    x_imputed[idx, missing_cols] = sample[missing_cols]
                    continue

                # FAISS couldn't find the amount of requested neighbors, warn user and proceed
                if len(valid_indices) < self.n_neighbors:
                    if not self.warned_unsufficient_neighbors:
                        logger.warning(f"FAISS couldn't find all the requested neighbors. "
                                       f"This warning will be displayed only once.")
                        self.warned_unsufficient_neighbors = True

                # Apply strategy on neighbors data
                neighbors = training_data[valid_indices]
                if self.strategy == "weighted":
                    weights = 1 / (distances[0] + 1e-10)[:, np.newaxis]
                    neighbor_vals = neighbors[:, missing_cols]
                    weighted_sum = np.nansum(neighbor_vals * weights, axis=0)
                    weight_sum = np.nansum(weights, axis=0)
                    x_imputed[idx, missing_cols] = weighted_sum / weight_sum
                else:
                    func = np.nanmean if self.strategy == "mean" else np.nanmedian
                    x_imputed[idx, missing_cols] = func(neighbors[:, missing_cols], axis=0)

            # Transfer back to X
            # Features added by _fit_train_imputer for training purpose only are placed on the right
            # so we can just select the features on the left
            self.X_full[:, feature_indices_being_imputed] = x_imputed[:, np.arange(len(feature_indices_being_imputed))]

            # Remove the imputed features from the to-do list
            feature_indices_to_impute = [feature_indice for feature_indice in feature_indices_to_impute if feature_indice not in feature_indices_being_imputed]

        assert not np.isnan(self.X_full).any()
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
            # Train data features are those indexed by features_indices AND those already fully imputed in
            # the full X. Those indices are placed after the features to impute (ie on the right),
            # so it will be easy to filter them out later.
            train_indices = features_indices_to_impute + already_imputed_features_indices

            # Filter X with the features indices provided in column and not containing nan´s for rows
            x_subset = self.X_full[:, train_indices]
            x_non_missing = x_subset[~np.isnan(x_subset).any(axis=1)]

            # Check if we have enough data
            if x_non_missing.shape[0] >= self.X_full.shape[0] * self.min_data_ratio:
                # We have our list of features to impute!
                return features_indices_to_impute, train_indices, x_non_missing, self._train(x_non_missing)
            else:
                # One feature left, meaning we can't build an index
                if len(features_indices_to_impute) <= 1:
                    return features_indices, None, None, None

                # Remove the feature containing the largest amount of nan´s and iterate again
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

    def _train(self, x_train: np.ndarray) -> faiss.Index:
        index = faiss.index_factory(x_train.shape[1], self.index_factory)
        index.metric_type = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        index.train(x_train)
        index.add(x_train)
        return index

    def _warn_fallback(self):
        if not self.warned_fallback:
            logger.warning(f"Fallback data (as defined by passed strategy) were used. "
                           f"This warning will only be displayed once.")
            self.warned_fallback = True

