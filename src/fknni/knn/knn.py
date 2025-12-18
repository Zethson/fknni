from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import array_api_compat
import numpy as np
from numpy import dtype
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from importlib.metadata import distributions

    import faiss

    HAS_FAISS_GPU = any(d.metadata["Name"].startswith("faiss-gpu") for d in distributions())
except ImportError:
    raise ImportError("faiss-cpu or faiss-gpu required") from None

try:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as cuMLNearestNeighbors

    HAS_CUML = True
except ImportError:
    HAS_CUML = False
    cp = None

if TYPE_CHECKING:
    import cupy as cp

ArrayT = TypeVar("ArrayT", np.ndarray, "cp.ndarray")


class NNIndex(Protocol):
    """Protocol for nearest neighbor index implementations."""

    def search(self, query: ArrayT, k: int) -> tuple[ArrayT, ArrayT]:
        """Search for k nearest neighbors.

        Args:
            query: Query vectors to search for.
            k: Number of neighbors to return.

        Returns:
            Tuple of (distances, indices) arrays.
        """
        ...


class CuMLIndexWrapper:
    """Wrapper around cuML NearestNeighbors to match FAISS search interface."""

    def __init__(self, model: cuMLNearestNeighbors, X_train: cp.ndarray):
        """Initialize wrapper with fitted cuML model.

        Args:
            model: Fitted cuML NearestNeighbors model.
            X_train: Training data used to fit the model.
        """
        self._model = model
        self._X_train = X_train

    def search(self, query: cp.ndarray, k: int) -> tuple[cp.ndarray, cp.ndarray]:
        """Search for k nearest neighbors using cuML.

        Args:
            query: Query vectors to search for.
            k: Number of neighbors to return.

        Returns:
            Tuple of (distances, indices) arrays.
        """
        distances, indices = self._model.kneighbors(query, n_neighbors=k)
        return distances, indices


class FastKNNImputer(BaseEstimator, TransformerMixin):
    """Imputer for completing missing values using Faiss or cuML, incorporating weighted averages based on distance.

    Supports both numpy arrays (using FAISS) and cupy arrays (using cuML) for GPU-accelerated imputation.
    When cupy arrays are passed, all computations stay on GPU.
    """

    def __init__(
        self,
        missing_values: int | float | str | None = np.nan,
        n_neighbors: int = 5,
        *,
        metric: Literal["l2", "ip"] = "l2",
        strategy: Literal["mean", "median", "weighted"] = "mean",
        index_factory: str = "Flat",
        min_data_ratio: float = 0.25,
        temporal_mode: Literal["flatten", "per_variable"] = "flatten",
        use_gpu: bool = False,
    ):
        """Initializes FaissImputer with specified parameters that are used for the imputation.

        Args:
            missing_values: The missing value to impute.
            n_neighbors: Number of neighbors to use for imputation.
            metric: Distance metric to use for neighbor search. 'l2' for Euclidean, 'ip' for inner product.
            strategy: Method to compute imputed values among neighbors.
                      The weighted strategy is similar to scikit-learn's implementation,
                      where closer neighbors have a higher influence on the imputation.
            index_factory: Description of the Faiss index type to build (ignored for cupy arrays).
            min_data_ratio: The minimum (dimension 0) size of the FAISS index relative to the (dimension 0) size of the
                            dataset that will be used to train FAISS. Defaults to 0.25. See also `fit_transform`.
            temporal_mode: How to handle 3D temporal data. 'flatten' treats all (variable, timestep) pairs as
                       independent features (fast but allows temporal leakage).
                       'per_variable' imputes each variable independently across time (slower but respects temporal causality).
            use_gpu: Whether to train FAISS using GPU (only applies to numpy arrays, cupy arrays always use GPU via cuML).
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if strategy not in {"mean", "median", "weighted"}:
            raise ValueError("Unknown strategy. Choose one of 'mean', 'median', 'weighted'")
        if temporal_mode not in {"flatten", "per_variable"}:
            raise ValueError("Unknown temporal_mode. Choose one of 'flatten', 'per_variable'")

        self.use_gpu = use_gpu
        if use_gpu and not HAS_FAISS_GPU:
            raise ValueError("use_gpu=True requires faiss-gpu package, install with: pip install faiss-gpu")

        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.strategy = strategy
        self.index_factory = index_factory
        self.temporal_mode = temporal_mode
        self.X_full: np.ndarray | cp.ndarray | None = None
        self.features_nan: list[int] | None = None
        self.min_data_ratio = min_data_ratio
        self.warned_fallback = False
        self.warned_unsufficient_neighbors = False
        super().__init__()

    def fit_transform(  # noqa: D417
        self, X: np.ndarray | cp.ndarray, y: np.ndarray | None = None, **fit_params
    ) -> np.ndarray[Any, dtype[Any]] | cp.ndarray:
        """Imputes missing values in the data using the fitted Faiss/cuML index. This imputation will be performed in place.

        This imputation will use `min_data_ratio` to check if the index is of sufficient (dimension 0) size to perform a qualitative KNN lookup.
        If not, it will temporarily exclude enough features to reach this threshold and try again.
        If an index still can't be built, it will use fallbacks values as defined by self.strategy.

        For cupy arrays, computation stays entirely on GPU using cuML's NearestNeighbors.

        Args:
            X: Input data with potential missing values. Can be 2D (samples × features) or 3D (samples × features × timesteps).
               Accepts numpy arrays (uses FAISS) or cupy arrays (uses cuML).
            y: Ignored, present for compatibility with sklearn's TransformerMixin.

        Returns:
            Data with imputed values. Returns same array type as input (numpy or cupy).
        """
        original_shape = X.shape
        xp = array_api_compat.array_namespace(X)

        if X.ndim == 3 and self.temporal_mode == "per_variable":
            n_obs, n_vars, n_t = X.shape
            result = xp.empty_like(X, dtype=xp.float64)
            for var_idx in range(n_vars):
                X_slice = X[:, var_idx, :]
                result[:, var_idx, :] = self._impute_2d(X_slice)
            return result

        if X.ndim == 3:
            n_obs, n_vars, n_t = X.shape
            X = X.reshape(n_obs, n_vars * n_t)

        result = self._impute_2d(X)

        if len(original_shape) == 3:
            result = result.reshape(original_shape)

        return result

    def _impute_2d(self, X: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """Impute missing values in a 2D array.

        Args:
            X: 2D input array with potential missing values.

        Returns:
            Array with imputed values, same type as input.
        """
        xp = array_api_compat.array_namespace(X)
        self.X_full = xp.asarray(X, dtype=xp.float64) if not xp.issubdtype(X.dtype, xp.floating) else X

        if xp.isnan(self.X_full).all(axis=0).any():
            raise ValueError("Features with only missing values cannot be handled.")

        # Prepare fallback values, used to prefill the query vectors nan´s
        # or as an imputation fallback if we can't build an index
        global_fallbacks_ = (
            xp.nanmean(self.X_full, axis=0)
            if self.strategy in ["mean", "weighted"]
            else xp.nanmedian(self.X_full, axis=0)
        )

        # We will need to impute all features having nan´s
        feature_indices_to_impute = [i for i in range(self.X_full.shape[1]) if xp.isnan(self.X_full[:, i]).any()]

        # Now impute iteratively
        while feature_indices_to_impute:
            feature_indices_being_imputed, training_indices, training_data, index = self._fit_train_imputer(
                feature_indices_to_impute
            )

            # Use fallback data if we can't build an index and iterate again
            if index is None:
                self._warn_fallback()
                self.X_full[:, feature_indices_being_imputed] = global_fallbacks_[feature_indices_being_imputed]
                continue

            # Extract the features from X that was used to train the index, and compute the sparseness matrix
            x_imputed = self.X_full[:, training_indices]
            x_imputed_missing_mask = xp.isnan(x_imputed)

            row_indices = xp.where(x_imputed_missing_mask.any(axis=1))[0]
            n_rows = row_indices.shape[0]

            if n_rows == 0:
                # No rows to impute, transfer back and continue
                n_imputed = len(feature_indices_being_imputed)
                self.X_full[:, feature_indices_being_imputed] = x_imputed[:, xp.arange(n_imputed)]
                feature_indices_to_impute = [
                    fi for fi in feature_indices_to_impute if fi not in feature_indices_being_imputed
                ]
                continue

            # Batch prefill: replace NaNs with fallback values for all query rows
            queries = x_imputed[row_indices].copy()
            query_missing_mask = x_imputed_missing_mask[row_indices]
            fallbacks_for_training = global_fallbacks_[training_indices]
            queries = xp.where(query_missing_mask, fallbacks_for_training, queries)

            # Batch search: single call for all queries
            distances, indices = index.search(queries, self.n_neighbors)

            # Check for invalid indices (FAISS returns -1 for not found)
            valid_mask = indices >= 0
            any_invalid = ~valid_mask.all()

            if any_invalid:
                if not valid_mask.any():
                    # No valid neighbors found at all, use fallback
                    self._warn_fallback()
                    n_imputed = len(feature_indices_being_imputed)
                    self.X_full[:, feature_indices_being_imputed] = x_imputed[:, xp.arange(n_imputed)]
                    feature_indices_to_impute = [
                        fi for fi in feature_indices_to_impute if fi not in feature_indices_being_imputed
                    ]
                    continue

                if not self.warned_unsufficient_neighbors:
                    warnings.warn(
                        "Couldn't find all requested neighbors for some samples. This warning will be displayed only once.",
                        stacklevel=2,
                    )
                    self.warned_unsufficient_neighbors = True

            # Retrieve all neighbors: shape (n_rows, n_neighbors, n_features)
            # Clamp negative indices to 0 for gathering, then mask later
            safe_indices = xp.where(valid_mask, indices, 0)
            all_neighbors = training_data[safe_indices]

            # Compute imputed values based on strategy
            if self.strategy == "weighted":
                # Weights: (n_rows, n_neighbors, 1)
                weights = 1.0 / (distances + 1e-10)
                weights = xp.where(valid_mask, weights, 0.0)
                weights = weights[:, :, xp.newaxis]

                # Weighted sum across neighbors
                weighted_vals = all_neighbors * weights
                weighted_sum = weighted_vals.sum(axis=1)
                weight_sum = weights.sum(axis=1)
                imputed_rows = weighted_sum / (weight_sum + 1e-10)
            elif self.strategy == "mean":
                # Mask invalid neighbors with NaN, then nanmean
                all_neighbors = xp.where(valid_mask[:, :, xp.newaxis], all_neighbors, xp.nan)
                imputed_rows = xp.nanmean(all_neighbors, axis=1)
            else:  # median
                all_neighbors = xp.where(valid_mask[:, :, xp.newaxis], all_neighbors, xp.nan)
                imputed_rows = xp.nanmedian(all_neighbors, axis=1)

            # Only update the missing positions
            x_imputed[row_indices] = xp.where(query_missing_mask, imputed_rows, x_imputed[row_indices])

            # Transfer back to X
            # Features added by _fit_train_imputer for training purpose only are placed on the right
            # so we can just select the features on the left
            n_imputed = len(feature_indices_being_imputed)
            self.X_full[:, feature_indices_being_imputed] = x_imputed[:, xp.arange(n_imputed)]

            # Remove the imputed features from the to-do list
            feature_indices_to_impute = [
                fi for fi in feature_indices_to_impute if fi not in feature_indices_being_imputed
            ]

        assert not xp.isnan(self.X_full).any()

        return self.X_full

    def _fit_train_imputer(
        self, features_indices: Sequence[int]
    ) -> tuple[list[int], list[int] | None, np.ndarray | cp.ndarray | None, NNIndex | None]:
        """Build and train the nearest neighbor index for imputation.

        Args:
            features_indices: Indices of features that need imputation.

        Returns:
            Tuple of (features_to_impute, training_indices, training_data, index).
            If index cannot be built, returns (features_indices, None, None, None).
        """
        xp = array_api_compat.array_namespace(self.X_full)
        features_indices_to_impute = list(features_indices)

        already_imputed_features_indices = [
            i for i in range(self.X_full.shape[1]) if not xp.isnan(self.X_full[:, i]).any()
        ]

        while True:
            train_indices = features_indices_to_impute + already_imputed_features_indices
            x_subset = self.X_full[:, train_indices]
            non_missing_rows = ~xp.isnan(x_subset).any(axis=1)
            x_non_missing = x_subset[non_missing_rows]

            # Check if we have enough data
            if x_non_missing.shape[0] >= self.X_full.shape[0] * self.min_data_ratio:
                # We have our list of features to impute!
                return (
                    features_indices_to_impute,
                    train_indices,
                    x_non_missing,
                    self._train(x_non_missing),
                )
            else:
                # One feature left, meaning we can't build an index
                if len(features_indices_to_impute) <= 1:
                    return list(features_indices), None, None, None

                # Remove the feature containing the largest amount of nan´s and iterate again
                for value in self._features_indices_sorted_descending_on_nan():
                    if value in features_indices_to_impute:
                        features_indices_to_impute.remove(value)
                        break

    def _features_indices_sorted_descending_on_nan(self) -> list[int]:
        """Get feature indices sorted by number of NaN values in descending order.

        Returns:
            List of feature indices with NaN values, sorted by NaN count (highest first).
        """
        if self.features_nan is None:
            xp = array_api_compat.array_namespace(self.X_full)
            nan_counts = [(i, int(xp.isnan(self.X_full[:, i]).sum())) for i in range(self.X_full.shape[1])]
            self.features_nan = [i for i, c in sorted(nan_counts, key=lambda x: x[1], reverse=True) if c > 0]
        return self.features_nan

    def _train(self, x_train: np.ndarray | cp.ndarray) -> NNIndex:
        """Train the nearest neighbor index.

        Automatically selects FAISS for numpy arrays or cuML for cupy arrays.

        Args:
            x_train: Training data for building the index.

        Returns:
            Trained nearest neighbor index (FAISS Index or CuMLIndexWrapper).
        """
        if array_api_compat.is_cupy_array(x_train):
            return self._train_cuml(x_train)
        return self._train_faiss(x_train)

    def _train_faiss(self, x_train: np.ndarray) -> faiss.Index:
        """Train a FAISS index for nearest neighbor search.

        Args:
            x_train: Training data as numpy array.

        Returns:
            Trained FAISS index.
        """
        index = faiss.index_factory(x_train.shape[1], self.index_factory)
        index.metric_type = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.train(x_train)
        index.add(x_train)
        return index

    def _train_cuml(self, x_train: cp.ndarray) -> CuMLIndexWrapper:
        """Train a cuML NearestNeighbors model for GPU-native nearest neighbor search.

        Args:
            x_train: Training data as cupy array.

        Returns:
            CuMLIndexWrapper with trained model.

        Raises:
            ImportError: If cuML is not installed.
        """
        if not HAS_CUML:
            raise ImportError("cuML is required for GPU array imputation. Install with: pip install cuml-cu12")
        metric = "euclidean" if self.metric == "l2" else "cosine"
        model = cuMLNearestNeighbors(n_neighbors=self.n_neighbors, metric=metric)
        model.fit(x_train)
        return CuMLIndexWrapper(model, x_train)

    def _warn_fallback(self):
        """Emit a warning when fallback values are used for imputation."""
        if not self.warned_fallback:
            warnings.warn(
                "Fallback data (as defined by passed strategy) were used. This warning will only be displayed once.",
                stacklevel=2,
            )
            self.warned_fallback = True


class FaissImputer(FastKNNImputer):
    """Deprecated: Use FastKNNImputer instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FaissImputer is deprecated, use FastKNNImputer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
