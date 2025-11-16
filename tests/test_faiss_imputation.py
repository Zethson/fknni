from typing import Any
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from fknni.faiss.faiss import FaissImputer


@pytest.fixture
def simple_test_df(rng):
    data = pd.DataFrame(rng.integers(0, 100, size=(10, 5)), columns=list("ABCDE"))
    data_missing = data.copy()
    indices = [(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])]
    rng.shuffle(indices)
    for i, j in indices[:5]:
        data_missing.iat[i, j] = np.nan
    return data.to_numpy(), data_missing.to_numpy()


@pytest.fixture
def regression_dataset(rng):
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X_missing = X.copy()
    indices = [(i, j) for i in range(X.shape[0]) for j in range(X.shape[1])]
    rng.shuffle(indices)
    for i, j in indices[:50]:
        X_missing[i, j] = np.nan
    return X, X_missing, y


def _base_check_imputation(
    data_original: np.ndarray,
    data_imputed: np.ndarray,
):
    """Provides the following base checks:
    - Imputation doesn't leave any NaN behind
    - Imputation doesn't modify any data that wasn't NaN

    Args:
        data_before_imputation: Dataset before imputation
        data_after_imputation: Dataset after imputation

    Raises:
        AssertionError: If any of the checks fail.
    """
    if data_original.shape != data_imputed.shape:
        raise AssertionError("The shapes of the two datasets do not match")

    # Ensure no NaN remains in the imputed dataset
    if np.isnan(data_imputed).any():
        raise AssertionError("NaN found in imputed columns of layer_after.")

    # Ensure imputation does not alter non-NaN values in the imputed columns
    imputed_non_nan_mask = ~np.isnan(data_original)
    if not _are_ndarrays_equal(data_original[imputed_non_nan_mask], data_imputed[imputed_non_nan_mask]):
        raise AssertionError("Non-NaN values in imputed columns were modified.")

    # If reaching here: all checks passed
    return


def test_median_imputation(simple_test_df):
    """Tests if median imputation successfully fills all NaN values"""
    data, data_missing = simple_test_df
    data_original = data_missing.copy()
    FaissImputer(n_neighbors=5, strategy="median").fit_transform(data_missing)
    _base_check_imputation(data_original, data_missing)


def test_mean_imputation(simple_test_df):
    """Tests if mean imputation successfully fills all NaN values"""
    data, data_missing = simple_test_df
    data_original = data_missing.copy()
    FaissImputer(n_neighbors=5, strategy="mean").fit_transform(data_missing)
    _base_check_imputation(data_original, data_missing)


def test_imputer_with_no_missing_values(simple_test_df):
    """Tests if imputer preserves data when no values are missing"""
    data, _ = simple_test_df
    data_original = data.copy()
    FaissImputer(n_neighbors=5, strategy="median").fit_transform(data)
    _base_check_imputation(data_original, data)


def test_imputer_with_all_nan_column(rng):
    """Tests if imputer raises error when entire column is NaN"""
    data = rng.uniform(0, 100, size=(10, 5))
    data_missing = data.copy()
    data_missing[:, 2] = np.nan
    with pytest.raises(ValueError):
        FaissImputer(n_neighbors=5).fit_transform(data_missing)


def test_imputer_with_all_nan_row(rng):
    """Tests if imputer handles all-NaN rows by imputing them"""
    data = rng.uniform(0, 100, size=(10, 5))
    data[3, :] = np.nan
    data_missing = data.copy()
    data_original = data.copy()

    FaissImputer(n_neighbors=5).fit_transform(data_missing)

    _base_check_imputation(data_original, data_missing)


def test_imputer_different_n_neighbors(simple_test_df):
    """Tests if different n_neighbors values produce different results"""
    data, data_missing = simple_test_df
    data_original = data_missing.copy()
    imputer_3 = data_missing.copy()
    imputer_7 = data_missing.copy()
    FaissImputer(n_neighbors=3).fit_transform(imputer_3)
    FaissImputer(n_neighbors=7).fit_transform(imputer_7)
    _base_check_imputation(data_original, imputer_3)
    _base_check_imputation(data_original, imputer_7)
    assert not np.array_equal(imputer_3, imputer_7)


def test_regression_imputation(regression_dataset):
    """Tests if imputed data maintains predictive power in regression task"""
    X, X_missing, y = regression_dataset
    X_original = X_missing.copy()
    FaissImputer(n_neighbors=5).fit_transform(X_missing)
    _base_check_imputation(X_original, X_missing)

    from sklearn.linear_model import LinearRegression

    model_orig = LinearRegression().fit(X, y)
    model_imputed = LinearRegression().fit(X_missing, y)

    score_orig = model_orig.score(X, y)
    score_imputed = model_imputed.score(X_missing, y)
    assert abs(score_orig - score_imputed) < 0.1


def test_invalid_strategy():
    """Tests if imputer raises error for invalid strategy"""
    with pytest.raises(ValueError):
        FaissImputer(strategy="invalid")


def test_invalid_n_neighbors():
    """Tests if imputer raises error for invalid n_neighbors values"""
    with pytest.raises(ValueError):
        FaissImputer(n_neighbors=0)
    with pytest.raises(ValueError):
        FaissImputer(n_neighbors=-1)


def test_no_full_rows():
    """Tests whether a dataset with no full rows can be imputed."""
    arr = np.array(
        [
            [np.nan, np.nan, 27.81265195, 89.7247631, np.nan],
            [np.nan, np.nan, 63.35486059, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 64.19054628],
            [np.nan, np.nan, 10.16766562, np.nan, np.nan],
            [91.24215742, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 97.29442362, np.nan, np.nan, np.nan],
        ]
    )
    arr_original = arr.copy()
    FaissImputer(n_neighbors=1).fit_transform(arr)
    _base_check_imputation(arr_original, arr)


def _are_ndarrays_equal(arr1: np.ndarray, arr2: np.ndarray) -> np.bool_:
    """Check if two arrays are equal member-wise.

    Note: Two NaN are considered equal.

    Args:
        arr1: First array to compare
        arr2: Second array to compare

    Returns:
        True if the two arrays are equal member-wise
    """
    return np.all(np.equal(arr1, arr2, dtype=object) | ((arr1 != arr1) & (arr2 != arr2)))
