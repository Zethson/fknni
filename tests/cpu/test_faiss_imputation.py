import numpy as np
import pytest
from sklearn.datasets import make_regression
from tests.compare_predictions import _base_check_imputation

from fknni.knn.knn import FastKNNImputer


@pytest.fixture
def regression_dataset(rng):
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X_missing = X.copy()
    indices = [(i, j) for i in range(X.shape[0]) for j in range(X.shape[1])]
    rng.shuffle(indices)
    for i, j in indices[:50]:
        X_missing[i, j] = np.nan
    return X, X_missing, y


def test_median_imputation(simple_test_df):
    """Tests if median imputation successfully fills all NaN values"""
    data, data_missing = simple_test_df
    data_original = data_missing.copy()
    FastKNNImputer(n_neighbors=5, strategy="median").fit_transform(data_missing)
    _base_check_imputation(data_original, data_missing)


def test_mean_imputation(simple_test_df):
    """Tests if mean imputation successfully fills all NaN values"""
    data, data_missing = simple_test_df
    data_original = data_missing.copy()
    FastKNNImputer(n_neighbors=5, strategy="mean").fit_transform(data_missing)
    _base_check_imputation(data_original, data_missing)


def test_imputer_with_no_missing_values(simple_test_df):
    """Tests if imputer preserves data when no values are missing"""
    data, _ = simple_test_df
    data_original = data.copy()
    FastKNNImputer(n_neighbors=5, strategy="median").fit_transform(data)
    _base_check_imputation(data_original, data)


def test_imputer_with_all_nan_column(rng):
    """Tests if imputer raises error when entire column is NaN"""
    data = rng.uniform(0, 100, size=(10, 5))
    data_missing = data.copy()
    data_missing[:, 2] = np.nan
    with pytest.raises(ValueError):
        FastKNNImputer(n_neighbors=5).fit_transform(data_missing)


def test_imputer_with_all_nan_row(rng):
    """Tests if imputer handles all-NaN rows by imputing them"""
    data = rng.uniform(0, 100, size=(10, 5))
    data[3, :] = np.nan
    data_missing = data.copy()
    data_original = data.copy()

    FastKNNImputer(n_neighbors=5).fit_transform(data_missing)

    _base_check_imputation(data_original, data_missing)


def test_imputer_different_n_neighbors(simple_test_df):
    """Tests if different n_neighbors values produce different results"""
    data, data_missing = simple_test_df
    data_original = data_missing.copy()
    imputer_3 = data_missing.copy()
    imputer_7 = data_missing.copy()
    FastKNNImputer(n_neighbors=3).fit_transform(imputer_3)
    FastKNNImputer(n_neighbors=7).fit_transform(imputer_7)
    _base_check_imputation(data_original, imputer_3)
    _base_check_imputation(data_original, imputer_7)
    assert not np.array_equal(imputer_3, imputer_7)


def test_regression_imputation(regression_dataset):
    """Tests if imputed data maintains predictive power in regression task"""
    X, X_missing, y = regression_dataset
    X_original = X_missing.copy()
    FastKNNImputer(n_neighbors=5).fit_transform(X_missing)
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
        FastKNNImputer(strategy="invalid")


def test_invalid_n_neighbors():
    """Tests if imputer raises error for invalid n_neighbors values"""
    with pytest.raises(ValueError):
        FastKNNImputer(n_neighbors=0)
    with pytest.raises(ValueError):
        FastKNNImputer(n_neighbors=-1)


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
    FastKNNImputer(n_neighbors=1).fit_transform(arr)
    _base_check_imputation(arr_original, arr)


def test_3d_flatten_imputation(rng):
    """Tests if 3D imputation with flatten mode successfully fills all NaN values"""
    data_3d = rng.uniform(0, 100, size=(10, 5, 3))
    data_missing = data_3d.copy()
    indices = [
        (i, j, k) for i in range(data_3d.shape[0]) for j in range(data_3d.shape[1]) for k in range(data_3d.shape[2])
    ]
    rng.shuffle(indices)
    for i, j, k in indices[:20]:
        data_missing[i, j, k] = np.nan

    data_original = data_missing.copy()
    FastKNNImputer(n_neighbors=5, temporal_mode="flatten").fit_transform(data_missing)
    _base_check_imputation(data_original, data_missing)
    assert data_missing.shape == (10, 5, 3)


def test_3d_per_variable_imputation(rng):
    """Tests if 3D imputation with per_variable mode successfully fills all NaN values"""
    data_3d = rng.uniform(0, 100, size=(10, 5, 3))
    data_missing = data_3d.copy()
    indices = [
        (i, j, k) for i in range(data_3d.shape[0]) for j in range(data_3d.shape[1]) for k in range(data_3d.shape[2])
    ]
    rng.shuffle(indices)
    for i, j, k in indices[:20]:
        data_missing[i, j, k] = np.nan

    data_original = data_missing.copy()
    FastKNNImputer(n_neighbors=5, temporal_mode="per_variable").fit_transform(data_missing)
    _base_check_imputation(data_original, data_missing)
    assert data_missing.shape == (10, 5, 3)


def test_3d_modes_produce_different_results(rng):
    """Tests if flatten and per_variable modes produce different results"""
    data_3d = rng.uniform(0, 100, size=(10, 5, 3))
    data_missing = data_3d.copy()
    indices = [
        (i, j, k) for i in range(data_3d.shape[0]) for j in range(data_3d.shape[1]) for k in range(data_3d.shape[2])
    ]
    rng.shuffle(indices)
    for i, j, k in indices[:20]:
        data_missing[i, j, k] = np.nan

    data_flatten = data_missing.copy()
    data_per_var = data_missing.copy()

    FastKNNImputer(n_neighbors=5, temporal_mode="flatten").fit_transform(data_flatten)
    FastKNNImputer(n_neighbors=5, temporal_mode="per_variable").fit_transform(data_per_var)

    assert not np.array_equal(data_flatten, data_per_var)


def test_invalid_temporal_mode():
    """Tests if imputer raises error for invalid temporal_mode"""
    with pytest.raises(ValueError):
        FastKNNImputer(temporal_mode="invalid")
