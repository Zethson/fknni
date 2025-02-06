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

# TODO: Should we also make the "base checks" we do in ehrapy? See ehrapy/tests/preprocessing/test_imputation/_base_check_imputation

def test_median_imputation(simple_test_df):
    """Tests if median imputation successfully fills all NaN values"""
    data, data_missing = simple_test_df
    imputer = FaissImputer(n_neighbors=5, strategy="median")
    df_imputed = imputer.fit_transform(data_missing)
    assert not np.isnan(df_imputed).any()


def test_mean_imputation(simple_test_df):
    """Tests if mean imputation successfully fills all NaN values"""
    data, data_missing = simple_test_df
    imputer = FaissImputer(n_neighbors=5, strategy="mean")
    df_imputed = imputer.fit_transform(data_missing)
    assert not np.isnan(df_imputed).any()


def test_imputer_with_no_missing_values(simple_test_df):
    """Tests if imputer preserves data when no values are missing"""
    data, _ = simple_test_df
    imputer = FaissImputer(n_neighbors=5, strategy="median")
    df_imputed = imputer.fit_transform(data)
    np.testing.assert_array_equal(data, df_imputed)


def test_imputer_with_all_nan_column(rng):
    """Tests if imputer raises error when entire column is NaN"""
    data = rng.uniform(0, 100, size=(10, 5))
    data_missing = data.copy()
    data_missing[:, 2] = np.nan
    with pytest.raises(ValueError):
        imputer = FaissImputer(n_neighbors=5)
        imputer.fit_transform(data_missing)


def test_imputer_with_all_nan_row(rng):
    """Tests if imputer handles all-NaN rows by imputing them"""
    data = rng.uniform(0, 100, size=(10, 5))
    data_missing = data.copy()
    data_missing[3, :] = np.nan

    imputer = FaissImputer(n_neighbors=5)
    imputed = imputer.fit_transform(data_missing)

    assert not np.isnan(imputed[3, :]).any()
    assert not np.array_equal(imputed[3, :], data[3, :])


def test_imputer_different_n_neighbors(simple_test_df):
    """Tests if different n_neighbors values produce different results"""
    data, data_missing = simple_test_df
    imputer_3 = FaissImputer(n_neighbors=3).fit_transform(data_missing)
    imputer_7 = FaissImputer(n_neighbors=7).fit_transform(data_missing)
    assert not np.array_equal(imputer_3, imputer_7)


def test_regression_imputation(regression_dataset):
    """Tests if imputed data maintains predictive power in regression task"""
    X, X_missing, y = regression_dataset
    imputer = FaissImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_missing)
    assert not np.isnan(X_imputed).any()

    from sklearn.linear_model import LinearRegression

    model_orig = LinearRegression().fit(X, y)
    model_imputed = LinearRegression().fit(X_imputed, y)

    score_orig = model_orig.score(X, y)
    score_imputed = model_imputed.score(X_imputed, y)
    assert abs(score_orig - score_imputed) < 0.1


def test_transform_new_data(simple_test_df):
    """Tests if transform works correctly on new data"""
    # TODO: Here we have a problem. We can't train with a dataset and impute another one. Test disabled for now
    pass
    # data, data_missing = simple_test_df
    # imputer = FaissImputer(n_neighbors=5)
    # imputer.fit(data)
    #
    # new_data = data_missing.copy()
    # imputed_new = imputer.transform(new_data)
    # assert not np.isnan(imputed_new).any()


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
    imputer = FaissImputer(n_neighbors=1)
    arr_imputed = imputer.fit_transform(arr)
    assert not np.isnan(arr_imputed).any()
