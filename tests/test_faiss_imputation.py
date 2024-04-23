import numpy as np
import pandas as pd
import pytest

from fknni.faiss.faiss import FaissImputer


@pytest.fixture
def simple_test_df():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.integers(0, 100, size=(10, 5)), columns=list("ABCDE"))
    data_missing = data.copy()
    indices = [(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])]
    rng.shuffle(indices)
    for i, j in indices[:5]:  # Making 5 entries NaN
        data_missing.iat[i, j] = np.nan

    return data.to_numpy(), data_missing.to_numpy()


def test_median_imputation(simple_test_df):
    data, data_missing = simple_test_df
    imputer = FaissImputer(n_neighbors=5, strategy="median")
    imputer.fit(data_missing)

    df_imputed = imputer.transform(data_missing)

    assert not np.isnan(df_imputed).any(), "NaNs remain after median imputation"


def test_imputer_with_no_missing_values(simple_test_df):
    data, _ = simple_test_df
    imputer = FaissImputer(n_neighbors=5, strategy="median")
    imputer.fit(data)
    df_imputed = imputer.transform(data)

    np.testing.assert_array_equal(data, df_imputed, err_msg="Imputer altered data without missing values")
