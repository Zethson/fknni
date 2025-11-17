import pytest
from tests.compare_predictions import _base_check_imputation

from fknni.faiss.faiss import FaissImputer


@pytest.mark.gpu
def test_median_imputation(simple_test_df):
    """Tests if median imputation successfully fills all NaN values"""
    data, data_missing = simple_test_df
    data_original = data_missing.copy()
    FaissImputer(n_neighbors=5, strategy="median", use_gpu=True).fit_transform(data_missing)
    _base_check_imputation(data_original, data_missing)
