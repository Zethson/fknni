import pytest
from tests.compare_predictions import _base_check_imputation

from fknni.knn.knn import FastKNNImputer

cupy = pytest.importorskip("cupy")


@pytest.mark.gpu
def test_median_imputation_faiss(simple_test_df):
    """Tests if median imputation successfully fills all NaN values"""
    _, data_missing = simple_test_df
    data_original = data_missing.copy()
    FastKNNImputer(n_neighbors=5, strategy="median", use_gpu=True).fit_transform(data_missing)
    _base_check_imputation(data_original, data_missing)


@pytest.mark.gpu
def test_median_imputation_cupy(simple_test_df):
    """Tests if median imputation with cupy arrays fills all NaN values and stays on GPU."""
    _, data_missing = simple_test_df
    data_missing_cp = cupy.asarray(data_missing)
    data_original_cp = data_missing_cp.copy()

    result = FastKNNImputer(n_neighbors=5, strategy="median").fit_transform(data_missing_cp)

    assert isinstance(result, cupy.ndarray), f"Expected cupy array, got {type(result)}"
    assert not cupy.isnan(result).any(), "NaNs remain after imputation"
    _base_check_imputation(data_original_cp.get(), result.get())


@pytest.mark.gpu
@pytest.mark.parametrize("strategy", ["mean", "median", "weighted"])
def test_cupy_strategies(simple_test_df, strategy):
    """Tests all imputation strategies with cupy arrays."""
    _, data_missing = simple_test_df
    data_missing_cp = cupy.asarray(data_missing)

    result = FastKNNImputer(n_neighbors=5, strategy=strategy).fit_transform(data_missing_cp)

    assert isinstance(result, cupy.ndarray)
    assert not cupy.isnan(result).any()


@pytest.mark.gpu
def test_cupy_numpy_produce_same_results(simple_test_df):
    """Tests that cupy and numpy paths produce equivalent results."""
    _, data_missing = simple_test_df
    data_missing_np = data_missing.copy()
    data_missing_cp = cupy.asarray(data_missing.copy())

    result_np = FastKNNImputer(n_neighbors=5, strategy="mean").fit_transform(data_missing_np)
    result_cp = FastKNNImputer(n_neighbors=5, strategy="mean").fit_transform(data_missing_cp)

    cupy.testing.assert_allclose(result_cp, cupy.asarray(result_np), rtol=1e-5)
