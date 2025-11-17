import numpy as np


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

    return
