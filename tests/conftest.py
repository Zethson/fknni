import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def simple_test_df(rng):
    data = pd.DataFrame(rng.integers(0, 100, size=(10, 5)), columns=list("ABCDE"))
    data_missing = data.copy()
    indices = [(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])]
    rng.shuffle(indices)
    for i, j in indices[:5]:
        data_missing.iat[i, j] = np.nan
    return data.to_numpy(), data_missing.to_numpy()
