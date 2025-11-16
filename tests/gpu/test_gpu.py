import pytest


@pytest.mark.gpu
def test_gpu():
    assert 1 + 1 == 2
