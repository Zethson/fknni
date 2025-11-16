import pytest


@pytest.mark.gpu
def gpu_test():
    assert 1 + 1 == 2
