import pytest
import numpy as np

from data_describe.metrics.univariate import spikey, skewed


@pytest.mark.base
def test_spikey():
    np.random.seed(1)
    assert spikey(np.random.standard_cauchy(100))


@pytest.mark.base
def test_skewed():
    np.random.seed(1)
    assert skewed(np.random.lognormal(0, 5, 100))
