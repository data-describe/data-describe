import numpy as np
from mwdata.metrics.univariate import spikey, skewed


def test_spikey():
    assert spikey(np.random.standard_cauchy(100))


def test_skewed():
    assert skewed(np.random.lognormal(0, 5, 100))
