import matplotlib
import pytest

import data_describe as dd
from data_describe.core.distributions import DistributionWidget


def test_distribution(data):
    w = dd.distribution(data)
    assert isinstance(w, DistributionWidget), "Output was not a DistributionWidget"
    assert isinstance(w.plot_histogram("a"), matplotlib.axes.Axes), "Histogram was not a mpl plot"
    assert isinstance(w.plot_violin("a"), matplotlib.axes.Axes), "Violin was not a mpl plot"
    with pytest.raises(ValueError):
        assert isinstance(w.plot_histogram("d"), matplotlib.axes.Axes), "Histogram did not raise ValueError on categorical"
    with pytest.raises(ValueError):
        assert isinstance(w.plot_violin("d"), matplotlib.axes.Axes), "Violin did not raise ValueError on categorical"
    assert isinstance(w.plot_bar("d"), matplotlib.axes.Axes), "Bar was not a mpl plot"
    assert isinstance(w.plot_bar("a"), matplotlib.axes.Axes), "Bar was not a mpl plot"
