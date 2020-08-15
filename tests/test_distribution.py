import matplotlib

import data_describe as dd
from data_describe.core.distributions import DistributionWidget


def test_distribution(data):
    w = dd.distribution(data)
    assert isinstance(w, DistributionWidget), "Output was not a DistributionWidget"
    assert isinstance(
        w.plot_distribution("a"), matplotlib.figure.Figure
    ), "plot_distribution[numeric] was not a mpl figure"
    assert isinstance(
        w.plot_distribution("a", contrast="e"), matplotlib.figure.Figure
    ), "plot_distribution[numeric] with contrast was not a mpl figure"
    assert isinstance(
        w.plot_distribution("d"), matplotlib.figure.Figure
    ), "plot_distribution[categorical] was not a mpl figure"
    assert isinstance(
        w.plot_distribution("d", contrast="e"), matplotlib.figure.Figure
    ), "plot_distribution[categorical] with contrast was not a mpl figure"
