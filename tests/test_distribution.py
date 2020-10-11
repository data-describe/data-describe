import matplotlib

import data_describe as dd
from data_describe.compat import _is_series
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
    assert w.spike_factor == 10, "Wrong default spike factor"
    assert w.skew_factor == 3, "Wrong default skew factor"
    assert _is_series(w.spike_value), "Spike values not a Pandas series"
    assert _is_series(w.skew_value), "Skew values not a Pandas series"
