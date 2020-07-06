import matplotlib
import seaborn
import pytest

import data_describe as mw


matplotlib.use("Agg")


def test_scatter_plot_matrix(data):
    data = data.dropna(axis=1, how="all")
    fig = mw.scatter_plots(data, plot_mode="matrix")
    assert isinstance(fig, seaborn.axisgrid.PairGrid)


def test_scatter_plot_all(data):
    data = data.dropna(axis=1, how="all")
    fig = mw.scatter_plots(data, plot_mode="all")
    assert isinstance(fig, list)
    assert isinstance(fig[0], seaborn.axisgrid.JointGrid)


def test_scatter_plot(data):
    fig = mw.scatter_plots(data, plot_mode="diagnostic", threshold=0.15)
    assert isinstance(fig, list)
    assert isinstance(fig[0], seaborn.axisgrid.JointGrid)


def test_scatter_plot_dict(data):
    fig = mw.scatter_plots(
        data, plot_mode="diagnostic", threshold={"Outlier": 0.5}, dist_kws={"rug": True}
    )
    assert isinstance(fig, list)
    assert isinstance(fig[0], seaborn.axisgrid.JointGrid)


def test_scatter_plot_outside_threshold(data):
    with pytest.raises(ValueError):
        mw.scatter_plots(
            data,
            plot_mode="diagnostic",
            threshold={"Outlier": 0.999},
            dist_kws={"rug": True},
        )


def test_scatter_plot_wrong_data_type(data):
    with pytest.raises(NotImplementedError):
        mw.scatter_plots([1, 2, 3])
