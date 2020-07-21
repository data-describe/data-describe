import matplotlib
import seaborn
import pytest

import data_describe as dd

matplotlib.use("Agg")


def test_scatter_plot_matrix(data):
    data = data.dropna(axis=1, how="all")
    fig = dd.scatter_plots(data, mode="matrix")
    assert isinstance(fig, seaborn.axisgrid.PairGrid)


def test_scatter_plot_all(data):
    data = data.dropna(axis=1, how="all")
    fig = dd.scatter_plots(data, mode="all")
    assert isinstance(fig, list)
    assert isinstance(fig[0], seaborn.axisgrid.JointGrid)


def test_scatter_plot(data):
    fig = dd.scatter_plots(data, mode="diagnostic", threshold=0.15)
    assert isinstance(fig, list)
    assert isinstance(fig[0], seaborn.axisgrid.JointGrid)


def test_scatter_plot_dict(data):
    fig = dd.scatter_plots(
        data, mode="diagnostic", threshold={"Outlying": 0.1}, dist_kws={"rug": True}
    )
    assert isinstance(fig, list)
    assert isinstance(fig[0], seaborn.axisgrid.JointGrid)


def test_scatter_plot_outside_threshold(data):
    with pytest.raises(UserWarning, match="No plots identified by diagnostics"):
        dd.scatter_plots(
            data,
            mode="diagnostic",
            threshold={"Outlying": 0.999},
            dist_kws={"rug": True},
        )


def test_scatter_plot_wrong_data_type(data):
    with pytest.raises(ValueError):
        dd.scatter_plots([1, 2, 3])
