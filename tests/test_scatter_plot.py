import matplotlib
import seaborn
import pytest

import data_describe as dd
from data_describe.core.scatter import ScatterWidget


matplotlib.use("Agg")


@pytest.mark.base
def test_scatter_plot_matrix(data):
    data = data.dropna(axis=1, how="all")
    swidget = dd.scatter_plots(data, mode="matrix")

    assert isinstance(swidget, ScatterWidget)
    assert isinstance(swidget.show(), seaborn.axisgrid.PairGrid)


@pytest.mark.base
def test_scatter_plot_all(data):
    data = data.dropna(axis=1, how="all")
    fig = dd.scatter_plots(data, mode="all").show()

    assert isinstance(fig, list)
    assert isinstance(fig[0], seaborn.axisgrid.JointGrid)


def test_scatter_plot_diagnostic(_pyscagnostics, data):
    fig = dd.scatter_plots(data, mode="diagnostic", threshold=0.15).show()

    assert isinstance(fig, list)
    assert isinstance(fig[0], seaborn.axisgrid.JointGrid)


def test_scatter_plot_diagnostic_dict(_pyscagnostics, data):
    fig = dd.scatter_plots(
        data, mode="diagnostic", threshold={"Outlying": 0.1}, dist_kws={"rug": True}
    ).show()
    assert isinstance(fig, list)
    assert isinstance(fig[0], seaborn.axisgrid.JointGrid)


def test_scatter_plot_diagnostic_outside_threshold(_pyscagnostics, data):
    with pytest.warns(UserWarning, match="No plots identified by diagnostics"):
        dd.scatter_plots(
            data,
            mode="diagnostic",
            threshold={"Outlying": 0.999},
        ).show()


@pytest.mark.base
def test_scatter_plot_wrong_data_type(data):
    with pytest.raises(ValueError):
        dd.scatter_plots([1, 2, 3])
