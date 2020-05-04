import pandas as pd
import mwdata as mw
import matplotlib

matplotlib.use("Agg")
import seaborn
import pytest


@pytest.fixture
def data():
    file_path = "data/er_data.csv"
    df = pd.read_csv(file_path)
    num_df = df.select_dtypes(["number"])
    num_df = num_df.sample(500, random_state=1)
    num_df = num_df.iloc[:, 0:3]
    return num_df


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
    fig = mw.scatter_plots(data, plot_mode="diagnostic", threshold=0.85)
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
