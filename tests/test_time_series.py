import matplotlib
import pytest
import plotly
import statsmodels

from data_describe.backends.compute._pandas.time_series import (
    compute_stationarity_test,
    adf_test,
    kpss_test,
    compute_decompose_timeseries,
)
from data_describe.backends.viz._plotly import (
    viz_decomposition as plotly_decomp,
    viz_plot_time_series as plotly_viz,
)
from data_describe.backends.viz._seaborn import (
    viz_decomposition as sns_decomp,
    viz_plot_time_series as sns_viz,
)
import data_describe as mw
from data_describe.compat import _DATAFRAME_TYPE


matplotlib.use("Agg")


def test_not_df():
    with pytest.raises(NotImplementedError):
        mw.time_series("this_is_a_string")


def test_compute_stationarity_test():
    df = "test"  # need to replace
    test_df = compute_stationarity_test(df, test="dickey-fuller")
    assert isinstance(test_df, _DATAFRAME_TYPE)
    assert test_df.shape == (1, 7)
    test_df = compute_stationarity_test(df, test="kpss")
    assert isinstance(df, _DATAFRAME_TYPE)
    assert test_df.shape == (1, 7)


def test_adf_test():
    df = "test"  # need to replace
    df = adf_test(df)
    adf_idx = [
        "Test Statistic",
        "p-value",
        "#Lags Used",
        "Number of Observations Used",
        "Critical Value (1%)",
        "Critical Value (5%)",
        "Critical Value (10%)",
    ]

    assert df.shape == (1, 7)
    df.index.tolist = adf_idx
    # assert values


def test_kpss_test():
    df = "test"  # need to replace
    df = kpss_test(df)
    kpss_idx = [
        "Test Statistic",
        "p-value",
        "Lags Used",
        "Critical Value (10%)",
        "Critical Value (5%)",
        "Critical Value (2.5%)",
        "Critical Value (1%)",
    ]

    assert df.shape == (1, 7)
    df.index.tolist = kpss_idx
    # assert values


def test_decompose_timeseries():
    df = "test"  # need to replace
    result = compute_decompose_timeseries(df)
    assert isinstance(result, statsmodels.tsa.seasonal.DecomposeResult)


def test_plotly_viz_decomposition():
    result = "test"  # need to remove
    fig = plotly_decomp(result)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_plotly_plot_time_series():
    result = "test"  # need to remove
    fig = plotly_viz(result)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_sns_viz_decomposition():
    result = "test"  # need to remove
    fig = sns_decomp(result)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_sns_plot_time_series():
    result = "test"  # need to remove
    fig = sns_viz(result)
    assert isinstance(fig, matplotlib.figure.Figure)
