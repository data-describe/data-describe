import matplotlib
import pytest
from plotly.graph_objects import Figure
import statsmodels


from data_describe.backends.compute._pandas.time_series import (
    compute_stationarity_test,
    adf_test,
    kpss_test,
    compute_decompose_timeseries,
    compute_autocorrelation,
)
from data_describe.core.time_series import plot_autocorrelation
import data_describe as dd
from data_describe.compat import _DATAFRAME_TYPE

matplotlib.use("Agg")


def test_unsupported(compute_time_data):
    with pytest.raises(ValueError):
        dd.plot_time_series("this_is_a_string")
    with pytest.raises(ValueError):
        compute_stationarity_test(compute_time_data, test="not a valid test")
    with pytest.raises(ValueError):
        compute_decompose_timeseries(df="not a valid type", col="Not a valid")


def test_compute_stationarity_test(compute_time_data):
    test_df = compute_stationarity_test(compute_time_data, test="dickey-fuller")
    assert isinstance(test_df, _DATAFRAME_TYPE)
    assert test_df.shape == (7, 1)
    test_df = compute_stationarity_test(compute_time_data, test="kpss")
    assert isinstance(test_df, _DATAFRAME_TYPE)
    assert test_df.shape == (7, 1)


def test_adf_test(compute_time_data):
    df = adf_test(compute_time_data)
    adf_idx = [
        "Test Statistic",
        "p-value",
        "Lags Used",
        "Number of Observations Used",
        "Critical Value (1%)",
        "Critical Value (5%)",
        "Critical Value (10%)",
    ]

    assert df.shape == (7, 1)
    assert df.index.tolist() == adf_idx
    assert df.columns[0] == "stats"


def test_kpss_test(compute_time_data):
    df = kpss_test(compute_time_data)
    kpss_idx = [
        "Test Statistic",
        "p-value",
        "Lags Used",
        "Critical Value (10%)",
        "Critical Value (5%)",
        "Critical Value (2.5%)",
        "Critical Value (1%)",
    ]
    assert df.shape == (7, 1)
    assert df.index.tolist() == kpss_idx
    assert df.columns[0] == "stats"


def test_decompose_timeseries(compute_time_data):
    result = compute_decompose_timeseries(
        compute_time_data, col="var", model="additive"
    )
    assert isinstance(result, statsmodels.tsa.seasonal.DecomposeResult)
    assert len(result.trend) == 15
    assert len(result.observed) == 15
    assert len(result.seasonal) == 15
    assert len(result.resid) == 15


# check kwargs are passed
def test_compute_autocorrelation(compute_time_data):
    data = compute_autocorrelation(compute_time_data["var"], n_lags=1, plot_type="pacf")
    assert len(data) == 2

    data = compute_autocorrelation(
        compute_time_data["var"], n_lags=1, plot_type="acf", fft=False
    )
    assert len(data) == 15


# NOTE: decomposition object in modin does not preserve index
def test_plotly(compute_time_data):
    fig = dd.plot_time_series(
        compute_time_data, col="var", viz_backend="plotly", model="additive"
    )
    assert isinstance(fig, Figure)
    fig = dd.plot_time_series(
        compute_time_data, col=["var"], viz_backend="plotly", model="additive"
    )
    assert isinstance(fig, Figure)
    fig = dd.plot_time_series(
        compute_time_data,
        col="var",
        decompose=True,
        model="additive",
        viz_backend="plotly",
    )
    assert isinstance(fig, Figure)


def test_seaborn(compute_time_data):
    fig = dd.plot_time_series(compute_time_data, col="var")
    assert isinstance(fig, matplotlib.artist.Artist)
    fig = dd.plot_time_series(compute_time_data, col=["var"])
    assert isinstance(fig, matplotlib.artist.Artist)
    fig = dd.plot_time_series(
        compute_time_data, col="var", decompose=True, model="additive"
    )
    assert isinstance(fig, matplotlib.figure.Figure)


# can not find plot_autocorrelations in dd
def test_auto(compute_time_data):
    fig = plot_autocorrelation(
        compute_time_data, col="var", n_lags=1, plot_type="acf", fft=False
    )
    assert isinstance(fig, matplotlib.figure.Figure)
