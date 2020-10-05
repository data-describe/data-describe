import matplotlib
import pytest
import plotly.graph_objects as go

from data_describe.core.time_series import (
    _pandas_compute_stationarity_test,
    adf_test,
    kpss_test,
    _pandas_compute_decompose_timeseries,
    _pandas_compute_autocorrelation,
    plot_autocorrelation,
    stationarity_test,
)
import data_describe as dd
from data_describe.compat import _DATAFRAME_TYPE

matplotlib.use("Agg")


def test_plot_unsupported(compute_time_data):
    with pytest.raises(ValueError):
        dd.plot_time_series("this_is_a_string", col="var")

    with pytest.raises(ValueError):
        dd.plot_time_series(df=compute_time_data, col=1, decompose=True)


def test_stationarity_unsupported(compute_time_data):
    with pytest.raises(ValueError):
        _pandas_compute_stationarity_test(compute_time_data["var"], test="not a valid test")
    with pytest.raises(ValueError):
        stationarity_test(compute_time_data, col=["var"])
    with pytest.raises(ValueError):
        stationarity_test("Not a dataframe", col=["var"])


def test_pandas_compute_stationarity_test(compute_time_data):
    test_df = _pandas_compute_stationarity_test(compute_time_data["var"], test="dickey-fuller")
    assert isinstance(test_df, _DATAFRAME_TYPE)
    assert test_df.shape == (7, 1)
    test_df = _pandas_compute_stationarity_test(compute_time_data["var"], test="kpss")
    assert isinstance(test_df, _DATAFRAME_TYPE)
    assert test_df.shape == (7, 1)


def test_adf_test(compute_time_data):
    df = adf_test(compute_time_data["var"])
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
    df = kpss_test(compute_time_data["var"])
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


def test_decompose_timeseries(_statsmodels, compute_time_data):
    result = _pandas_compute_decompose_timeseries(
        compute_time_data, col="var", model="additive"
    )
    assert isinstance(result, _statsmodels.tsa.seasonal.DecomposeResult)
    assert len(result.trend) == 15
    assert len(result.observed) == 15
    assert len(result.seasonal) == 15
    assert len(result.resid) == 15


def test_pandas_compute_autocorrelation(compute_time_data):
    data, white_noise = _pandas_compute_autocorrelation(
        compute_time_data["var"], n_lags=1, plot_type="pacf"
    )
    assert len(data) == 2
    assert isinstance(white_noise, float)

    data, white_noise = _pandas_compute_autocorrelation(
        compute_time_data["var"], n_lags=1, plot_type="acf", fft=False
    )
    assert len(data) == 15
    assert isinstance(white_noise, float)


# NOTE: decomposition object in modin does not preserve index
def test_plotly(compute_time_data):
    fig = dd.plot_time_series(
        compute_time_data, col="var", viz_backend="plotly", model="additive"
    )
    assert isinstance(fig, go.Figure)
    fig = dd.plot_time_series(
        compute_time_data, col=["var"], viz_backend="plotly", model="additive"
    )
    assert isinstance(fig, go.Figure)
    fig = dd.plot_time_series(
        compute_time_data,
        col="var",
        decompose=True,
        model="additive",
        viz_backend="plotly",
    )
    assert isinstance(fig, go.Figure)

    fig = plot_autocorrelation(
        compute_time_data,
        col="var",
        n_lags=1,
        plot_type="acf",
        fft=False,
        viz_backend="plotly",
    )
    assert isinstance(fig, go.Figure)
    fig = plot_autocorrelation(
        compute_time_data, col="var", n_lags=1, plot_type="pacf", viz_backend="plotly"
    )
    assert isinstance(fig, go.Figure)


def test_seaborn(compute_time_data):
    fig = dd.plot_time_series(compute_time_data, col="var")
    assert isinstance(fig, matplotlib.artist.Artist)
    fig = dd.plot_time_series(
        compute_time_data, col="var", decompose=True, model="additive"
    )
    assert isinstance(fig, matplotlib.artist.Artist)
    fig = plot_autocorrelation(compute_time_data, col="var", n_lags=1, plot_type="pacf")
    assert isinstance(fig, matplotlib.figure.Figure)
