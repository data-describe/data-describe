import warnings
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
from IPython import get_ipython

from data_describe.config._config import get_option
from data_describe.backends import _get_viz_backend, _get_compute_backend
from data_describe.compat import _DATAFRAME_TYPE


def plot_time_series(
    df,
    col,
    decompose=False,
    model="additive",
    compute_backend=None,
    viz_backend=None,
    **kwargs,
):
    """Plots time series given a dataframe with datetime index. Statistics are computed using the statsmodels API.

    Args:
        df: The dataframe with datetime index
        col (str or [str]): Column of interest. Column datatype must be numerical
        decompose: Set as True to decompose the timeseries with moving average. Defaults to False.
        model: Specify seasonal component when decompose is True. Defaults to "additive".
        compute_backend: Select computing backend. Defaults to None (pandas).
        viz_backend: Select visualization backend. Defaults to None (seaborn).
        **kwargs: Keyword arguments

    Raises:
        ValueError: Invalid input data type.
        ValueError: ```col``` not a list or string.

    Returns:
        The visualization
    """
    if not isinstance(df, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")
    if not isinstance(col, (list, str)):
        raise ValueError(f"{col} must be list type or string type")
    if decompose:
        result = _get_compute_backend(compute_backend, df).compute_decompose_timeseries(
            df, col=col, model=model, **kwargs
        )
        fig = _get_viz_backend(viz_backend).viz_plot_time_series(
            df, col=col, result=result, decompose=decompose, **kwargs
        )
    else:
        fig = _get_viz_backend(viz_backend).viz_plot_time_series(df, col, **kwargs)
    return fig


def stationarity_test(
    df, col, test="dickey-fuller", regression="c", compute_backend=None, **kwargs
):
    """Perform stationarity tests to see if mean and variance are changing over time.

    Backend uses statsmodel's statsmodels.tsa.stattools.adfuller or statsmodels.tsa.stattools.kpss

    Args:
        df: The dataframe. Must contain a datetime index
        col: The feature of interest
        test: Choice of stationarity test. "kpss" or "dickey-fuller". Defaults to "dickey-fuller".
        regression: Constant and trend order to include in regression. Choose between 'c','ct','ctt', and 'nc'. Defaults to 'c'
        compute_backend: Select computing backend. Defaults to None (pandas).
        **kwargs: Keyword arguments

    Raises:
        ValueError: Invalid input data type.
        ValueError: `col` not found in dataframe.

    Returns:
        Pandas dataframe containing the statistics
    """
    if not isinstance(df, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")
    if not isinstance(col, str):
        raise ValueError(f"{col} not found in dataframe")

    data = _get_compute_backend(compute_backend, df).compute_stationarity_test(
        df[col], test, regression, **kwargs
    )
    return data


def plot_autocorrelation(
    df,
    col,
    plot_type="acf",
    n_lags=40,
    fft=False,
    compute_backend=None,
    viz_backend=None,
    **kwargs,
):
    """Correlation estimate using partial autocorrelation or autocorrelation.

    Statistics are computed using the statsmodels API.

    Args:
        df: The dataframe with datetime index
        col: The feature of interest
        plot_type: Choose between 'acf' or 'pacf. Defaults to "pacf".
        n_lags: Number of lags to return autocorrelation for. Defaults to 40.
        fft: If True, computes ACF via fourier fast transform (FFT). Defaults to False.
        compute_backend: Select computing backend. Defaults to None (pandas).
        viz_backend: Select visualization backend. Defaults to None (seaborn).
        **kwargs: Keyword arguments

    Raises:
        ValueError: Invalid input data type.
        ValueError: `col` not found in dataframe.

    Returns:
        The visualization
    """
    if not isinstance(df, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")
    if isinstance(col, str):
        if col not in df.columns:
            raise ValueError(f"{col} not found in dataframe")
    if viz_backend == "plotly":
        data, white_noise = _get_compute_backend(
            compute_backend, df
        ).compute_autocorrelation(
            df[col], plot_type=plot_type, n_lags=n_lags, fft=fft, **kwargs
        )
        fig = _get_viz_backend(viz_backend).viz_plot_autocorrelation(
            data, plot_type=plot_type, white_noise=white_noise, n_lags=n_lags, **kwargs
        )
    else:
        fig = _get_viz_backend(viz_backend).viz_plot_autocorrelation(
            df[col], plot_type=plot_type, n_lags=n_lags, fft=fft, **kwargs
        )
    return fig


def _pandas_compute_stationarity_test(
    timeseries, test: str = "dickey-fuller", regression: str = "c", **kwargs
):
    """Perform stationarity tests to see if mean and variance are changing over time.

    Backend uses statsmodel's  statsmodels.tsa.stattools.adfuller or statsmodels.tsa.stattools.kpss

    Args:
        timeseries: Series containing a datetime index
        test: Choice of stationarity test. "kpss" or "dickey-fuller". Defaults to "dickey-fuller".
        regression: Constant and trend order to include in regression. Choose between 'c','ct','ctt', and 'nc'
        **kwargs: Keyword arguments for adf and kpss

    Raises:
        ValueError: Invalid `test` type.

    Returns:
        Pandas dataframe containing the statistics
    """
    if test.lower() == "dickey-fuller":
        st = adf_test(timeseries, regression=regression, **kwargs)
    elif test.lower() == "kpss":
        st = kpss_test(timeseries, regression=regression, **kwargs)
    else:
        raise ValueError(f"{test} not implemented")
    return st


def adf_test(timeseries, autolag: str = "AIC", regression: str = "c", **kwargs):
    """Compute the Augmented Dickey-Fuller (ADF) test for stationarity.

    Backend uses statsmodels.tsa.stattools.adfuller

    Args:
        timeseries: The timeseries
        autolag: Method to use when determining the number of lags. Defaults to 'AIC'. Choose between 'AIC', 'BIC', 't-stat', and None
        regression: Constant and trend order to include in regression. Choose between 'c','ct','ctt', and 'nc'
        **kwargs: Keyword arguments for adfuller

    Returns:
        Pandas dataframe containing the statistics
    """
    test = adfuller(timeseries, autolag=autolag, regression=regression, **kwargs)
    adf_output = pd.Series(
        test[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in test[4].items():
        adf_output["Critical Value (%s)" % key] = value
    return pd.DataFrame(adf_output, columns=["stats"])


def kpss_test(timeseries, regression: str = "c", nlags: Optional[int] = None, **kwargs):
    """Compute the Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test for stationarity.

    Backend uses statsmodels.tsa.stattools.kpss

    Args:
        timeseries: The timeseries
        regression: The null hypothesis for the KPSS test.
            'c' : The data is stationary around a constant (default).
            'ct' : The data is stationary around a trend.
        nlags:  Indicates the number of lags to be used. Defaults to None.
        **kwargs: Keyword arguments for kpss

    Returns:
        Pandas dataframe containing the statistics
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="The behavior of using lags=None will change in the next release.",
        )
        test = kpss(timeseries, regression="c", **kwargs)
    kpss_output = pd.Series(test[0:3], index=["Test Statistic", "p-value", "Lags Used"])
    for key, value in test[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    return pd.DataFrame(kpss_output, columns=["stats"])


def _pandas_compute_decompose_timeseries(df, col, model: str = "additive", **kwargs):
    """Seasonal decomposition using moving averages.

    Note:
        The decomposition object in Modin does not preserve datetime index.

    Args:
        df: The dataframe
        col: The col of interest. Must be numeric datatype
        model: Type of seasonal component. Defaults to "additive".
        **kwargs: Keyword arguments

    Returns:
        statsmodels.tsa.seasonal.DecomposeResult object
    """
    return seasonal_decompose(df[col], model=model, **kwargs)


def _pandas_compute_autocorrelation(
    timeseries,
    n_lags: Optional[int] = 40,
    plot_type: str = "acf",
    fft: bool = False,
    **kwargs,
):
    """Correlation estimate using partial autocorrelation or autocorrelation.

    Args:
        timeseries: Series object containing datetime index
        n_lags: Number of lags to return autocorrelation for. Defaults to 40.
        plot_type: Choose between 'acf' or 'pacf. Defaults to "acf".
        fft: If True, computes ACF via fourier fast transform (FFT). Defaults to False.
        **kwargs: Keyword arguments

    Raises:
        ValueError: Invalid `plot_type`.

    Returns:
        numpy.ndarray containing the correlations
    """
    if plot_type == "pacf":
        data = pacf(timeseries, n_lags, **kwargs)
    elif plot_type == "acf":
        data = acf(timeseries, n_lags, fft=fft, **kwargs)
    else:
        raise ValueError("Unsupported input data type")
    white_noise = 1.96 / np.sqrt(len(data))
    return data, white_noise


def _plotly_viz_plot_time_series(
    df, col, result=None, decompose=False, title="Time Series"
):
    """Create timeseries visualization.

    Args:
        df: The dataframe
        col (str or [str]): Column of interest. Column datatype must be numerical.
        result: The statsmodels.tsa.seasonal.DecomposeResult object. Defaults to None.
        decompose: Set as True to decompose the timeseries with moving average. result must not be None. Defaults to False.
        title: Title of the plot. Defaults to "Time Series".

    Returns:
        The visualization
    """
    if isinstance(col, list):
        data = [go.Scatter(x=df.index, y=df[c], name=c) for c in col]
        ylabel = "Variable" if len(col) > 1 else col[0]
        fig = go.Figure(data=data, layout=figure_layout(title=title, ylabel=ylabel))
    elif isinstance(col, str) and not decompose:
        fig = go.Figure(
            data=go.Scatter(x=df.index, y=df[col], name=col),
            layout=figure_layout(title=title, ylabel=col),
        )
    elif decompose:
        fig = _plotly_viz_decomposition(result, dates=df.index)
    if get_ipython() is not None:
        init_notebook_mode(connected=True)
        return iplot(fig, config={"displayModeBar": False})
    else:
        return fig


def _plotly_viz_decomposition(result, dates, title="Time Series Decomposition"):
    """Create timeseries decomposition visualization.

    Args:
        result: The statsmodels.tsa.seasonal.DecomposeResult object. Defaults to None.
        dates: The datetime index
        title: Title of the plot. Defaults to "Time Series Decomposition".

    Returns:
        The visualization
    """
    fig = make_subplots(rows=4, cols=1, x_title="Time", shared_xaxes=True)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=result.observed,
            name="observed",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=dates, y=result.trend, name="trend"), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=dates, y=result.seasonal, name="seasonal"),
        row=3,
        col=1,
    )
    fig.add_trace(go.Scatter(x=dates, y=result.resid, name="residual"), row=4, col=1)
    fig.update_layout(
        height=get_option("display.plotly.fig_height"),
        width=get_option("display.plotly.fig_width"),
        title_text=title,
        legend_title_text="Decomposition",
    )
    return fig


def _plotly_viz_plot_autocorrelation(
    data, white_noise, n_lags, plot_type="acf", title="Autocorrelation Plot"
):
    """Create timeseries autocorrelation visualization.

    Args:
        data: numpy.ndarray containing the correlations
        white_noise: Significance threshold
        n_lags: The number of lags to plot.
        plot_type: Choose between 'acf' or 'pacf. Defaults to "pacf".
        title: Title of the plot. Defaults to "Autocorrelation Plot".

    Raises:
        ValueError: Invalid `plot_type`.

    Returns:
        The visualization
    """
    if plot_type == "acf":
        data = [go.Bar(y=data, showlegend=False, name=plot_type)]
    elif plot_type == "pacf":
        data = [go.Bar(y=data, showlegend=False, name=plot_type)]
    else:
        raise ValueError("Unsupported input data type")

    fig = go.Figure(data=data, layout=figure_layout(title, "Lags", plot_type))
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(n_lags + 1)],
            y=[white_noise for i in range(n_lags + 1)],
            mode="lines",
            name="95% Confidence",
            line={"dash": "dash"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[i for i in range(n_lags + 1)],
            y=[-white_noise for i in range(n_lags + 1)],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            showlegend=False,
        )
    )
    return fig


def figure_layout(title="Time Series", xlabel="Date", ylabel="Variable"):
    """Generates the figure layout.

    Args:
        title: Title of the plot. Defaults to "Time Series".
        xlabel: x-axis label. Defaults to "Date".
        ylabel: y-axis label. Defaults to "Variable".

    Returns:
        The plotly layout
    """
    layout = go.Layout(
        title={
            "text": title,
            "font": {"size": get_option("display.plotly.title_size")},
        },
        width=get_option("display.plotly.fig_width"),
        height=get_option("display.plotly.fig_height"),
        xaxis=go.layout.XAxis(ticks="", title=xlabel, showgrid=True),
        yaxis=go.layout.YAxis(ticks="", title=ylabel, automargin=True, showgrid=True),
    )
    return layout


def _seaborn_viz_plot_time_series(df, col, result=None, decompose=False):
    """Create timeseries visualization.

    Args:
        df: The dataframe
        col (str or [str]): Column of interest. Column datatype must be numerical.
        result: The statsmodels.tsa.seasonal.DecomposeResult object. Defaults to None.
        decompose: Set as True to decompose the timeseries with moving average. result must not be None. Defaults to False.

    Returns:
        The visualization
    """
    fig, ax = plt.subplots(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )

    if isinstance(col, list):
        for i in col:
            fig = sns.lineplot(x=df.index, y=df[i], legend="full", ax=ax)
        ax.legend(labels=col)
    elif isinstance(col, str) and not decompose:
        fig = sns.lineplot(x=df.index, y=df[col], legend="full", ax=ax)
    elif decompose:
        fig = _seaborn_viz_decomposition(df, result)
        plt.close()
    return fig


def _seaborn_viz_decomposition(df, result):
    """Create timeseries decomposition visualization.

    Args:
        df: The dataframe
        result: The statsmodels.tsa.seasonal.DecomposeResult object.

    Returns:
        The visualization
    """
    fig, ax = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        ),
    )
    sns.lineplot(y=result.observed, x=df.index, ax=ax[0])
    sns.lineplot(y=result.trend, x=df.index, ax=ax[1])
    sns.lineplot(y=result.seasonal, x=df.index, ax=ax[2])
    sns.lineplot(y=result.resid, x=df.index, ax=ax[3])
    fig.suptitle("Time Series Decomposition", fontsize=18)

    plt.close()
    return fig


def _seaborn_viz_plot_autocorrelation(
    timeseries, plot_type="acf", n_lags=40, fft=False, **kwargs
):
    """Create timeseries autocorrelation visualization.

    Args:
        timeseries: Series object containing datetime index
        plot_type: Choose between 'acf' or 'pacf. Defaults to "pacf".
        n_lags: Number of lags to return autocorrelation for. Defaults to 40.
        fft (bool): If True, computes ACF via FFT.
        **kwargs: Keyword arguments for plot_acf or plot_pacf.

    Raises:
        ValueError: Invalid `plot_type`.

    Returns:
        The visualization
    """
    fig, ax = plt.subplots(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )
    if plot_type == "acf":
        fig = sm.graphics.tsa.plot_acf(
            timeseries, ax=ax, lags=n_lags, fft=fft, **kwargs
        )
    elif plot_type == "pacf":
        fig = sm.graphics.tsa.plot_pacf(timeseries, ax=ax, lags=n_lags, **kwargs)
    else:
        raise ValueError("Unsupported input data type")
    plt.xlabel("Lags")
    plt.ylabel(plot_type)
    plt.close()
    return fig
