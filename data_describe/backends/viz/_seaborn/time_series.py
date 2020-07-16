import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels

from data_describe.config._config import get_option


def viz_plot_time_series(df, col, result=None, decompose=False, **kwargs):
    """Create timeseries visualization

    Args:
        df: The dataframe
        col (str or [str]): Column of interest. Column datatype must be numerical.
        result: The statsmodels.tsa.seasonal.DecomposeResult object. Defaults to None.
        decompose: Set as True to decompose the timeseries with moving average. result must not be None. Defaults to False.

    Returns:
        fig: The visualization
    """
    fig, ax = plt.subplots(
        figsize=(get_option("display.fig_height"), get_option("display.fig_width"))
    )
    if isinstance(col, list):
        for i in col:
            fig = sns.lineplot(x=df.index, y=df[i], legend="full", ax=ax, **kwargs)
        ax.legend(labels=col)
    elif isinstance(col, str):
        fig = sns.lineplot(x=df.index, y=df[col], legend="full", ax=ax, **kwargs)
    elif decompose and isinstance(result, statsmodels.tsa.seasonal.DecomposeResult):
        fig = viz_decomposition(df, result)
        plt.close()
    return fig


def viz_decomposition(df, result):
    """Create timeseries decomposition visualization

    Args:
        df: The dataframe
        result: The statsmodels.tsa.seasonal.DecomposeResult object.

    Returns:
        fig: The visualization
    """
    fig, ax = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        figsize=(get_option("display.fig_height"), get_option("display.fig_width")),
    )
    sns.lineplot(y=result.observed, x=df.index, ax=ax[0])
    sns.lineplot(y=result.trend, x=df.index, ax=ax[1])
    sns.lineplot(y=result.seasonal, x=df.index, ax=ax[2])
    sns.lineplot(y=result.resid, x=df.index, ax=ax[3])
    plt.close()
    return fig


def viz_plot_autocorrelation(timeseries, plot_type="acf", n_lags=40, **kwargs):
    """Create timeseries autocorrelation visualization

    Args:
        timeseries: Series object containing datetime index
        plot_type: Choose between 'acf' or 'pacf. Defaults to "pacf".
        n_lags: Number of lags to return autocorrelation for. Defaults to 40.

    Returns:
        [type]: [description]
    """
    fig, ax = plt.subplots(
        figsize=(get_option("display.fig_height"), get_option("display.fig_width"))
    )
    if plot_type == "acf":
        fig = sm.graphics.tsa.plot_acf(timeseries, ax=ax, lags=n_lags, **kwargs)
    elif plot_type == "pacf":
        fig = sm.graphics.tsa.plot_pacf(timeseries, ax=ax, lags=n_lags, **kwargs)
    else:
        raise ValueError("Unsupported input data type")
    return fig
