import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from data_describe.config._config import get_option


def viz_plot_time_series(df=None, cols=None, result=None, decompose=False, **kwargs):
    fig, ax = plt.subplots(
        figsize=(get_option("display.fig_height"), get_option("display.fig_width"))
    )
    if isinstance(cols, list):
        for i in cols:
            fig = sns.lineplot(x=df.index, y=df[i], legend="full", ax=ax, **kwargs)
    elif isinstance(cols, str):
        fig = sns.lineplot(x=df.index, y=df[cols], legend="full", ax=ax, **kwargs)
    elif decompose:
        fig = viz_decomposition(df, result)
    return fig


def viz_decomposition(df, result):
    fig, ax = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(get_option("display.fig_height"), get_option("display.fig_width")),
    )
    sns.lineplot(y=result.observed, x=df.index, ax=ax[0])
    sns.lineplot(y=result.trend, x=df.index, ax=ax[1])
    sns.lineplot(y=result.seasonal, x=df.index, ax=ax[2])
    sns.lineplot(y=result.resid, x=df.index, ax=ax[3])
    return fig


def viz_plot_autocorrelation(data, plot_type="acf", n_lags=40, **kwargs):
    fig, ax = plt.subplots(
        figsize=(get_option("display.fig_height"), get_option("display.fig_width"))
    )
    if plot_type == "acf":
        fig = sm.graphics.tsa.plot_acf(data, ax=ax, lags=n_lags, **kwargs)
    elif plot_type == "pacf":
        fig = sm.graphics.tsa.plot_pacf(data, ax=ax, lags=n_lags, **kwargs)
    else:
        raise ValueError("Unsupported input data type")
    return fig
