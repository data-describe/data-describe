import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def viz_plot_time_series(df=None, cols=None, result=None, decompose=False):
    fig, ax = plt.subplots(figsize=(11, 9))
    if isinstance(cols, list):
        for i in cols:
            fig = sns.lineplot(x=df.index, y=df[i], legend="full", ax=ax)
    elif isinstance(cols, str):
        fig = sns.lineplot(x=df.index, y=df[cols], legend="full", ax=ax)
    elif decompose:
        fig = viz_decomposition(df, result)
    return fig


def viz_decomposition(df, result):
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(11, 9))
    sns.lineplot(y=result.observed, x=df.index, ax=ax[0])
    sns.lineplot(y=result.trend, x=df.index, ax=ax[1])
    sns.lineplot(y=result.seasonal, x=df.index, ax=ax[2])
    sns.lineplot(y=result.resid, x=df.index, ax=ax[3])
    return fig


def viz_plot_autocorrelation(data, plot_type="acf", n_lags=40):
    fig, ax = plt.subplots(figsize=(11, 9))
    if plot_type == "acf":
        fig = sm.graphics.tsa.plot_acf(data, ax=ax, lags=n_lags)
    elif plot_type == "pacf":
        fig = sm.graphics.tsa.plot_pacf(data, ax=ax, lags=n_lags)
    else:
        raise ValueError("Unsupported input data type")
    return fig
