import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api.graphics.tsa import plot_acf, plot_pacf


def viz_plot_time_series(df=None, cols=None, result=None, decompose=False):
    fig, ax = plt.subplots(figsize=(11, 9))

    if isinstance(cols, list):
        for i in cols:
            fig = sns.lineplot(x=df.index, y=df[i], legend="full", ax=ax)
    elif isinstance(cols, str):
        fig = sns.lineplot(x=df.index, y=df[cols], legend="full", ax=ax)
    elif decompose:
        fig = viz_decomposition(result)  # need to pass into figure
    return fig


def viz_decomposition(result):
    decompose_results = [result.observed, result.trend, result.seasonal, result.resid]
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(11, 9))
    for idx, ts in enumerate(decompose_results):
        sns.lineplot(y=ts, x=ts.index, ax=ax[idx])
    return fig


def viz_plot_autocorrelation(data, plot_type="acf"):
    fig, ax = plt.subplots(figsize=(11, 9))
    if plot_type == "acf":
        fig = plot_acf(data, ax=ax)
    elif plot_type == "pacf":
        fig = plot_pacf(data, ax=ax)
    else:
        raise ValueError("Unsupported input data type")
    return fig
