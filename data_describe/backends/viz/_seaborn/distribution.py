import matplotlib.pyplot as plt
import seaborn as sns

from data_describe.config._config import get_option


def viz_distribution(data, **kwargs):
    pass


def viz_single_histogram(data, x, category=None, **kwargs):
    default_hist_kwargs = {"kde": False, "rug": False}
    hist_kwargs = {**default_hist_kwargs, **(kwargs or {})}
    plt.figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    if category:
        g = sns.FacetGrid(data, hue=category)
        ax = g.map(sns.distplot, x, **hist_kwargs)
    else:
        ax = sns.distplot(data[x], **hist_kwargs)
    ax.set_title("Histogram of " + x.name)
    plt.ylabel("Count")
    plt.xlabel(x.name)
    return ax


def viz_multiple_histogram(data, columns=None, **kwargs):
    if columns is None:
        columns = data.columns

    return


def viz_single_violin(data, **kwargs):
    pass


def viz_multiple_violin(data, columns=None, **kwargs):
    if columns is None:
        columns = data.columns

    return


def viz_single_bar(data, **kwargs):
    pass


def viz_multiple_bar(data, columns=None, **kwargs):
    if columns is None:
        columns = data.columns

    return
