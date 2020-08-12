import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import seaborn as sns

from data_describe.config._config import get_option


def viz_distribution_diagnostic(data, is_skewed, is_spikey, **kwargs):
    skew_plots = [
        viz_distribution(data, x, **kwargs)
        for x in is_skewed.where(lambda x: x).dropna().index
    ]
    spikey_plots = [
        viz_distribution(data, x, **kwargs)
        for x in is_skewed.where(lambda x: x).dropna().index
    ]
    return skew_plots + spikey_plots


def viz_distribution(data, x, contrast=None, **kwargs):
    """Plots the numeric or categorical distribution."""
    if x in data.select_dtypes("number").columns:
        return viz_hist_violin(data, x, contrast, **kwargs)
    else:
        return viz_bar(data, x, contrast, **kwargs)


def viz_hist_violin(
    data, x, contrast=None, hist_kwargs=None, violin_kwargs=None, **kwargs
):
    """Plots a histogram/violin combo plot.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot
        contrast (str, optional): The name of the categrical column to use for multiple contrasts.
        hist_kwargs (dict, optional): Keyword args for seaborn.distplot.
        violin_kwargs (dict, optional): Keyword args for seaborn.violinplot.
    """
    fig = Figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    gs = GridSpec(nrows=5, ncols=1)
    ax1 = fig.add_subplot(gs.new_subplotspec((0, 0), 1, 1))
    ax2 = fig.add_subplot(gs.new_subplotspec((1, 0), 4, 1))

    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    _viz_histogram(data, x, contrast=None, ax=ax1)
    _viz_violin(data, x, contrast, ax=ax2)
    ax1.set_title(x)
    return fig


def _viz_histogram(data, x, contrast=None, **kwargs):
    """Plot a single histogram.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the categorical column to use for multiple contrasts.

    Returns:
        Seaborn Axis Object
    """
    if x not in data.select_dtypes("number").columns:
        raise ValueError("x must be numeric column")

    default_hist_kwargs = {"kde": False, "rug": True}
    hist_kwargs = {**default_hist_kwargs, **(kwargs or {})}
    if contrast:
        # TODO (haishiro): Use histplot from seaborn PR #2125
        raise NotImplementedError(
            "Multiple histograms with contrasts is not yet implemented."
        )
    else:
        ax = sns.distplot(data[x], **hist_kwargs)
        ax.set_title(f"Histogram of {x}")
    return ax


def viz_multiple_histogram(data, columns=None, **kwargs):
    """Plot histograms for multiple features.

    Args:
        data (DataFrame): The data
        columns: The columns to plot

    Returns:
        Seaborn Axis Object
    """
    raise NotImplementedError()


def _viz_violin(data, x, contrast=None, **kwargs):
    """Plot a single violin plot.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the categorical column to use for multiple contrasts.

    Returns:
        Seaborn Axis Object
    """
    if x not in data.select_dtypes("number").columns:
        raise ValueError("x must be numeric column")

    default_violin_kwargs = {"cut": 0}
    violin_kwargs = {**default_violin_kwargs, **(kwargs or {})}
    if contrast:
        data[contrast] = data[contrast].astype("category")
        ax = sns.violinplot(x=x, y=contrast, data=data, **violin_kwargs)
    else:
        ax = sns.violinplot(x=x, data=data, **violin_kwargs)
    return ax


def viz_multiple_violin(data, x, contrast=None, columns=None, **kwargs):
    """Plot violin plots for multiple features.

    Args:
        data (DataFrame): The data
        columns: The columns to plot

    Returns:
        Seaborn Axis Object
    """
    raise NotImplementedError()


def viz_bar(data, x, contrast=None, **kwargs):
    """Plot a bar chart (count plot) for categorical features.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the column to use for multiple histograms.

    Returns:
        Seaborn Axis Object
    """
    default_bar_kwargs = {"orient": "h"}
    plt.figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    bar_kwargs = {**default_bar_kwargs, **(kwargs or {})}
    if contrast:
        ax = sns.countplot(x=x, hue=contrast, data=data, **bar_kwargs)
        ax.set_title(f"{x} vs {contrast}")
    else:
        ax = sns.countplot(x=x, data=data, **bar_kwargs)
        ax.set_title(f"{x}")
    return ax


def viz_multiple_bar(data, columns=None, **kwargs):
    """Plot bar charts for multiple features.

    Args:
        data (DataFrame): The data
        columns: The columns to plot

    Returns:
        Seaborn Axis Object
    """
    raise NotImplementedError()
