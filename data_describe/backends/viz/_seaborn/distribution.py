from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import seaborn as sns

from data_describe.config._config import get_option


def viz_distribution(data, x: str, contrast: str = None, **kwargs):
    """Plots the distribution.

    Args:
        data (DataFrame): The data
        x (str): The column to visualize.
        contrast (str, optional): The column to use as a contrast.
        **kwargs: Keyword arguments passed to underlying plot functions.

    Returns:
        matplotlib.figure.Figure
    """
    if x in data.select_dtypes("number").columns:
        return viz_numeric(data, x, contrast, **kwargs)
    else:
        return viz_categorical(data, x, contrast, **kwargs)


def viz_numeric(
    data,
    x: str,
    contrast: str = None,
    mode: str = "combo",
    hist_kwargs: dict = None,
    violin_kwargs: dict = None,
    **kwargs,
):
    """Plots a histogram/violin plot.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot
        contrast (str, optional): The name of the categorical column to use for multiple contrasts.
        mode (str): {'combo', 'violin', 'hist'} The type of plot to display. Defaults to a combined histogram/violin plot.
        hist_kwargs (dict, optional): Keyword args for seaborn.distplot.
        violin_kwargs (dict, optional): Keyword args for seaborn.violinplot.
        **kwargs: Keyword args to be passed to all underlying plotting functions.
    """
    hist_kwargs = hist_kwargs or {}
    violin_kwargs = violin_kwargs or {}
    fig = Figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    if mode == "combo":
        gs = GridSpec(nrows=5, ncols=1)
        ax1 = fig.add_subplot(gs.new_subplotspec((0, 0), 1, 1))
        ax2 = fig.add_subplot(gs.new_subplotspec((1, 0), 4, 1))

        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        _viz_histogram(data, x, contrast=None, ax=ax1, **hist_kwargs)
        _viz_violin(data, x, contrast, ax=ax2, **violin_kwargs)
        ax1.set_title(x)
        return fig
    elif mode == "hist":
        ax = fig.add_subplot()
        _viz_histogram(data, x, contrast=None, ax=ax, **hist_kwargs, **kwargs)
        ax.set_title(x)
        return fig
    elif mode == "violin":
        ax = fig.add_subplot()
        _viz_violin(data, x, contrast=None, ax=ax, **violin_kwargs, **kwargs)
        ax.set_title(x)
        return fig
    else:
        raise ValueError("Unknown value for 'mode' plot type")


def viz_categorical(
    data, x: str, contrast: str = None, bar_kwargs: dict = None,
):
    """Plots a bar count plot for a categorical feature.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot
        contrast (str, optional): The name of the categorical column to use for multiple contrasts.
        bar_kwargs (dict, optional): Keyword args for seaborn.countplot.
    """
    bar_kwargs = bar_kwargs or {}
    fig = Figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    ax = fig.add_subplot()
    _viz_bar(data, x, contrast, ax=ax, **bar_kwargs)
    ax.set_title(x)
    return fig


def _viz_histogram(data, x: str, contrast: str = None, **kwargs):
    """Plot a single histogram.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the categorical column to use for multiple contrasts.
        **kwargs: Keyword arguments passed to seaborn.distplot

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


def _viz_violin(data, x: str, contrast: str = None, **kwargs):
    """Plot a single violin plot.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the categorical column to use for multiple contrasts.
        **kwargs: Keyword arguments passed to seaborn.violinplot

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


def _viz_bar(data, x: str, contrast: str = None, **kwargs):
    """Plot a bar chart (count plot) for categorical features.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the column to use for multiple histograms.
        **kwargs: Keyword arguments passed to seaborn.countplot

    Returns:
        Seaborn Axis Object
    """
    default_bar_kwargs = {"orient": "h"}
    bar_kwargs = {**default_bar_kwargs, **(kwargs or {})}
    if contrast:
        ax = sns.countplot(x=x, hue=contrast, data=data, **bar_kwargs)
    else:
        ax = sns.countplot(x=x, data=data, **bar_kwargs)
    return ax
