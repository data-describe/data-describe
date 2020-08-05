import matplotlib.pyplot as plt
import seaborn as sns

from data_describe.config._config import get_option


def viz_distribution(data, **kwargs):
    """Visualize all distributions."""
    pass


def viz_histogram(data, x, contrast=None, **kwargs):
    """Plot a single histogram.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the column to use for multiple histograms.

    Returns:
        Seaborn Axis Object
    """
    # TODO (haishiro): Use histplot from seaborn PR #2125
    default_hist_kwargs = {"kde": False, "rug": False}
    hist_kwargs = {**default_hist_kwargs, **(kwargs or {})}
    plt.figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    if contrast:
        g = sns.FacetGrid(data, hue=contrast)
        ax = g.map(sns.distplot, x, **hist_kwargs)
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"Histogram of {x}")
    else:
        ax = sns.distplot(data[x], **hist_kwargs)
        ax.set_title(f"Histogram of {x}")
    plt.ylabel("Count")
    plt.xlabel(x)
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


def viz_violin(data, x, contrast=None, **kwargs):
    """Plot a single violin plot.

    Args:
        data (DataFrame): The data
        x (str): The name of the column to plot.
        contrast (str, optional): The name of the column to use for multiple histograms.

    Returns:
        Seaborn Axis Object
    """
    default_violin_kwargs = {"cut": 0}
    plt.figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    violin_kwargs = {**default_violin_kwargs, **(kwargs or {})}
    if contrast:
        ax = sns.violinplot(y=x, x=contrast, data=data, **violin_kwargs)
        ax.set_title(f"Violin Plot of {x} vs {contrast}")
        plt.xticks(rotation=45, ha="right")
    else:
        ax = sns.violinplot(x, data=data, **violin_kwargs)
        ax.set_title(f"Violin Plot of {x}")
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
    default_bar_kwargs = {}
    plt.figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    bar_kwargs = {**default_bar_kwargs, **(kwargs or {})}
    if contrast:
        ax = sns.countplot(x=x, hue=contrast, data=data, **bar_kwargs)
        ax.set_title(f"Count Plot of {x} vs {contrast}")
        plt.xticks(rotation=45, ha="right")
    else:
        ax = sns.countplot(x=x, data=data, **bar_kwargs)
        ax.set_title(f"Count Plot of {x}")
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
