import warnings

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_describe.metrics.univariate import skewed, spikey
from data_describe.metrics.bivariate import heteroscedastic, varying


def distribution(
    data,
    plot_all=False,
    max_categories=20,
    spike=10,
    skew=3,
    hist_kwargs=None,
    violin_kwargs=None,
):
    """ Plots all "interesting" distribution plots

    Args:
        data: A pandas data frame
        plot_all: If True, plot all features without filtering for "interesting" features
        max_categories: Maximum categories to show in violin plots. Additional categories will be combined into the "__OTHER__" category.
        spike: The factor threshold for identifying spikey histograms
        skew: The skew threshold for identifying skewed histograms
        hist_kwargs: Keyword arguments to be passed to seaborn.distplot
        violin_kwargs: Keyword arguments to be passed to seaborn.violinplot

    Returns:
        Matplotlib graphics
    """
    if isinstance(data, pd.DataFrame):
        num = data.select_dtypes(["number"])
        cat = data[[col for col in data.columns if col not in num.columns]]
    else:
        raise NotImplementedError

    fig_hist = plot_histograms(
        num, plot_all, spike=spike, skew=skew, hist_kwargs=hist_kwargs
    )
    fig_violin = plot_violins(
        data, num, cat, max_categories, plot_all, violin_kwargs=violin_kwargs,
    )
    return fig_hist + fig_violin


def plot_histograms(data, plot_all, spike=10, skew=6, hist_kwargs=None):
    """ Makes histogram plots

    Args:
        data: A pandas data frame
        plot_all: If True, plot all histograms without filtering for skew or spikes
        spike: The factor threshold for identifying spikey histograms
        skew: The skew threshold for identifying skewed histograms
        hist_kwargs: Keyword arguments to be passed to seaborn.distplot

    Returns:
        Matplotlib graphics
    """
    fig = []
    for column in data.columns:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module="scipy",
                message=r"Using a non-tuple sequence for multidimensional indexing is deprecated",
            )
            if plot_all:
                fig.append(plot_histogram(data[column].dropna(), hist_kwargs))
            elif spikey(data[column].dropna(), factor=spike) or skewed(
                data[column].dropna(), threshold=skew
            ):
                fig.append(plot_histogram(data[column].dropna(), hist_kwargs))
        return fig


def plot_histogram(x, hist_kwargs=None):
    """ Make a single histogram plot

    Args:
        x: The 1-dimensional data, as a list or numpy array
        hist_kwargs: Keyword arguments to be passed to seaborn.distplot

    Returns:
        Matplotlib graphic
    """
    # plt.figure(figsize=(context.fig_width, context.fig_height))
    if hist_kwargs is None:
        hist_kwargs = {"kde": False, "rug": False}
    fig = sns.distplot(x, **hist_kwargs)
    fig.set_title("Histogram of " + x.name)
    plt.ylabel("Count")
    plt.xlabel(x.name)
    plt.show()
    plt.close()
    return fig


def plot_violins(
    data, num, cat, max_categories, plot_all=False, alpha=0.01, violin_kwargs=None,
):
    """ Makes violin plots

    Args:
        data: A pandas data frame
        num: The numeric feature names
        cat: The categorical feature names
        max_categories: Maximum categories to show in violin plots. Additional categories will be combined into
        plot_all: If True, plot all violin plots without variation or heteroscedascity
        alpha: The significance level for Levene's test and one-way ANOVA
        violin_kwargs: Keyword arguments to be passed to seaborn.violinplot

    Returns:
        Matplotlib graphics
    """
    fig = []
    for n in num:
        for c in cat:
            if plot_all:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        module="scipy",
                        message=r"Using a non-tuple sequence for multidimensional",
                    )
                    y, x = roll_up_categories(data[n], data[c], max_categories)
                    fig.append(plot_violin(x, y, data, violin_kwargs,))
            else:
                grp = split_by_category(data[[c, n]].dropna(), c, n)
                y, x = roll_up_categories(data[n], data[c], max_categories)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        module="scipy",
                        message=r"Using a non-tuple sequence for multidimensional",
                    )
                    if varying(grp, alpha=alpha) or heteroscedastic(grp, alpha=alpha):
                        fig.append(plot_violin(x, y, data, violin_kwargs))
    return fig


def plot_violin(x, y, data, violin_kwargs=None):
    """ Make a single violin plot

    Args:
        x: The x (categorical) data vector or name string
        y: The y (numeric) data vector or name string
        data: The data
        violin_kwargs: Keyword arguments to be passed to seaborn.violinplot
        context: The context

    Returns:
        Matplotlib graphic
    """
    # plt.figure(figsize=(context.fig_width, context.fig_height))
    if violin_kwargs is None:
        violin_kwargs = {"cut": 0}
    fig = sns.violinplot(x, y, data=data, **violin_kwargs)
    fig.set_title("Violin Plot of " + y.name + " vs " + x.name)
    plt.xticks(rotation=45, ha="right")
    plt.show()
    plt.close()
    return fig


def split_by_category(df, category, num):
    """ Splits the numeric feature `num` by the categorical feature `category`

    Args:
        df: A pandas data frame
        category: The name of the categorical feature, as a string
        num: The name of the numeric feature, as a string

    Returns:
        A list of lists, where each list is the numeric data for a category
    """
    g = df.groupby(category)
    return [x[1][num].to_numpy() for x in g]


def roll_up_categories(num, cat, max_categories=20):
    """ Combine "extra" categories into "__OTHER__"

    Args:
        num: The numeric data
        cat: The categorical data
        max_categories: The maximum number of categories, including __OTHER__

    Returns:
        A tuple of the numeric data and the categorical data, rolled up into `max_categories`
    """
    if max_categories is not None:
        groups = pd.concat([num, cat], axis=1).groupby(cat.name).agg("count")
        if groups.shape[0] > max_categories:
            groups = groups.sort_values(num.name, ascending=False)
            groups_to_combine = groups.iloc[max_categories - 1 :, :].index.values
            return num, cat.map(lambda x: "__OTHER__" if x in groups_to_combine else x)
        else:
            return num, cat
    else:
        return num, cat
