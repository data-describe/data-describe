from itertools import combinations

import seaborn as sns

from data_describe.config._config import get_option


def viz_scatter_plot(data, mode, sample, threshold, **kwargs):
    """Scatter plots.

    Args:
        data: A Pandas data frame

        mode: The visualization mode
            diagnostic: Plots selected by scagnostics (scatter plot diagnostics)
            matrix: Generate the full scatter plot matrix
            all: Generate all individual scatter plots

        sample: The sampling method to use

        threshold: The scatter plot diagnostic threshold value [0,1] for returning a plot. Only used with "diagnostic" mode.
            If a number: Returns all plots where at least one metric is above this threshold
            If a dictionary: Returns plots where the metric is above its threshold.
            For example, {"Outlying": 0.9} returns plots with outlier metrics above 0.9.
            See pyscagnostics.measure_names for a list of metrics.

        kwargs: Passed to the visualization framework

    Returns:
        The seaborn visualization
    """
    data, diagnostics, *_ = data
    if mode == "matrix":
        fig = sns.pairplot(data)
        return fig
    elif mode == "all":
        col_pairs = combinations(data.columns, 2)
        fig = []
        for p in col_pairs:
            fig.append(_scatter_plot(data, p[0], p[1], **kwargs))
        return fig
    elif mode == "diagnostic":
        if threshold is not None:
            diagnostics = _filter_threshold(diagnostics, threshold)

        if len(diagnostics) == 0:
            raise UserWarning("No plots identified by diagnostics")

        fig = []
        for d in diagnostics:
            fig.append(_scatter_plot(data, d[0], d[1], **kwargs))

        return fig
    else:
        raise ValueError(f"Unknown plot mode: {mode}")


def _scatter_plot(data, xname, yname, **kwargs):
    """Generate one scatter (joint) plot.

    Args:
        data: A Pandas data frame
        xname: The x-axis column name
        yname: The y-axis column name
        kwargs: Keyword arguments

    Returns:
        The Seaborn figure
    """
    default_joint_kwargs = {
        "height": max(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    }
    default_scatter_kwargs = {}
    default_dist_kwargs = {}
    default_joint_kwargs.update(kwargs.get("joint_kwargs", {}))
    default_scatter_kwargs.update(kwargs.get("scatter_kwargs", {}))
    default_dist_kwargs.update(kwargs.get("dist_kwargs", {}))

    g = sns.JointGrid(data=data, x=xname, y=yname, **default_joint_kwargs)
    g = g.plot_joint(sns.scatterplot, **default_scatter_kwargs)
    g = g.plot_marginals(sns.histplot, **default_dist_kwargs)
    return g


def _filter_threshold(diagnostics, threshold=0.85):
    """Filter the plots by scatter plot diagnostic threshold.

    Args:
        diagnostics: The diagnostics generator from pyscagnostics

        threshold: The scatter plot diagnostic threshold value [0,1] for returning a plot
            If a number: Returns all plots where at least one metric is above this threshold
            If a dictionary: Returns plots where the metric is above its threshold
            For example, {"Outlier": 0.9} returns plots with outlier metrics above 0.9
            The available metrics are: Outlier, Convex, Skinny, Skewed, Stringy, Straight, Monotonic, Clumpy, Striated

    Returns:
        A dictionary of pairs that match the filter
    """
    if isinstance(threshold, dict):
        return [
            d
            for d in diagnostics
            if all([d[2][0][m] >= threshold[m] for m in threshold.keys()])
        ]
    else:
        return [
            d for d in diagnostics if any([v > threshold for v in d[2][0].values()])
        ]
