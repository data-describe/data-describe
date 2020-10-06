from itertools import combinations

import seaborn as sns
from pyscagnostics import scagnostics

from data_describe.config._config import get_option
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends import _get_compute_backend, _get_viz_backend


def scatter_plots(
    data,
    mode="matrix",
    sample=None,
    threshold=None,
    compute_backend=None,
    viz_backend=None,
    **kwargs,
):
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
        compute_backend: The compute backend
        viz_backend: The vizualization backend
        **kwargs: Passed to the visualization framework

    Raises:
        ValueError: Invalid input data type.

    Returns:
        Scatter plot.
    """
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")

    data = _get_compute_backend(compute_backend, data).compute_scatter_plot(
        data, mode, sample, threshold, **kwargs
    )

    return _get_viz_backend(viz_backend).viz_scatter_plot(
        data, mode, sample, threshold, **kwargs
    )


def _pandas_compute_scatter_plot(data, mode, sample, threshold, **kwargs):
    """Compute scatter plot.

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
        **kwargs: Passed to the visualization framework

    Returns:
        data: The data
        diagnostics: The diagnostic values
    """
    data = data.select_dtypes(["number"])
    if mode == "diagnostic":
        diagnostics = scagnostics(data)
        return data, diagnostics
    else:
        return data, None


def _seaborn_viz_scatter_plot(data, mode, sample, threshold, **kwargs):
    """Scatter plots.

    Args:
        data: A Pandas data frame
        mode: {'diagnostic', 'matrix', 'all} The visualization mode
            - diagnostic: Plots selected by scagnostics (scatter plot diagnostics)
            - matrix: Generate the full scatter plot matrix
            - all: Generate all individual scatter plots
        sample: The sampling method to use
        threshold: The scatter plot diagnostic threshold value [0,1] for returning a
            plot. Only used with "diagnostic" mode. For example, {"Outlying": 0.9}
            returns plots with outlier metrics above 0.9. See
            `pyscagnostics.measure_names` for a list of metrics.
            - If a number: Returns all plots where at least one metric is above this threshold
            - If a dictionary: Returns plots where the metric is above its threshold.
        **kwargs: Passed to the visualization framework

    Raises:
        ValueError: Unknown plot `mode`.
        UserWarning: No plots identified by diagnostics.

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
    default_dist_kwargs = {"kde": False, "rug": False}
    default_joint_kwargs.update(kwargs.get("joint_kwargs", {}))
    default_scatter_kwargs.update(kwargs.get("scatter_kwargs", {}))
    default_dist_kwargs.update(kwargs.get("dist_kwargs", {}))

    g = sns.JointGrid(data[xname], data[yname], **default_joint_kwargs)
    g = g.plot_joint(sns.scatterplot, **default_scatter_kwargs)
    g = g.plot_marginals(sns.displot, **default_dist_kwargs)
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
