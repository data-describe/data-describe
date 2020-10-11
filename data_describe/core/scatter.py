from itertools import combinations
import warnings

import seaborn as sns

from data_describe._widget import BaseWidget
from data_describe.config._config import get_option
from data_describe.compat import _is_dataframe, _requires, _compat
from data_describe.backends import _get_compute_backend, _get_viz_backend


class ScatterWidget(BaseWidget):
    """Container for scatter plots.

    This class (object) is returned from the ``scatter_plots`` function. The
    attributes documented below can be accessed or extracted.

    Attributes:
        input_data: The input data.
        num_data: Numeric data only
        mode: {'diagnostic', 'matrix', 'all'} The visualization mode.
            **diagnostic**: Plots selected by scagnostics (scatter plot diagnostics)
            **matrix**: Generate the full scatter plot matrix
            **all**: Generate all individual scatter plots
        sample: The sampling method to use. Currently not used.
        diagnostics: The diagnostics from ``pyscagnostics.scagnostics``
        threshold: The scatter plot diagnostic threshold value [0,1] for returning a
            plot. Only used with "diagnostic" mode. For example, ``{"Outlying": 0.9}``
            returns plots with outlier metrics above 0.9. See
            ``pyscagnostics.measure_names`` for a list of metrics.
            **If a number**: Returns all plots where at least one metric is above this threshold
            **If a dictionary**: Returns plots where the metric is above its threshold.
    """

    def __init__(
        self,
        input_data=None,
        num_data=None,
        mode=None,
        sample=None,
        diagnostics=None,
        threshold=None,
        compute_backend=None,
        viz_backend=None,
        **kwargs,
    ):
        """Data heatmap.

        Args:
            input_data: The input data.
            num_data: Numeric data only
            mode: {'diagnostic', 'matrix', 'all'} The visualization mode.
                **diagnostic**: Plots selected by scagnostics (scatter plot diagnostics)
                **matrix**: Generate the full scatter plot matrix
                **all**: Generate all individual scatter plots
            sample: The sampling method to use. Currently not used.
            diagnostics: The diagnostics from ``pyscagnostics.scagnostics``
            threshold: The scatter plot diagnostic threshold value [0,1] for returning a
                plot. Only used with "diagnostic" mode. For example, ``{"Outlying": 0.9}``
                returns plots with outlier metrics above 0.9. See
                ``pyscagnostics.measure_names`` for a list of metrics.
                **If a number**: Returns all plots where at least one metric is above this threshold
                **If a dictionary**: Returns plots where the metric is above its threshold.
            compute_backend: The compute backend.
            viz_backend: The visualization backend.
        """
        super(ScatterWidget, self).__init__(**kwargs)
        self.input_data = input_data
        self.num_data = num_data
        self.mode = mode
        self.sample = sample
        self.diagnostics = diagnostics
        self.threshold = threshold
        self.compute_backend = compute_backend
        self.viz_backend = viz_backend
        self.kwargs = kwargs

    def __str__(self):
        return "data-describe Scatter Plot Widget"

    def __repr__(self):
        return "data-describe Scatter Plot Widget"

    def show(self, viz_backend=None, **kwargs):
        """The default display for this output.

        Displays a scatter plot matrix.

        Args:
            viz_backend: The visualization backend.
            **kwargs: Keyword arguments.

        Raises:
            ValueError: No numeric data to plot.

        Returns:
            The correlation matrix plot.
        """
        if self.num_data is None:
            raise ValueError("Could not find data to visualize.")

        viz_backend = viz_backend or self.viz_backend

        return _get_viz_backend(viz_backend).viz_scatter_plot(
            self.num_data,
            self.mode,
            self.sample,
            self.diagnostics,
            self.threshold,
            **{**self.kwargs, **kwargs},
        )


def scatter_plots(
    data,
    mode="matrix",
    sample=None,
    threshold=None,
    compute_backend=None,
    viz_backend=None,
    **kwargs,
):
    """Scatter plots of numeric data.

    Args:
        data: A Pandas data frame
        mode (str): {``diagnostic``, ``matrix``, ``all``} The visualization mode.

            * ``diagnostic``: Plots selected by scagnostics (scatter plot diagnostics)
            * ``matrix``: Generate the full scatter plot matrix
            * ``all``: Generate all individual scatter plots
        sample: The sampling method to use. Currently not used.
        threshold: The scatter plot diagnostic threshold value [0,1] for returning a
            plot. Only used with "diagnostic" mode. For example, ``{"Outlying": 0.9}``
            returns plots with outlier metrics above 0.9. See
            ``pyscagnostics.measure_names`` for a list of metrics.

            * If a number: Returns all plots where at least one metric is above this threshold
            * If a dictionary: Returns plots where the metric is above its threshold.
        compute_backend: The compute backend
        viz_backend: The vizualization backend
        **kwargs: Passed to the visualization framework

    Raises:
        ValueError: Invalid input data type.

    Returns:
        Scatter plot.
    """
    if not _is_dataframe(data):
        raise ValueError("Unsupported input data type")

    swidget = _get_compute_backend(compute_backend, data).compute_scatter_plot(
        data, mode, sample, threshold, **kwargs
    )

    swidget.compute_backend = compute_backend
    swidget.viz_backend = viz_backend
    return swidget


def _pandas_compute_scatter_plot(
    data, mode, sample, threshold, **kwargs
) -> ScatterWidget:
    """Compute scatter plot.

    Args:
        data: A Pandas data frame
        mode: {'diagnostic', 'matrix', 'all'} The visualization mode.
            **diagnostic**: Plots selected by scagnostics (scatter plot diagnostics)
            **matrix**: Generate the full scatter plot matrix
            **all**: Generate all individual scatter plots
        sample: The sampling method to use. Currently not used.
        threshold: The scatter plot diagnostic threshold value [0,1] for returning a
            plot. Only used with "diagnostic" mode. For example, ``{"Outlying": 0.9}``
            returns plots with outlier metrics above 0.9. See
            ``pyscagnostics.measure_names`` for a list of metrics.
            **If a number**: Returns all plots where at least one metric is above this threshold
            **If a dictionary**: Returns plots where the metric is above its threshold.
        **kwargs: Passed to the visualization framework

    Returns:
        ScatterWidget
    """
    num_data = data.select_dtypes(["number"])
    if mode == "diagnostic":
        diagnostics = _get_scagnostics(num_data)
        return ScatterWidget(
            input_data=data,
            num_data=num_data,
            mode=mode,
            sample=sample,
            diagnostics=diagnostics,
            threshold=threshold,
            **kwargs,
        )
    else:
        return ScatterWidget(
            input_data=data,
            num_data=num_data,
            mode=mode,
            sample=sample,
            **kwargs,
        )


@_requires("pyscagnostics")
def _get_scagnostics(data):
    """Scatterplot diagnostics."""
    return _compat["pyscagnostics"].scagnostics(data)


def _seaborn_viz_scatter_plot(data, mode, sample, diagnostics, threshold, **kwargs):
    """Scatter plots.

    Args:
        data: A Pandas data frame
        mode: {'diagnostic', 'matrix', 'all'} The visualization mode.
            **diagnostic**: Plots selected by scagnostics (scatter plot diagnostics)
            **matrix**: Generate the full scatter plot matrix
            **all**: Generate all individual scatter plots
        sample: The sampling method to use. Currently not used.
        diagnostics: The computed scatterplot diagnostics.
        threshold: The scatter plot diagnostic threshold value [0,1] for returning a
            plot. Only used with "diagnostic" mode. For example, ``{"Outlying": 0.9} ``
            returns plots with outlier metrics above 0.9. See
            ``pyscagnostics.measure_names`` for a list of metrics.
            **If a number**: Returns all plots where at least one metric is above this threshold
            **If a dictionary**: Returns plots where the metric is above its threshold.
        compute_backend: The compute backend
        viz_backend: The vizualization backend
        **kwargs: Passed to the visualization framework

    Raises:
        ValueError: Invalid input data type.

    Returns:
        Seaborn plot.
    """
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
            warnings.warn("No plots identified by diagnostics")

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
    default_dist_kwargs = {"kde": False}
    default_joint_kwargs.update(kwargs.get("joint_kwargs", {}))
    default_scatter_kwargs.update(kwargs.get("scatter_kwargs", {}))
    default_dist_kwargs.update(kwargs.get("dist_kwargs", {}))

    g = sns.JointGrid(x=data[xname], y=data[yname], **default_joint_kwargs)
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
