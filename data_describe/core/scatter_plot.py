from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends import _get_compute_backend, _get_viz_backend


def scatter_plots(
    data,
    mode="matrix",
    sample=None,
    threshold=None,
    compute_backend=None,
    viz_backend=None,
    **kwargs
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
        kwargs: Passed to the visualization framework

    Returns:
        The vizualization
    """
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")

    data = _get_compute_backend(compute_backend, data).compute_scatter_plot(
        data, mode, sample, threshold, **kwargs
    )

    return _get_viz_backend(viz_backend).viz_scatter_plot(
        data, mode, sample, threshold, **kwargs
    )
