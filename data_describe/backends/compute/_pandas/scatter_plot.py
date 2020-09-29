from data_describe.compat import _compat, requires


def compute_scatter_plot(data, mode, sample, threshold, **kwargs):
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

        kwargs: Passed to the visualization framework

    Returns:
        data: The data
        diagnostics: The diagnostic values
    """
    data = data.select_dtypes(["number"])
    if mode == "diagnostic":
        return _compute_scagnostics(data)
    else:
        return data, None


@requires("pyscagnostics")
def _compute_scagnostics(data):
    diagnostics = _compat["pyscagnostics"].scagnostics(data)
    return data, diagnostics
