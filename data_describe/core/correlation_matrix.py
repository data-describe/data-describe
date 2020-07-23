from data_describe.backends import _get_viz_backend, _get_compute_backend


def correlation_matrix(
    data,
    cluster=False,
    categorical=False,
    return_values=False,
    compute_backend=None,
    viz_backend="plotly",
):
    """Correlation matrix of numeric variables.

    Args:
        data: A pandas data frame
        cluster: If True, use clustering to reorder similar columns together

        categorical: If True, calculate categorical associations using Cramer's V, Correlation Ratio, and
            Point-biserial coefficient (aka Matthews correlation coefficient). All associations (including Pearson
            correlation) are in the range [0, 1]

        return_values: If True, return the correlation/association values manager

    Returns:
        Plotly figure
    """
    association_matrix = _get_compute_backend(
        compute_backend, data
    ).compute_correlation_matrix(data, cluster=cluster, categorical=categorical)

    if return_values:
        return association_matrix
    return _get_viz_backend(viz_backend).viz_plot_correlation_matrix(association_matrix)
