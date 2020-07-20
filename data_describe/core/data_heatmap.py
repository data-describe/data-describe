from data_describe.backends import _get_viz_backend, _get_compute_backend


def data_heatmap(data, missing=False, compute_backend=None, viz_backend=None, **kwargs):
    """ Generate a data heatmap showing standardized data or missing values

    The data heatmap shows an overview of numeric features that have been standardized.

    Args:
        data: A pandas data frame
        missing: If True, show only missing values
        interactive: If True, return an interactive visualization (using Plotly). Otherwise, uses Seaborn.


    Returns:
        Visualization
    """
    data, colnames = _get_compute_backend(compute_backend, data).process_data_heatmap(
        data, missing=missing, **kwargs
    )

    return _get_viz_backend(viz_backend).viz_data_heatmap(
        data.transpose(), colnames=colnames, missing=missing, **kwargs
    )
