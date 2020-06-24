import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_describe.backends import _get_viz_backend


def data_heatmap(data, missing=False, compute_backend=None, viz_backend=None, **kwargs):
    """ Generate a data heatmap showing standardized data and/or missing values

    The data heatmap shows an overview of numeric features that have been standardized.

    Args:
        data: A pandas data frame
        missing: If True, show only missing values
        interactive: If True, return an interactive visualization (using Plotly). Otherwise, uses Seaborn.
        context: The context

    Returns:
        Visualization
    """
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(["number"])
        colnames = data.columns.values
    else:
        raise ValueError("Unsupported input data type")

    if missing:
        data = data.isna().astype(int)
    else:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    return _get_viz_backend(viz_backend).plot_data_heatmap(
        data.transpose(), colnames=colnames, missing=missing, **kwargs
    )
