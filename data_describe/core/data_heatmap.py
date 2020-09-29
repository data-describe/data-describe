from typing import Tuple, List, Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends import _get_viz_backend, _get_compute_backend


def data_heatmap(data, missing=False, compute_backend=None, viz_backend=None, **kwargs):
    """Generate a data heatmap showing standardized data or missing values.

    The data heatmap shows an overview of numeric features that have been standardized.

    Args:
        data: A pandas data frame
        missing: If True, show only missing values
        interactive: If True, return an interactive visualization (using Plotly). Otherwise, uses Seaborn.


    Returns:
        Visualization
    """
    data, colnames = _get_compute_backend(compute_backend, data).compute_data_heatmap(
        data, missing=missing, **kwargs
    )

    return _get_viz_backend(viz_backend).viz_data_heatmap(
        data, colnames=colnames, missing=missing, **kwargs
    )


def _pandas_compute_data_heatmap(
    data, missing: bool = False, **kwargs
) -> Tuple[Any, List[str]]:
    """Pre-processes data for the data heatmap.

    Values are standardized (removing the mean and scaling to unit variance).
    If `missing` is set to True, the dataframe flags missing records using 1/0.

    Args:
        data: The dataframe
        missing: If True, uses missing values instead
        kwargs: Not implemented

    Returns:
        (dataframe, column_names)
    """
    if isinstance(data, _DATAFRAME_TYPE):
        data = data.select_dtypes(["number"])
        colnames = data.columns.values
    else:
        raise ValueError("Unsupported input data type")

    if missing:
        data = data.isna().astype(int)
    else:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data.transpose(), colnames
