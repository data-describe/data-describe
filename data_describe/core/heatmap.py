from typing import Tuple, List, Any

import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode, iplot
from IPython import get_ipython
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler

from data_describe.config._config import get_option
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends import _get_viz_backend, _get_compute_backend


def data_heatmap(data, missing=False, compute_backend=None, viz_backend=None, **kwargs):
    """Generate a data heatmap showing standardized data or missing values.

    The data heatmap shows an overview of numeric features that have been standardized.

    Args:
        data: A pandas data frame
        missing (bool): If True, show only missing values
        compute_backend: The compute backend.
        viz_backend: The visualization backend.
        **kwargs: Keyword arguments

    Returns:
        The data heatmap.
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
        missing (bool): If True, uses missing values instead
        **kwargs: Keyword arguments.

    Raises:
        ValueError: Invalid input data type.

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


def _plotly_viz_data_heatmap(
    data, colnames: List[str], missing: bool = False, **kwargs
):
    """Plots the data heatmap.

    Args:
        data: The dataframe
        colnames (List[str]): The column names, used for tick labels
        missing (bool): If True, plots missing values instead
        **kwargs: Keyword arguments.

    Returns:
        The data heatmap as a Plotly figure.
    """
    data_fig = go.Heatmap(
        z=np.flip(data.values, axis=0),
        x=list(range(data.shape[0])),
        y=list(colnames[::-1]),
        ygap=1,
        zmin=-3 if not missing else 0,
        zmax=3 if not missing else 1,
        colorscale="viridis" if not missing else "greys",
        colorbar={"title": "z-score (bounded)" if not missing else "Missing"},
    )

    figure = go.Figure(
        data=[data_fig],
        layout=go.Layout(
            autosize=False,
            title={
                "text": "Data Heatmap",
                "font": {"size": get_option("display.plotly.title_size")},
            },
            width=get_option("display.plotly.fig_width"),
            height=get_option("display.plotly.fig_height"),
            xaxis=go.layout.XAxis(ticks="", title="Record #", showgrid=False),
            yaxis=go.layout.YAxis(
                ticks="", title="Variable", automargin=True, showgrid=False
            ),
            plot_bgcolor="rgb(0,0,0,0)",
            paper_bgcolor="rgb(0,0,0,0)",
        ),
    )

    if get_ipython() is not None:
        init_notebook_mode(connected=True)
        return iplot(figure, config={"displayModeBar": False})
    else:
        return figure


def _seaborn_viz_data_heatmap(
    data, colnames: List[str], missing: bool = False, **kwargs
):
    """Plots the data heatmap.

    Args:
        data: The dataframe
        colnames: The column names, used for tick labels
        missing: If True, plots missing values instead
        kwargs: Keyword arguments passed to seaborn.heatmap

    Returns:
        The seaborn figure
    """
    plot_options = {
        "cmap": "viridis" if not missing else "Greys",
        "robust": True,
        "center": 0 if not missing else 0.5,
        "xticklabels": False,
        "yticklabels": colnames,
        "cbar_kws": {"shrink": 0.5},
        "vmin": -3 if not missing else 0,
        "vmax": 3 if not missing else 1,
    }

    plot_options.update(kwargs)

    plt.figure(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )
    heatmap = sns.heatmap(data, **plot_options)
    plt.title("Data Heatmap")
    plt.ylabel("Variable")
    plt.xlabel("Record #")

    return heatmap
