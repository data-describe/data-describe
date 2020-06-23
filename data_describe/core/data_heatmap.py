import logging

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from plotly.offline import init_notebook_mode, iplot

from data_describe.utilities.contextmanager import _context_manager


@_context_manager
def data_heatmap(data, missing=False, interactive=True, context=None):
    """ Generate a data heatmap showing standardized data and/or missing values

    The data heatmap shows an overview of numeric features that have been standardized.

    Args:
        data: A pandas data frame
        missing: If True, show only missing values
        interactive: If True, return an interactive visualization (using Plotly). Otherwise, uses Seaborn.
        context: The context

    Returns:
        Plotly graphic
    """
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(["number"])
        colnames = data.columns.values
        data = data.to_numpy()
    elif isinstance(data, np.ndarray):
        logging.warning("Numpy type not fully implemented")
        colnames = range(data.shape[1])
    else:
        raise ValueError("Unsupported input data type")

    if missing:
        nulls = np.empty(data.shape) * np.nan
        nulls[np.isnan(data)] = 1

        if interactive:
            data_fig = go.Heatmap(
                z=np.flip(nulls.T, axis=0),
                x=list(range(data.shape[0])),
                y=colnames[::-1],
                ygap=2,
                showscale=False,
                colorscale=[[0.0, "rgb(255,0,0)"], [1.0, "rgb(255,0,0)"]],
            )
            plotly_fig = plotly_data_heatmap(data_fig, context)
            return display_plotly(plotly_fig)
        else:
            # Set up the matplotlib figure
            plt.figure(figsize=(context.fig_width, context.fig_height))
            seaheatmap = sns.heatmap(
                np.flip(nulls.T, axis=0),
                cmap="PuRd",
                robust=True,
                center=0,
                xticklabels=False,
                yticklabels=colnames,
                cbar_kws={"shrink": 0.5},
            )
            plt.title("Data Heatmap")
            plt.ylabel("Variable")
            plt.xlabel("Record #")
            return seaheatmap
    else:
        scaler = StandardScaler()
        data_std = scaler.fit_transform(data)

        data_std = data_std.T

        if interactive:
            data_fig = go.Heatmap(
                z=np.flip(data_std, axis=0),
                x=list(range(data.shape[0])),
                y=list(colnames[::-1]),
                ygap=1,
                zmin=-3,
                zmax=3,
                colorscale="Viridis",
                colorbar={"title": "z-score (bounded)"},
            )

            plotly_fig = plotly_data_heatmap(data_fig, context)
            return display_plotly(plotly_fig)
        else:
            # Set up the matplotlib figure
            plt.figure(figsize=(context.fig_width, context.fig_height))
            seaheatmap = sns.heatmap(
                data_std,
                cmap="viridis",
                robust=True,
                center=0,
                xticklabels=False,
                yticklabels=colnames,
                cbar_kws={"shrink": 0.5},
            )
            plt.title("Data Heatmap")
            plt.ylabel("Variable")
            plt.xlabel("Record #")
            return seaheatmap


def plotly_data_heatmap(data_figure, context):
    """Defines the Plotly figure layout and style for Data Heatmap

    Args:
        data_figure: The Plotly data object, as a dictionary

    Returns:
        Plotly Figure object
    """
    return go.Figure(
        data=[data_figure],
        layout=go.Layout(
            autosize=False,
            title={"text": "Data Heatmap", "font": {"size": 25}},
            width=context.viz_size,
            height=context.viz_size,
            xaxis=go.layout.XAxis(ticks="", title="Record #", showgrid=False),
            yaxis=go.layout.YAxis(
                ticks="", title="Variable", automargin=True, showgrid=False
            ),
            plot_bgcolor="rgb(0,0,0,0)",
            paper_bgcolor="rgb(0,0,0,0)",
        ),
    )


def display_plotly(plotly_obj):
    """ Displays the interactive plotly figure in the notebook

    Args:
        plotly_obj: The Plotly Figure object

    Returns:
        The interactive plot
    """
    if get_ipython() is not None:
        init_notebook_mode(connected=True)
        return iplot(plotly_obj, config={"displayModeBar": False})
    else:
        raise EnvironmentError(
            "Could not detect IPython: Unable to display interactive plot."
        )
