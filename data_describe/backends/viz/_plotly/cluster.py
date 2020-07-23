import plotly.graph_objs as go
import plotly.offline as po

from data_describe.compat import _IN_NOTEBOOK
from data_describe.config._config import get_option


def viz_cluster(data, method: str, xlabel: str = None, ylabel: str = None, **kwargs):
    """Visualize clusters using Plotly.

    Args:
        data (DataFrame): The data
        method (str): The clustering method, to be used as the plot title
        xlabel (str, optional): The x-axis label. Defaults to "Reduced Dimension 1".
        ylabel (str, optional): The y-axis label. Defaults to "Reduced Dimension 2".

    Returns:
        Plotly plot
    """
    xlabel = xlabel or "Reduced Dimension 1"
    ylabel = ylabel or "Reduced Dimension 2"
    labels = data["clusters"].unique()

    trace_list = []
    for i in labels:
        if int(i) < 0:
            trace = go.Scatter(
                x=data.loc[data["clusters"] == i, "x"],
                y=data.loc[data["clusters"] == i, "y"],
                name="Noise",
                mode="markers",
                marker=dict(size=10, color="grey", line=dict(width=1)),
            )
            trace_list.append(trace)
        else:
            trace = go.Scatter(
                x=data.loc[data["clusters"] == i, "x"],
                y=data.loc[data["clusters"] == i, "y"],
                name=f"Cluster #{i}",
                mode="markers",
                marker=dict(size=10, colorscale="earth", line=dict(width=1)),
            )
            trace_list.append(trace)

    layout = dict(
        yaxis=dict(zeroline=False, title=data.columns[0]),
        xaxis=dict(zeroline=False, title=data.columns[1]),
        yaxis_title=ylabel,
        xaxis_label=xlabel,
        autosize=False,
        width=int(get_option("display.fig_width"))
        * 100,  # TODO (haishiro): Smarter defaults for fig size
        height=int(get_option("display.fig_height"))
        * 100,  # TODO (haishiro): Smarter defaults for fig size
        title={"text": "{} Cluster".format(method), "font": {"size": 25}},
    )

    fig = go.Figure(dict(data=trace_list, layout=layout))

    if _IN_NOTEBOOK:
        po.init_notebook_mode(connected=True)
        return po.iplot(fig)
    else:
        return fig
