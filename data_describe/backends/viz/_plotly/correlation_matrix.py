import numpy as np
import plotly.graph_objs as go
import plotly.offline as po

from data_describe.compat import _IN_NOTEBOOK
from data_describe.config._config import get_option
from data_describe.misc.colors import get_p_RdBl_cmap, mpl_to_plotly_cmap


def viz_correlation_matrix(association_matrix):
    """Plot the heatmap for the association matrix.

    Args:
        association_matrix (DataFrame): The association matrix

    Returns:
        The plotly figure
    """
    # Plot lower left triangle
    x_ind, y_ind = np.triu_indices(association_matrix.shape[0])
    corr = association_matrix.to_numpy()
    for x, y in zip(x_ind, y_ind):
        corr[x, y] = None

    # Set up the color scale
    cscale = mpl_to_plotly_cmap(get_p_RdBl_cmap())

    # Generate a custom diverging colormap
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=np.flip(corr, axis=0),
                x=association_matrix.columns.values,
                y=association_matrix.columns.values[::-1],
                connectgaps=False,
                xgap=2,
                ygap=2,
                colorscale=cscale,
                colorbar={"title": "Strength"},
            )
        ],
        layout=go.Layout(
            autosize=False,
            width=get_option("display.plotly.fig_width"),
            height=get_option("display.plotly.fig_height"),
            title={
                "text": "Correlation Matrix",
                "font": {"size": get_option("display.plotly.title_size")},
            },
            xaxis=go.layout.XAxis(
                automargin=True, tickangle=270, ticks="", showgrid=False
            ),
            yaxis=go.layout.YAxis(automargin=True, ticks="", showgrid=False),
            plot_bgcolor="rgb(0,0,0,0)",
            paper_bgcolor="rgb(0,0,0,0)",
        ),
    )

    if _IN_NOTEBOOK:
        po.init_notebook_mode(connected=True)
        return po.iplot(fig, config={"displayModeBar": False})
    else:
        return fig
