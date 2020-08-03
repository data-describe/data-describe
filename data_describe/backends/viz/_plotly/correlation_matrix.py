import numpy as np
from IPython import get_ipython
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from scipy.stats import percentileofscore

from data_describe.utilities.colorscale import color_fade, rgb_to_str
from data_describe.config._config import get_option


def viz_plot_correlation_matrix(association_matrix):
    """Plot the heatmap for the association matrix.

    Args:
        association_matrix: The association matrix

    Returns:
        The plotly figure
    """
    # Plot lower left triangle
    x_ind, y_ind = np.triu_indices(association_matrix.shape[0])
    corr = association_matrix.to_numpy()
    for x, y in zip(x_ind, y_ind):
        corr[x, y] = None

    # Set up the color scale
    blue_anchor = (65, 124, 167)
    white_anchor = (242, 242, 242)
    red_anchor = (217, 58, 70)

    vmin = min(corr.flatten()[~np.isnan(corr.flatten())])
    vmax = max(corr.flatten()[~np.isnan(corr.flatten())])

    if vmin > 0:
        cmin = color_fade(white_anchor, red_anchor, 1 - vmin)
        cmax = color_fade(white_anchor, red_anchor, 1 - vmax)
        cscale = [[0, rgb_to_str(cmin)], [1.0, rgb_to_str(cmax)]]
    elif vmax < 0:
        cmin = color_fade(blue_anchor, white_anchor, -vmin)
        cmax = color_fade(blue_anchor, white_anchor, -vmax)
        cscale = [[0, rgb_to_str(cmin)], [1.0, rgb_to_str(cmax)]]
    else:
        cmin = color_fade(blue_anchor, white_anchor, -vmin)
        cmax = color_fade(white_anchor, red_anchor, 1 - vmax)
        corr_values = corr[~np.isnan(corr)].flatten()
        z_val = percentileofscore(corr_values, 0.0) / 100.0
        cscale = [
            [0, rgb_to_str(cmin)],
            [z_val, rgb_to_str(white_anchor)],
            [1.0, rgb_to_str(cmax)],
        ]

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

    if get_ipython() is not None:
        init_notebook_mode(connected=True)
        return iplot(fig, config={"displayModeBar": False})
    else:
        return fig
