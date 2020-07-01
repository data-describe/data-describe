from plotly.offline import init_notebook_mode, iplot
from IPython import get_ipython
import plotly.graph_objs as go

from data_describe.config._config import get_option


def plot_data_heatmap(data, colnames, missing=False, **kwargs):
    data_fig = go.Heatmap(
        z=data,
        x=list(range(data.shape[0])),
        y=list(colnames[::-1]),
        ygap=1,
        zmin=-3,
        zmax=3,
        colorscale="Viridis",
        colorbar={"title": "z-score (bounded)"},
    )

    figure = go.Figure(
        data=[data_fig],
        layout=go.Layout(
            autosize=False,
            title={"text": "Data Heatmap", "font": {"size": 25}},
            width=get_option("display.fig_width") * 100,  # TODO: Separate size config for each backend
            height=get_option("display.fig_height") * 100,  # TODO: Separate size config for each backend
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