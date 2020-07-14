import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
from IPython import get_ipython

from data_describe.config._config import get_option


def viz_plot_time_series(df=None, cols=None, result=None, decompose=False):
    if isinstance(cols, list):
        data = [go.Scatter(x=df.index, y=df[c]) for c in cols]
        fig = go.Figure(data=data, layout=figure_layout(ylabel=str))
    elif isinstance(cols, str):
        fig = go.Figure(
            data=go.Scatter(x=df, y=df[cols]), layout=figure_layout(y_label=str)
        )
    elif decompose:
        fig = viz_decomposition(result)  # need to pass into figure
    if get_ipython() is not None:  # remove none?
        init_notebook_mode(connected=True)
        return iplot(fig, config={"displayModeBar": False})
    else:
        return fig


def viz_decomposition(result):
    decompose_results = [result.observed, result.trend, result.seasonal, result.resid]
    fig = make_subplots(rows=4, cols=1, layout=figure_layout())

    for idx, timeseries in enumerate(decompose_results, 1):
        fig.add_trace(go.Scatter(y=timeseries), row=idx, col=1)
    return fig


def viz_plot_autocorrelation(data, plot_type="acf"):
    if plot_type == "acf":
        data = [go.Bar(y=data)]
    elif plot_type == "pacf":
        data = [go.Bar(y=data)]
    else:
        raise ValueError("Unsupported input data type")
    return go.figure(data=data, layout=figure_layout("Autocorrelation Plot"))


def figure_layout(title, xlabel="Date", ylabel="Variable"):
    layout = go.Layout(
        title={"text": title, "font": {"size": 25}},
        width=get_option("display.fig_width") * 100,
        height=get_option("display.fig_height") * 100,
        xaxis=go.layout.XAxis(ticks="", title=xlabel, showgrid=True),
        yaxis=go.layout.YAxis(ticks="", title=ylabel, automargin=True, showgrid=True,),
    )
    return layout
