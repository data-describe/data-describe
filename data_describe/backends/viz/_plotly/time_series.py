import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
from IPython import get_ipython

from data_describe.config._config import get_option


def viz_plot_time_series(
    df=None, cols=None, result=None, decompose=False, title="Time Series"
):
    if isinstance(cols, list):
        data = [go.Scatter(x=df.index, y=df[c], name=c) for c in cols]
        fig = go.Figure(data=data, layout=figure_layout(title=title))
    elif isinstance(cols, str):
        fig = go.Figure(
            data=go.Scatter(x=df.index, y=df[cols], name=cols),
            layout=figure_layout(title=title, ylabel=cols),
        )
    elif decompose:
        fig = viz_decomposition(result, dates=df.index)
    if get_ipython() is not None:
        init_notebook_mode(connected=True)
        return iplot(fig, config={"displayModeBar": False})
    else:
        return fig


def viz_decomposition(result, dates, title="Time Series Decomposition"):
    fig = make_subplots(rows=4, cols=1, x_title="Time", shared_xaxes=True)
    print(result)
    fig.add_trace(
        go.Scatter(x=dates, y=result.observed, name="observed",), row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=dates, y=result.trend, name="trend"), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=dates, y=result.seasonal, name="seasonal"), row=3, col=1,
    )
    fig.add_trace(go.Scatter(x=dates, y=result.resid, name="residual"), row=4, col=1)
    fig.update_layout(
        height=get_option("display.fig_height") * 100,
        width=get_option("display.fig_width") * 100,
        title_text=title,
        legend_title_text="Decomposition",
    )
    return fig


def viz_plot_autocorrelation(data, plot_type="acf", title="Autocorrelation Plot"):
    if plot_type == "acf":
        data = [go.Bar(y=data)]
    elif plot_type == "pacf":
        data = [go.Bar(y=data)]
    else:
        raise ValueError("Unsupported input data type")
    fig = go.Figure(data=data, layout=figure_layout(title))
    return fig


def figure_layout(title="Time Series", xlabel="Date", ylabel="Variable"):
    layout = go.Layout(
        title={"text": title, "font": {"size": 25}},
        width=get_option("display.fig_width") * 100,
        height=get_option("display.fig_height") * 100,
        xaxis=go.layout.XAxis(ticks="", title=xlabel, showgrid=True),
        yaxis=go.layout.YAxis(ticks="", title=ylabel, automargin=True, showgrid=True),
    )
    return layout
