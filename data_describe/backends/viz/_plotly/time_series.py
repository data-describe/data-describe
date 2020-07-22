import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
from IPython import get_ipython

from data_describe.config._config import get_option


def viz_plot_time_series(df, col, result=None, decompose=False, title="Time Series"):
    """Create timeseries visualization.

    Args:
        df: The dataframe
        col (str or [str]): Column of interest. Column datatype must be numerical.
        result: The statsmodels.tsa.seasonal.DecomposeResult object. Defaults to None.
        decompose: Set as True to decompose the timeseries with moving average. result must not be None. Defaults to False.
        title: Title of the plot. Defaults to "Time Series".

    Returns:
        fig: The visualization
    """
    if isinstance(col, list):
        data = [go.Scatter(x=df.index, y=df[c], name=c) for c in col]
        ylabel = "Variable" if len(col) > 1 else col[0]
        fig = go.Figure(data=data, layout=figure_layout(title=title, ylabel=ylabel))
    elif isinstance(col, str) and not decompose:
        fig = go.Figure(
            data=go.Scatter(x=df.index, y=df[col], name=col),
            layout=figure_layout(title=title, ylabel=col),
        )
    elif decompose:
        fig = viz_decomposition(result, dates=df.index)
    if get_ipython() is not None:
        init_notebook_mode(connected=True)
        return iplot(fig, config={"displayModeBar": False})
    else:
        return fig


def viz_decomposition(result, dates, title="Time Series Decomposition"):
    """Create timeseries decomposition visualization.

    Args:
        result: The statsmodels.tsa.seasonal.DecomposeResult object. Defaults to None.
        dates: The datetime index
        title: Title of the plot. Defaults to "Time Series Decomposition".

    Returns:
        fig: The visualization
    """
    fig = make_subplots(rows=4, cols=1, x_title="Time", shared_xaxes=True)
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


def viz_plot_autocorrelation(
    data, white_noise, n_lags, plot_type="acf", title="Autocorrelation Plot"
):
    """Create timeseries autocorrelation visualization.

    Args:
        data: numpy.ndarray containing the correlations
        white_noise: Significance threshold
        plot_type: Choose between 'acf' or 'pacf. Defaults to "pacf".
        title: Title of the plot. Defaults to "Autocorrelation Plot".

    Returns:
        fig: The visualization
    """
    if plot_type == "acf":
        data = [go.Bar(y=data, showlegend=False, name=plot_type)]
    elif plot_type == "pacf":
        data = [go.Bar(y=data, showlegend=False, name=plot_type)]
    else:
        raise ValueError("Unsupported input data type")

    fig = go.Figure(data=data, layout=figure_layout(title, "Lags", plot_type))
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(n_lags + 1)],
            y=[white_noise for i in range(n_lags + 1)],
            mode="lines",
            name="95% Confidence",
            line={"dash": "dash"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[i for i in range(n_lags + 1)],
            y=[-white_noise for i in range(n_lags + 1)],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            showlegend=False,
        )
    )
    return fig


def figure_layout(title="Time Series", xlabel="Date", ylabel="Variable"):
    """Generates the figure layout.

    Args:
        title: Title of the plot. Defaults to "Time Series".
        xlabel: x-axis label. Defaults to "Date".
        ylabel; y-axis label. Defaults to "Variable".

    Returns:
        layour: The plotly layout
    """
    layout = go.Layout(
        title={"text": title, "font": {"size": 25}},
        width=get_option("display.fig_width") * 100,
        height=get_option("display.fig_height") * 100,
        xaxis=go.layout.XAxis(ticks="", title=xlabel, showgrid=True),
        yaxis=go.layout.YAxis(ticks="", title=ylabel, automargin=True, showgrid=True),
    )
    return layout
