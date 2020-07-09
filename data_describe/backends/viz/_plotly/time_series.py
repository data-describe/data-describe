import plotly.graph_objs as go
from statsmodels.tsa.stattools import acf, pacf


# def viz_time_series(data):


def plotly_time_series(df, cols=None):
    data = []
    if isinstance(cols, list):
        for i in cols:
            data.append(go.Scatter(x=df.index, y=df[i]))
    elif isinstance(cols, str):
        data.append(go.Scatter(x=df.index, y=df[cols]))
    return go.Figure(data=data)


def plotly_pacf(df, n_lags):
    data = [go.Bar(y=pacf(df, n_lags))]
    fig = go.figure(data=data)
    # use iplot to visualize
    return fig


def plotly_acf(df, n_lags=40):
    data = [go.Bar(y=acf(df, n_lags))]
    fig = go.figure(data=data)
    # use iplot to visualize
    return fig


def plotly_decomposition(result, plot_type="trend"):
    if plot_type == "trend":
        data = [go.Scatter(x=result.trend.index, y=result.trend)]
    elif plot_type == "residual":
        data = [go.Scatter(x=result.resid.index, y=result.resid)]
    elif plot_type == "seasonal":
        data = [go.Scatter(x=result.seasonal.index, y=result.seasonal)]
    else:
        raise ValueError(
            "Please choose between trend, residual, and seasonal for plot_types"
        )
    fig = go.figure(data=data)
    return fig
