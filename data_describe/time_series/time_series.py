import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
import plotly.graph_objs as go


def convert_todatetime(df, col, unit="ns"):
    df[col] = pd.to_datetime(df.col, unit=unit)
    df.set_index(col, inplace=True)
    return df


def decompose_ts(df, model="multiplicative"):
    result = seasonal_decompose(df, model=model)
    return result


def create_lineplot(result):
    fig = result.plot()
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


def plt_acf(df, n_lags=40):
    fig, ax = plt.subplots(figsize=(11, 9))
    fig = sm.graphics.tsa.plot_acf(df, lags=n_lags, title="ACF Plot", ax=ax)
    # fig returns and image automatically in notebooks
    return fig


def plt_pacf(df, n_lags=40):
    fig, ax = plt.subplots(figsize=(11, 9))
    fig = sm.graphics.tsa.plot_pacf(df, lags=n_lags, title="ACF Plot", ax=ax)
    # fig returns and image automatically in notebooks
    return fig


### Stationary tests
def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    return pd.DataFrame(dfoutput, columns=["stats"])


def kpss_test(timeseries):
    kpsstest = kpss(timeseries, regression="c")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value

    kpss_df = pd.DataFrame(kpss_output, columns=["stats"])
    return kpss_df


def plotly_time_series(df, cols=None):
    data = []
    if isinstance(cols, list):
        for i in cols:
            data.append(go.Scatter(x=df.index, y=df[i]))
    elif isinstance(cols, str):
        data.append(go.Scatter(x=df.index, y=df[cols]))
    return go.Figure(data=data)


def plot_time_series(df, cols=None):
    fig, ax = plt.subplots(figsize=(11, 9))
    if isinstance(cols, list):
        for i in cols:
            fig = sns.lineplot(x=df.index, y=df[i], legend="full", ax=ax)
            plt.legend(cols)
    elif isinstance(cols, str):
        fig = sns.lineplot(x=df.index, y=df[cols], legend="full", ax=ax)
        plt.legend(cols)
    return fig
