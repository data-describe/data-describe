from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd

from data_describe.compat import _DATAFRAME_TYPE


def compute_stationarity_test(df, test="dickey-fuller"):
    if test == "dickey-fuller":
        st_df = adf_test(df)
    elif test == "kpss":
        st_df = kpss_test(df)
    else:
        raise ValueError(f"{test} not implemented")
    return st_df


def adf_test(df, autolag="AIC"):
    test = adfuller(df, autolag=autolag)
    adf_output = pd.Series(
        test[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in test[4].items():
        adf_output["Critical Value (%s)" % key] = value
    return pd.DataFrame(adf_output, columns=["stats"])


def kpss_test(df):
    test = kpss(df, regression="c")
    kpss_output = pd.Series(test[0:3], index=["Test Statistic", "p-value", "Lags Used"])
    for key, value in test[3].items():
        kpss_output["Critical Value (%s)" % key] = value

    kpss_df = pd.DataFrame(kpss_output, columns=["stats"])
    return kpss_df


def compute_decompose_timeseries(df, cols, model="multiplicative"):
    if isinstance(df, _DATAFRAME_TYPE):
        result = seasonal_decompose(df[cols], model=model)
    else:
        raise ValueError("Unsupported input data type")
    return result


def compute_autocorrelation(df, n_lags=40, plot_type="pacf"):
    if plot_type == "pacf":
        data = pacf(df, n_lags)
    elif plot_type == "acf":
        data = acf(df, n_lags)
    else:
        raise ValueError("Unsupported input data type")
    return data
