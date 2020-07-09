# import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from data_describe.compat import _DATAFRAME_TYPE


def test_stationarity(df, test="dickey-fuller"):
    if test == "dickey-fuller":
        st_df = adf_test(df)
    elif test == "kpss":
        st_df = kpss_test(df)
    else:
        raise ValueError(f"{test} not implemented")
    return st_df


def adf_test(df, autolag="AIC"):
    test = adfuller(df, autolag=autolag)
    adf_output = _DATAFRAME_TYPE.Series(
        test[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in test[4].items():
        adf_output["Critical Value (%s)" % key] = value
    return _DATAFRAME_TYPE.DataFrame(adf_output, columns=["stats"])


def kpss_test(df):
    test = kpss(df, regression="c")
    kpss_output = _DATAFRAME_TYPE.Series(
        test[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in test[3].items():
        kpss_output["Critical Value (%s)" % key] = value

    kpss_df = _DATAFRAME_TYPE.DataFrame(kpss_output, columns=["stats"])
    return kpss_df


def decompose_ts(df, model):
    result = seasonal_decompose(df, model=model)
    return result
