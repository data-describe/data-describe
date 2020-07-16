import warnings

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
import pandas as pd


def compute_stationarity_test(
    timeseries, test="dickey-fuller", regression="c", **kwargs
):
    """Perform stationarity tests to see if mean and variance are changing over time.
    Backend uses statsmodel's  statsmodels.tsa.stattools.adfuller or statsmodels.tsa.stattools.kpss

    Args:
        timeseries: Series containing a datetime index
        regression: Constant and trend order to include in regression. Choose between 'c','ct','ctt', and 'nc'
        test: Choice of stationarity test. "kpss" or "dickey-fuller". Defaults to "dickey-fuller".

    Returns:
        st: Pandas dataframe containing the statistics
    """
    if test.lower() == "dickey-fuller":
        st = adf_test(timeseries, regression=regression, **kwargs)
    elif test.lower() == "kpss":
        st = kpss_test(timeseries, regression=regression, **kwargs)
    else:
        raise ValueError(f"{test} not implemented")
    return st


def adf_test(timeseries, autolag="AIC", regression="c"):
    """Compute the Augmented Dickey-Fuller (ADF) test for stationarity
    Backend uses statsmodels.tsa.stattools.adfuller

    Args:
        timeseries: The timeseries
        autolag: Method to use when determining the number of lags. Defaults to 'AIC'. Choose between 'AIC', 'BIC', 't-stat', and None
        regression: Constant and trend order to include in regression. Choose between 'c','ct','ctt', and 'nc'

    Returns:
        Pandas dataframe containing the statistics
    """
    test = adfuller(timeseries, autolag=autolag, regression=regression)
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


def kpss_test(timeseries, regression="c", nlags=None):
    """Compute the Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test for stationarity.
    Backend uses statsmodels.tsa.stattools.kpss

    Args:
        timeseries: The timeseries
        regression: The null hypothesis for the KPSS test.
            'c' : The data is stationary around a constant (default).
            'ct' : The data is stationary around a trend.
        nlags:  Indicates the number of lags to be used. Defaults to None.

    Returns:
        Pandas dataframe containing the statistics
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="The behavior of using lags=None will change in the next release.",
        )
        test = kpss(timeseries, regression="c")
    kpss_output = pd.Series(test[0:3], index=["Test Statistic", "p-value", "Lags Used"])
    for key, value in test[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    return pd.DataFrame(kpss_output, columns=["stats"])


def compute_decompose_timeseries(df, col, model="additive"):
    """Seasonal decomposition using moving averages

    Args:
        df: The dataframe
        col: The col of interest. Must be numeric datatype
        model: Type of seasonal component. Defaults to "additive".

    Returns:
        result: statsmodels.tsa.seasonal.DecomposeResult object
    """
    return seasonal_decompose(df[col], model=model)


# check kwargs are passed
def compute_autocorrelation(timeseries, plot_type="acf", n_lags=40, fft=False):
    """Correlation estimate using partial autocorrelation or autocorrelation

    Args:
        timeseries: Series object containing datetime index
        plot_type: Choose between 'acf' or 'pacf. Defaults to "acf".
        n_lags: Number of lags to return autocorrelation for. Defaults to 40.
        fft: If True, computes ACF via fourier fast transform (FFT). Defaults to False.

    Returns:
        data: numpy.ndarray containing the correlations
    """
    if plot_type == "pacf":
        data = pacf(timeseries, n_lags)
    elif plot_type == "acf":
        data = acf(timeseries, n_lags, fft=fft)
    else:
        raise ValueError("Unsupported input data type")
    return data
