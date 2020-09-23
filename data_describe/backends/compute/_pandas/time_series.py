import warnings
from typing import Optional

import pandas as pd
import numpy as np

from data_describe.compat import requires, _compat


@requires("statsmodels")
def compute_stationarity_test(
    timeseries, test: str = "dickey-fuller", regression: str = "c", **kwargs
):
    """Perform stationarity tests to see if mean and variance are changing over time.

    Backend uses statsmodel's  statsmodels.tsa.stattools.adfuller or statsmodels.tsa.stattools.kpss

    Args:
        timeseries: Series containing a datetime index
        test: Choice of stationarity test. "kpss" or "dickey-fuller". Defaults to "dickey-fuller".
        regression: Constant and trend order to include in regression. Choose between 'c','ct','ctt', and 'nc'
        **kwargs: Keyword arguments for adf and kpss

    Returns:
        st: Pandas dataframe containing the statistics
    """
    if test.lower() == "dickey-fuller":
        st = _compat["statsmodels"].tsa.stattools.adf_test(
            timeseries, regression=regression, **kwargs
        )
    elif test.lower() == "kpss":
        st = _compat["statsmodels"].tsa.stattools.kpss_test(
            timeseries, regression=regression, **kwargs
        )
    else:
        raise ValueError(f"{test} not implemented")
    return st


@requires("statsmodels")
def adf_test(timeseries, autolag: str = "AIC", regression: str = "c", **kwargs):
    """Compute the Augmented Dickey-Fuller (ADF) test for stationarity.

    Backend uses statsmodels.tsa.stattools.adfuller

    Args:
        timeseries: The timeseries
        autolag: Method to use when determining the number of lags. Defaults to 'AIC'. Choose between 'AIC', 'BIC', 't-stat', and None
        regression: Constant and trend order to include in regression. Choose between 'c','ct','ctt', and 'nc'
        **kwargs: Keyword arguments for adfuller

    Returns:
        Pandas dataframe containing the statistics
    """
    test = _compat["statsmodels"].tsa.stattools.adfuller(
        timeseries, autolag=autolag, regression=regression, **kwargs
    )
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


@requires("statsmodels")
def kpss_test(timeseries, regression: str = "c", nlags: Optional[int] = None, **kwargs):
    """Compute the Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test for stationarity.

    Backend uses statsmodels.tsa.stattools.kpss

    Args:
        timeseries: The timeseries

        regression: The null hypothesis for the KPSS test.
            'c' : The data is stationary around a constant (default).
            'ct' : The data is stationary around a trend.

        nlags:  Indicates the number of lags to be used. Defaults to None.
        **kwargs: Keyword arguments for kpss

    Returns:
        Pandas dataframe containing the statistics
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="The behavior of using lags=None will change in the next release.",
        )
        test = _compat["statsmodels"].tsa.stattools.kpss(
            timeseries, regression="c", **kwargs
        )
    kpss_output = pd.Series(test[0:3], index=["Test Statistic", "p-value", "Lags Used"])
    for key, value in test[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    return pd.DataFrame(kpss_output, columns=["stats"])


@requires("statsmodels")
def compute_decompose_timeseries(df, col, model: str = "additive", **kwargs):
    """Seasonal decomposition using moving averages.

    NOTE: decomposition object in modin does not preserve datetime index.

    Args:
        df: The dataframe
        col: The col of interest. Must be numeric datatype
        model: Type of seasonal component. Defaults to "additive".
        **kwargs: Keyword arguments

    Returns:
        result: statsmodels.tsa.seasonal.DecomposeResult object
    """
    return _compat["statsmodels"].tsa.seasonal.seasonal_decompose(
        df[col], model=model, **kwargs
    )


@requires("statsmodels")
def compute_autocorrelation(
    timeseries,
    n_lags: Optional[int] = 40,
    plot_type: str = "acf",
    fft: bool = False,
    **kwargs,
):
    """Correlation estimate using partial autocorrelation or autocorrelation.

    Args:
        timeseries: Series object containing datetime index
        n_lags: Number of lags to return autocorrelation for. Defaults to 40.
        plot_type: Choose between 'acf' or 'pacf. Defaults to "acf".
        fft: If True, computes ACF via fourier fast transform (FFT). Defaults to False.
        **kwargs: Keyword arguments

    Returns:
        data: numpy.ndarray containing the correlations
    """
    if plot_type == "pacf":
        data = _compat["statsmodels"].tsa.stattools.pacf(timeseries, n_lags, **kwargs)
    elif plot_type == "acf":
        data = _compat["statsmodels"].tsa.stattools.acf(
            timeseries, n_lags, fft=fft, **kwargs
        )
    else:
        raise ValueError("Unsupported input data type")
    white_noise = 1.96 / np.sqrt(len(data))
    return data, white_noise
