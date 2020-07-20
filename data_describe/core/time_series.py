from data_describe.backends import _get_viz_backend, _get_compute_backend
from data_describe.compat import _DATAFRAME_TYPE


def plot_time_series(
    df,
    col,
    decompose=False,
    model="additive",
    compute_backend=None,
    viz_backend=None,
    **kwargs,
):
    """Plots time series given a dataframe with datetime index. Statistics are computed using the statsmodels API.

    Args:
        df: The dataframe with datetime index
        col (str or [str]): Column of interest. Column datatype must be numerical
        decompose: Set as True to decompose the timeseries with moving average. Defaults to False.
        model: Specify seasonal component when decompose is True. Defaults to "additive".
        compute_backend: Select computing backend. Defaults to None (pandas).
        viz_backend: Select visualization backend. Defaults to None (seaborn).

    Returns:
        fig: The visualization
    """
    if not isinstance(df, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")
    if not isinstance(col, (list, str)):
        raise ValueError(f"{col} must be list type or string type")
    if decompose:
        result = _get_compute_backend(compute_backend, df).compute_decompose_timeseries(
            df, col=col, model=model, **kwargs
        )
        fig = _get_viz_backend(viz_backend).viz_plot_time_series(
            df, col=col, result=result, decompose=decompose, **kwargs
        )
    else:
        fig = _get_viz_backend(viz_backend).viz_plot_time_series(df, col, **kwargs)
    return fig


def stationarity_test(
    df, col, test="dickey-fuller", regression="c", compute_backend=None, **kwargs
):
    """Perform stationarity tests to see if mean and variance are changing over time. Backend uses statsmodel's  statsmodels.tsa.stattools.adfuller or statsmodels.tsa.stattools.kpss.

    Args:
        df: The dataframe. Must contain a datetime index
        col: The feature of interest
        test: Choice of stationarity test. "kpss" or "dickey-fuller". Defaults to "dickey-fuller".
        regression: Constant and trend order to include in regression. Choose between 'c','ct','ctt', and 'nc'. Defaults to 'c'
        compute_backend: Select computing backend. Defaults to None (pandas).

    Returns:
        data: Pandas dataframe containing the statistics
    """
    if not isinstance(df, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")
    if not isinstance(col, str):
        raise ValueError(f"{col} not found in dataframe")

    data = _get_compute_backend(compute_backend, df).compute_stationarity_test(
        df[col], test, regression, **kwargs
    )
    return data


def plot_autocorrelation(
    df,
    col,
    plot_type="acf",
    n_lags=40,
    fft=False,
    compute_backend=None,
    viz_backend=None,
    **kwargs,
):
    """Correlation estimate using partial autocorrelation or autocorrelation. Statistics are computed using the statsmodels API.

    Args:
        df: The dataframe with datetime index
        col: The feature of interest
        plot_type: Choose between 'acf' or 'pacf. Defaults to "pacf".
        n_lags: Number of lags to return autocorrelation for. Defaults to 40.
        fft: If True, computes ACF via fourier fast transform (FFT). Defaults to False.
        compute_backend: Select computing backend. Defaults to None (pandas).
        viz_backend: Select visualization backend. Defaults to None (seaborn).

    Returns:
        fig: The visualization
    """
    if not isinstance(df, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")
    if isinstance(col, str):
        if col not in df.columns:
            raise ValueError(f"{col} not found in dataframe")
    if viz_backend == "plotly":
        data = _get_compute_backend(compute_backend, df).compute_autocorrelation(
            df[col], plot_type=plot_type, n_lags=n_lags, fft=fft, **kwargs
        )
        fig = _get_viz_backend(viz_backend).viz_plot_autocorrelation(
            data, plot_type=plot_type, **kwargs
        )
    else:
        fig = _get_viz_backend(viz_backend).viz_plot_autocorrelation(
            df[col], plot_type=plot_type, n_lags=n_lags, fft=fft, **kwargs
        )
    return fig
