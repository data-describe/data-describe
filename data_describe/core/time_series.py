from data_describe.backends import _get_viz_backend, _get_compute_backend
from data_describe.compat import _DATAFRAME_TYPE


def plot_time_series(
    data,
    cols=None,
    decompose=False,
    model="multiplicative",
    compute_backend=None,
    viz_backend=None,
    **kwargs
):
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")
    if decompose:
        result = _get_compute_backend(
            compute_backend, data
        ).compute_decompose_timeseries(
            data, cols=cols, model=model, **kwargs  # need to ensure4 that cols is a str
        )
        fig = _get_viz_backend(viz_backend).viz_plot_time_series(
            data, decompose=decompose, result=result, **kwargs
        )
    else:
        fig = _get_viz_backend(viz_backend).viz_plot_time_series(data, cols, **kwargs)
    return fig


def stationarity_test(data, cols, test="dickey-fuller", compute_backend=None, **kwargs):
    data = _get_compute_backend(compute_backend, data).compute_stationarity_test(
        data[cols], test, **kwargs
    )
    return data


def plot_autocorrelation(
    df,
    col,
    plot_type="acf",
    n_lags=40,
    compute_backend=None,
    viz_backend=None,
    **kwargs
):
    if viz_backend == "plotly":
        data = _get_compute_backend(compute_backend, df).compute_autocorrelation(
            df[col], plot_type=plot_type, n_lags=n_lags, **kwargs
        )
        fig = _get_viz_backend(viz_backend).viz_plot_autocorrelation(
            data, plot_type=plot_type, **kwargs
        )
    else:
        fig = _get_viz_backend(viz_backend).viz_plot_autocorrelation(
            df[col], plot_type=plot_type, n_lags=n_lags, **kwargs
        )
    return fig
