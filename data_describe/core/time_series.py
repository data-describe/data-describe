import statsmodels

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
    if decompose:
        data = _get_compute_backend(compute_backend, data).compute_decompose_timeseries(
            data, model, **kwargs
        )

    if isinstance(data, statsmodels.tsa.seasonal.DecomposeResult):
        fig = _get_viz_backend(viz_backend).viz_plot_time_series(
            decompose=decompose, result=data, **kwargs
        )

    if isinstance(data, _DATAFRAME_TYPE):
        fig = _get_viz_backend(viz_backend).viz_plot_time_series(data, cols, **kwargs)
    return fig


def stationarity_test(data, test="dickey-fuller", compute_backend=None, **kwargs):
    data = _get_compute_backend(compute_backend, data).compute_stationarity_test(
        data, test, **kwargs
    )
    return data


def plot_autocorrelation(
    df, plot_type="acf", n_lags=40, compute_backend=None, viz_backend=None, **kwargs
):
    if viz_backend == "plotly":
        data = _get_compute_backend(compute_backend, df).compute_autocorrelation(
            df, plot_type=plot_type, n_lags=n_lags, **kwargs
        )
        fig = _get_viz_backend(viz_backend).viz_plot_autocorrelation(
            data, plot_type=plot_type, **kwargs
        )
    else:
        fig = _get_viz_backend(viz_backend).viz_plot_autocorrelation(
            df, plot_type=plot_type, n_lags=n_lags, **kwargs
        )
    return fig
