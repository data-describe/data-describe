import statsmodels

from data_describe.backends import _get_viz_backend, _get_compute_backend
from data_describe.compat import _DATAFRAME_TYPE


def plot_time_series(
    data, cols=None, decompose=False, compute_backend=None, viz_backend=None, **kwargs
):
    if decompose:
        data = _get_compute_backend(compute_backend, data).compute_decompose_timeseries(
            data, **kwargs
        )

    if isinstance(data, statsmodels.tsa.seasonal.DecomposeResult):
        fig = _get_viz_backend(viz_backend).viz_plot_time_series(
            decompose=decompose, result=data, **kwargs
        )

    if isinstance(data, _DATAFRAME_TYPE):
        fig = _get_viz_backend(viz_backend).viz_plot_time_series(data, cols, **kwargs)
    return fig


def stationarity_test(data, test=None, compute_backend=None, **kwargs):
    data = _get_compute_backend(compute_backend, data).compute_stationarity_test(
        data, test, **kwargs
    )
    return data
