from data_describe.backends import _get_viz_backend, _get_compute_backend


def plot_time_series(data, y, compute_backend=None, viz_backend=None, **kwargs):
    data = _get_compute_backend(compute_backend, data).process_time_series(
        data, y, **kwargs
    )
    return _get_viz_backend(viz_backend).viz_data_heatmap(data)


def test_stationarity(data, test, compute_backend=None, **kwargs):
    data = _get_compute_backend(compute_backend, data).process_time_series(
        data, test, **kwargs
    )
    return data
