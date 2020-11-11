import pytest

from data_describe.backends import _get_compute_backend, _get_viz_backend


def test_missing_compute_implementation():
    with pytest.raises(NotImplementedError):
        _get_compute_backend().compute_nothing()


def test_missing_viz_implementation():
    with pytest.raises(NotImplementedError):
        _get_viz_backend().viz_nothing()


@pytest.mark.parametrize(
    "api_method",
    [
        "data_summary",
        "data_heatmap",
        "cluster",
        "correlation_matrix",
        "distribution",
        "scatter_plot",
    ],
)
@pytest.mark.parametrize("backend_module", ["pandas", "modin", "seaborn", "plotly"])
@pytest.mark.parametrize("backend", ["compute", "viz"])
def test_api_methods(backend, backend_module, api_method):
    if backend == "compute":
        if backend_module in ["pandas", "modin"]:
            if (
                api_method not in ["distribution", "scatter_plot", "correlation_matrix"]
            ) and not (
                backend_module == "modin"
                and (api_method in ["data_heatmap", "cluster"])
            ):
                _get_compute_backend(backend_module).__getattr__(
                    "_".join([backend, api_method])
                )
    elif backend == "viz":
        if backend_module in ["seaborn", "plotly"]:
            if (
                api_method
                not in [
                    "data_summary",
                    "viz_distribution",
                    "viz_scatter_plot",
                ]
            ) and not (
                backend_module == "plotly"
                and (api_method in ["distribution", "scatter_plot"])
            ):
                _get_viz_backend(backend_module).__getattr__(
                    "_".join([backend, api_method])
                )
    else:
        pytest.skip(f"Skipped {backend}({backend_module})_{api_method}")
