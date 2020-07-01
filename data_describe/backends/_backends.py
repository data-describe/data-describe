import importlib
from types import ModuleType
from typing import Dict

from data_describe.config._config import get_option
from data_describe.compat import _DATAFRAME_BACKENDS


_viz_backends: Dict[str, ModuleType] = {}
_compute_backends: Dict[str, ModuleType] = {}


def _get_viz_backend(backend=None):
    backend = backend or get_option("backends.viz")

    if backend == "matplotlib":
        try:
            import data_describe.backends.viz._matplotlib  # noqa
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting when the "
                "default backend 'matplotlib' is selected."
            ) from None

    # Backend already loaded
    if backend in _viz_backends:
        return _viz_backends[backend]

    module = _find_viz_backend(backend)
    _viz_backends[backend] = module
    return module


def _find_viz_backend(backend=None):
    """Find a data describe visualization backend

    Args:
        backend: The identifier for the backend

    Returns:
        The imported backend
    """
    import pkg_resources  # noqa: delay import for performance

    for entry_point in pkg_resources.iter_entry_points("data_describe_viz_backends"):
        _viz_backends[entry_point.name] = entry_point.load()

    try:
        return _viz_backends[backend]
    except KeyError:
        try:
            module = importlib.import_module(backend)
            _viz_backends[backend] = module

            return module
        except ImportError:
            raise ValueError(f"Could not find visualization backend '{backend}'")


def _get_compute_backend(backend=None, df=None):
    backend = (
        backend
        or get_option("backends.compute")
        or _DATAFRAME_BACKENDS.get(str(type(df)), None)
    )

    if backend == "pandas":
        try:
            import data_describe.backends.compute._pandas  # noqa
        except ImportError:
            raise ImportError(
                "pandas is required when the default backend 'pandas' is selected."
            ) from None

    # Backend already loaded
    if backend in _compute_backends:
        return _compute_backends[backend]

    module = _find_compute_backend(backend)
    _compute_backends[backend] = module
    return module


def _find_compute_backend(backend=None):
    """Find a data describe compute backend

    Args:
        backend: The identifier for the backend

    Returns:
        The imported backend
    """
    import pkg_resources  # noqa: delay import for performance

    for entry_point in pkg_resources.iter_entry_points(
        "data_describe_compute_backends"
    ):
        _compute_backends[entry_point.name] = entry_point.load()

    try:
        return _compute_backends[backend]
    except KeyError:
        try:
            module = importlib.import_module(backend)
            _compute_backends[backend] = module

            return module
        except ImportError:
            raise ValueError(f"Could not find compute backend '{backend}'")
