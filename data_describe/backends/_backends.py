import importlib
from types import ModuleType
from typing import Dict, List

from data_describe.config._config import get_option
from data_describe.compat import _DATAFRAME_BACKENDS

_viz_backends: Dict[str, ModuleType] = {}
_compute_backends: Dict[str, ModuleType] = {}


class Backend:
    """Interface for compute and visualization backends."""

    def __init__(self, b: List[ModuleType]):
        """Initialize with list of modules to search for implementation."""
        self.b = b

    def __getattr__(self, f: str):
        """Try to find the method implementation in the module list."""
        for module in self.b:
            try:
                return module.__getattribute__(f)
            except AttributeError:
                pass
        raise ModuleNotFoundError(f"Could not find implementation for {f}")


def _get_viz_backend(backend: str = None):
    """Get the visualization backend by name.

    Args:
        backend: The name of the backend, usually the package name

    Returns:
        Backend
    """
    backend = backend or get_option("backends.viz")

    if backend not in _viz_backends:
        module = _find_viz_backend(backend)
        _viz_backends[backend] = module
    return Backend([_viz_backends[backend]])


def _find_viz_backend(backend: str):
    """Find a data describe visualization backend.

    Args:
        backend: The name of the backend, usually the package name

    Returns:
        The imported backend module
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
        except ModuleNotFoundError:
            raise ValueError(f"Could not find visualization backend '{backend}'")


def _get_compute_backend(backend: str = None, df=None):
    """Get the compute backend by name.

    Args:
        backend: The name of the backend, usually the package name
        df: The input dataframe which may be used to infer the backend

    Returns:
        Backend
    """
    data_type = str(type(df))
    backend_sources = [
        backend,
        _DATAFRAME_BACKENDS.get(data_type, None),
        get_option("backends.compute"),
    ]

    backend_list = []
    for backend in backend_sources:
        if backend:
            if backend not in _compute_backends:
                module = _find_compute_backend(backend)
                _compute_backends[backend] = module
            else:
                module = _compute_backends[backend]
            backend_list.append(module)
    return Backend(backend_list)


def _find_compute_backend(backend):
    """Find a data describe compute backend.

    Args:
        backend: The name of the backend, usually the package name

    Returns:
        The imported backend module
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
        except ModuleNotFoundError:
            raise ValueError(f"Could not find compute backend '{backend}'")
