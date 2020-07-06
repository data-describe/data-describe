import importlib
from types import ModuleType
from typing import Dict, List

from data_describe.config._config import get_option
from data_describe.compat import _DATAFRAME_BACKENDS

<<<<<<< HEAD

_viz_backends: Dict[str, ModuleType] = {}
=======
>>>>>>> b568645aa1b8c24349f7614c23f0fc32efad3e8c
_compute_backends: Dict[str, ModuleType] = {}


class Backend:
    def __init__(self, b: List[ModuleType]):
        """List of modules to search for implementation"""
        self.b = b

    def __getattr__(self, f):
        for module in self.b:
            try:
                return module.__getattribute__(f)
            except AttributeError:
                pass
        raise ModuleNotFoundError(f"Could not find implementation for {f}")


<<<<<<< HEAD
def _get_viz_backend(backend=None):
    backend = backend or get_option("backends.viz")

    if backend not in _viz_backends:
        module = _find_viz_backend(backend)
        _viz_backends[backend] = module
    return Backend([_viz_backends[backend]])


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


=======
>>>>>>> b568645aa1b8c24349f7614c23f0fc32efad3e8c
def _get_compute_backend(backend=None, df=None):
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


def _find_compute_backend(backend=None):
    """Find a data describe compute backend
<<<<<<< HEAD

    Args:
        backend: The identifier for the backend

=======
    Args:
        backend: The identifier for the backend
>>>>>>> b568645aa1b8c24349f7614c23f0fc32efad3e8c
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
<<<<<<< HEAD

=======
>>>>>>> b568645aa1b8c24349f7614c23f0fc32efad3e8c
            return module
        except ImportError:
            raise ValueError(f"Could not find compute backend '{backend}'")
