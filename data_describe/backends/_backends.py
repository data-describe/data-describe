import importlib
from types import ModuleType
from typing import Dict, List, Optional

from data_describe.config._config import get_option
from data_describe.compat import _DATAFRAME_BACKENDS

_viz_backends: Dict[str, Dict[str, ModuleType]] = {}
_compute_backends: Dict[str, Dict[str, ModuleType]] = {}


class _Backend:
    """Interface for compute and visualization backends.

    Attributes:
        backends: A list of Python modules that may implement one
            or more compute or visualization backends. To be used by
            data-describe, the module must expose functions with the
            naming pattern `compute_FEATURE` or `viz_FEATURE` and register
            as a data-describe entrypoint. (See setup.py)dag
    """

    def __init__(self, backends: List[ModuleType]):
        """Initialize with list of modules to search for implementation."""
        self.backends = backends

    def __getattr__(self, f: str):
        """Try to find the method implementation in the module list."""
        for module in self.b:
            try:
                return module.__getattribute__(f)
            except AttributeError:
                pass
        raise ModuleNotFoundError(
            f"Could not find implementation for {f} with available backends: {self.b}"
        )


def _get_viz_backend(backend: str = None) -> _Backend:
    """Get the visualization backend by name.

    Args:
        backend: The name of the backend, usually the package name

    Returns:
        _Backend
    """
    if backend:
        backend_types = [backend]
    else:
        backend_types = [get_option("backends.viz")]

    backend_collection = []
    for backend in backend_types:
        if _check_backend(backend, _viz_backends):
            modules = _viz_backends[backend]
        else:
            modules = _load_viz_backend(backend)
        backend_collection.append(modules)
    backend_list = [module for d in backend_collection for _, module in d.items()]
    return _Backend(backend_list)


def _load_viz_backend(backend: str) -> Dict[str, ModuleType]:
    """Find a data describe visualization backend.

    Args:
        backend: The name of the backend, usually the package name

    Returns:
        The imported backend module
    """
    from importlib_metadata import entry_points  # noqa: delay import for performance

    for entry_point in entry_points()["data_describe_viz_backends"]:
        _add_viz_backend(entry_point.name, entry_point.load())

    try:
        return _viz_backends[backend]
    except KeyError:
        try:
            module = importlib.import_module(backend)
            _add_viz_backend(backend, module)

            return _viz_backends[backend]
        except ImportError:
            raise ValueError(f"Could not find visualization backend '{backend}'")


def _get_compute_backend(backend: str = None, df=None) -> _Backend:
    """Get the compute backend by name.

    In addition to searching through entrypoints, the input data (DataFrame)
    type will be used to infer an appropriate compute backend.

    Args:
        backend: The name of the backend, usually the package name
        df: The input dataframe which may be used to infer the backend

    Returns:
        _Backend
    """
    if backend:
        backend_types = [backend]
    else:
        data_type = str(type(df))
        backend_types = [
            *_DATAFRAME_BACKENDS.get(data_type, ["None"]),
            get_option("backends.compute"),
        ]

        # Remove duplicates, maintain order
        seen = set()
        for idx, backend_name in enumerate(backend_types):
            if backend_name in seen:
                backend_types.pop(idx)
            elif backend_name != "None":
                seen.add(backend_name)

    backend_collection = []
    for backend in backend_types:
        if _check_backend(backend, _compute_backends):
            modules = _compute_backends[backend]
        else:
            modules = _load_compute_backend(backend)
        backend_collection.append(modules)
    backend_list = [module for d in backend_collection for _, module in d.items()]
    return _Backend(backend_list)


def _load_compute_backend(backend) -> Dict[str, ModuleType]:
    """Load implementations for a data describe compute backend.

    Args:
        backend: The name of the backend, usually the package name

    Returns:
        The dictionary of loaded backend module(s)
    """
    from importlib_metadata import entry_points  # noqa: delay import for performance

    for entry_point in entry_points()["data_describe_compute_backends"]:
        _add_compute_backend(entry_point.name, entry_point.load())

    try:
        return _compute_backends[backend]
    except KeyError:
        try:
            module = importlib.import_module(backend)
            _add_compute_backend(backend, module)

            return _compute_backends[backend]
        except ImportError:
            raise ValueError(f"Could not find compute backend '{backend}'")


def _check_backend(
    backend_type: str, loaded_backends: dict, module: Optional[ModuleType] = None
) -> bool:
    """Checks if the backend has already been loaded.

    Args:
        backend_type: The name of the backend
        loaded_backends: The global backends dictionary
        module: The module that implements the backend

    Returns:
        True if the backend/module exists in the loaded backends dictionary
    """
    if backend_type in loaded_backends:
        if module is None:
            return True
        else:
            if str(module.__path__) in loaded_backends[backend_type]:  # type: ignore
                return True
    return False


def _add_compute_backend(backend_type: str, module: ModuleType):
    """Adds the backend module to the global backends dictionary.

    Args:
        backend_type: The name of the backend
        module: The module that implements the backend
    """
    if backend_type not in _compute_backends:
        _compute_backends[backend_type] = {}

    _compute_backends[backend_type][str(module.__path__)] = module  # type: ignore


def _add_viz_backend(backend_type: str, module: ModuleType):
    """Adds the backend module to the global backends dictionary.

    Args:
        backend_type: The name of the backend
        module: The module that implements the backend
    """
    if backend_type not in _viz_backends:
        _viz_backends[backend_type] = {}

    _viz_backends[backend_type][str(module.__path__)] = module  # type: ignore
