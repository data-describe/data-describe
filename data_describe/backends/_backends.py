import importlib
from types import ModuleType
from typing import Dict, List, Optional
import logging

from data_describe.config._config import get_option
from data_describe.compat import _DATAFRAME_BACKENDS, _DATAFRAME_STATIC_TYPE

_viz_backends: Dict[str, Dict[str, ModuleType]] = {}
_compute_backends: Dict[str, Dict[str, ModuleType]] = {}


class Backend:
    """Interface for compute and visualization backends"""

    def __init__(self, b: List[ModuleType]):
        """Initialize with list of modules to search for implementation"""
        self.b = b

    def __getattr__(self, f: str):
        """Try to find the method implementation in the module list"""
        for module in self.b:
            try:
                return module.__getattribute__(f)
            except AttributeError:
                pass
        raise ModuleNotFoundError(
            f"Could not find implementation for {f} with available backends: {self.b}"
        )


def _get_viz_backend(backend: str = None):
    """Get the visualization backend by name

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
    """Find a data describe visualization backend

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


def _get_compute_backend(backend: str = None, df: _DATAFRAME_STATIC_TYPE = None):
    """Get the compute backend by name

    Args:
        backend: The name of the backend, usually the package name
        df: The input dataframe which may be used to infer the backend

    Returns:
        Backend
    """
    data_type = str(type(df))
    backend_types = set(
        [
            backend,
            _DATAFRAME_BACKENDS.get(data_type, None),
            get_option("backends.compute"),
        ]
    )
    logging.info(f"Backend sources are {backend_types}")

    backend_list = []
    for backend in backend_types:
        if backend:
            if _check_backend(backend, _compute_backends):
                modules = _compute_backends[backend]
            else:
                modules = _load_compute_backend(backend)
            backend_list.append(modules)
    backend_list = [module for d in backend_list for _, module in d.items()]
    return Backend(backend_list)


def _load_compute_backend(backend) -> Dict[str, ModuleType]:
    """Load implementations for a data describe compute backend

    Args:
        backend: The name of the backend, usually the package name

    Returns:
        The dictionary of loaded backend module(s)
    """
    import pkg_resources  # noqa: delay import for performance

    for entry_point in pkg_resources.iter_entry_points(
        "data_describe_compute_backends"
    ):
        logging.info(
            f"Loading entry point {entry_point.name} from {entry_point.load()}"
        )
        _add_backend(entry_point.name, _compute_backends, entry_point.load())

    logging.info(f"Loaded compute backends are {_compute_backends}")
    try:
        return _compute_backends[backend]
    except KeyError:
        try:
            module = importlib.import_module(backend)
            _add_backend(backend, _compute_backends, module)

            return _compute_backends[backend]
        except ModuleNotFoundError:
            raise ValueError(f"Could not find compute backend '{backend}'")


def _check_backend(
    backend_type: str, loaded_backends: dict, module: Optional[ModuleType] = None
) -> bool:
    """Checks if the backend has already been loaded

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
            if str(module.__path__) in loaded_backends[backend_type]:
                return True
    return False


def _add_backend(backend_type: str, loaded_backends: dict, module: ModuleType):
    """Adds the backend module to the global backends dictionary

    Uses the MD5 hash of the module as the key
    Args:
        backend_type: The name of the backend
        loaded_backends: The global backends dictionary
        module: The module that implements the backend
    """
    if backend_type not in loaded_backends:
        loaded_backends[backend_type] = {}

    loaded_backends[backend_type][str(module.__path__)] = module
