import importlib
from types import ModuleType
from typing import Dict, List, Optional

from data_describe.config._config import get_option
from data_describe._compat import _DATAFRAME_BACKENDS

_viz_backends: Dict[str, Dict[str, ModuleType]] = {}
_compute_backends: Dict[str, Dict[str, ModuleType]] = {}


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
        raise ModuleNotFoundError(
            f"Could not find implementation for {f} with available backends: {self.b}"
        )


def _get_viz_backend(backend: str = None):
    """Get the visualization backend by name.

    Args:
        backend: The name of the backend, usually the package name

    Returns:
        Backend
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
    return Backend(backend_list)


def _load_viz_backend(backend: str) -> Dict[str, ModuleType]:
    """Find a data describe visualization backend.

    Args:
        backend: The name of the backend, usually the package name

    Returns:
        The imported backend module
    """
    import pkg_resources  # noqa: delay import for performance

    for entry_point in pkg_resources.iter_entry_points("data_describe_viz_backends"):
        _add_backend(entry_point.name, _viz_backends, entry_point.load())

    try:
        return _viz_backends[backend]
    except KeyError:
        try:
            module = importlib.import_module(backend)
            _add_backend(backend, _viz_backends, module)

            return _viz_backends[backend]
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
    return Backend(backend_list)


def _load_compute_backend(backend) -> Dict[str, ModuleType]:
    """Load implementations for a data describe compute backend.

    Args:
        backend: The name of the backend, usually the package name

    Returns:
        The dictionary of loaded backend module(s)
    """
    import pkg_resources  # noqa: delay import for performance

    for entry_point in pkg_resources.iter_entry_points(
        "data_describe_compute_backends"
    ):
        _add_backend(entry_point.name, _compute_backends, entry_point.load())

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
            if str(module.__path__) in loaded_backends[backend_type]:  # type: ignore # mypy issue 1422
                return True
    return False


def _add_backend(backend_type: str, loaded_backends: dict, module: ModuleType):
    """Adds the backend module to the global backends dictionary.

    Args:
        backend_type: The name of the backend
        loaded_backends: The global backends dictionary
        module: The module that implements the backend
    """
    if backend_type not in loaded_backends:
        loaded_backends[backend_type] = {}

    loaded_backends[backend_type][str(module.__path__)] = module  # type: ignore # mypy issue 1422
