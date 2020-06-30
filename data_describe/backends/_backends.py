import importlib
from types import ModuleType
from typing import Dict

from data_describe.config import df_backend

_df_backends: Dict[str, ModuleType] = {}


def _get_df_backend(backend=None):
    backend = backend or df_backend

    if backend == "pandas":
        try:
            import data_describe.backends.compute._pandas
        except ImportError:
            raise ImportError(
                "pandas is required for computation when the "
                "default backend 'pandas' is selected."
            ) from None

    if backend in _df_backends:
        return _df_backends[backend]

    module = _find_df_backend(backend)
    _df_backends[backend] = module
    return module


def _find_df_backend(backend=None):
    import pkg_resources

    import pkg_resources

    for entry_point in pkg_resources.iter_entry_points("data_describe_df_backends"):
        _df_backends[entry_point.name] = entry_point.load()

    try:
        return _df_backends[backend]
    except KeyError:
        try:
            module = importlib.import_module(backend)
            _df_backends[backend] = module

            return module
        except ImportError:
            raise ValueError(f"Could not find DataFrame backend '{backend}'")
