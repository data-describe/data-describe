import contextlib
import copy
from typing import Any, Dict

import pandas as pd


_global_config: Dict = {
    "backends": {"compute": "pandas", "viz": "seaborn"},
    "display": {"fig_height": 10, "fig_width": 10},
}


def get_root(path: str) -> Any:
    """Get the parent dict (node) of a given path from the global config nested dict.

    Args:
        path: The "dot-style" path to the configuration item, e.g. 'backends.viz'

    Returns:
        The parent dict, to be used to access the configuration item or section
    """
    pathlist = path.split(".")
    root: Dict = _global_config
    try:
        for p in pathlist[:-1]:
            root = root[p]
        return root, pathlist[-1]
    except KeyError as e:
        raise ValueError("Option does not exist") from e


def get_option(path: str) -> Any:
    """Get the current value of the option.

    Args:
        path: The "dot-style" path to the configuration item, e.g. 'backends.viz'

    Returns:
        The current value of the option
    """
    root, key = get_root(path)
    return root[key]


def set_option(path: str, value: Any) -> None:
    """Set the current value of the option.

    Args:
        path: The "dot-style" path to the configuration item, e.g. 'backends.viz'
        value: The value to update
    """
    root, key = get_root(path)
    if not isinstance(root[key], dict):
        root[key] = value


def get_config() -> Dict:
    """Get a deep copy of the current configuration.

    Returns:
        The current configuration dictionary
    """
    return copy.deepcopy(_global_config)


def set_config(config: Dict) -> None:
    """Updates the current configuration dictionary.

    Args:
        config: A flattened configuration dictionary specifying values to update
    """
    for k, v in config.items():
        set_option(k, v)


def flatten_config(config: Dict) -> Dict:
    """Flattens the nested configuration dictionary into "dot-style" paths for each item.

    Args:
        config: A flattened configuration dictionary

    Returns:
        A flattened dictionary

        For example:

        {"backends": {"viz": "seaborn"}}

        becomes:
        {"backends.viz": "seaborn"}
    """
    return pd.json_normalize(config, sep=".").iloc[0].to_dict()


# Credit: Pandas config
# https://github.com/pandas-dev/pandas/blob/master/pandas/_config/config.py
class Options:
    """Provides module-style access to configuration items."""

    def __init__(self, config: dict, path: str = ""):
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "path", path)

    def __setattr__(self, key: str, value: Any):
        """Set attribute."""
        path = object.__getattribute__(self, "path")
        if path:
            path += "."
        path += key

        set_option(path, value)

    def __getattr__(self, key: str) -> Any:
        """Get attribute."""
        path = object.__getattribute__(self, "path")
        if path:
            path += "."
        path += key

        try:
            inner = object.__getattribute__(self, "config")[key]
        except KeyError as e:
            raise ValueError("Option does not exist") from e
        if isinstance(inner, dict):
            return Options(inner, path)
        else:
            return get_option(path)

    def __str__(self):
        """Create path string."""
        return f"{self.path}\n{self.config}"

    def __repr__(self):  # noqa:D105
        return self.config


options = Options(_global_config)


@contextlib.contextmanager
def update_context(*args):
    """Data Describe configuration context.

    This can be used to use certain configuration values for a limited block of code,
    without needing to explicitly change these values to what they were previously.

    For example, if the current figure size is (10, 10), the following can be used to
    make one plot with a different figure size:
    ```
    with dd.config.update_context("display.fig_height", 20):
        dd.plot() # fig_height = 20 # noqa:RST301
    ```

    Args:
        *args: May be one of two formats:

            1. A single dictionary, either nested or using the configuration "path"s for keys
            2. Pairs of arguments, where the first argument is the configuration "path" and the
                second is the value
    """
    if len(args) == 1:
        new_config = flatten_config(args[0])
    elif len(args) % 2 == 0:
        new_config = {
            k: v for k, v in [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
        }
    else:
        raise ValueError(
            "Arguments must be either a dictionary or pairs of path, value"
        )

    old_config = flatten_config(get_config())
    set_config(new_config)

    try:
        yield get_config()
    finally:
        set_config(old_config)
