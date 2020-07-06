from typing import Any, Dict


_global_config: Dict = {
<<<<<<< HEAD
    "backends": {"compute": "pandas", "viz": "matplotlib"},
    "display": {"fig_height": 10, "fig_width": 10},
=======
    "backends": {"compute": "pandas"},
>>>>>>> b568645aa1b8c24349f7614c23f0fc32efad3e8c
}


def get_root(path: str):
    pathlist = path.split(".")
    root: Dict = _global_config
    try:
        for p in pathlist[:-1]:
            root = root[p]
        return root, pathlist[-1]
    except KeyError as e:
        raise ValueError("Option does not exist") from e


def get_option(path: str) -> Any:
    root, key = get_root(path)
    return root[key]


def set_option(path: str, value: Any) -> Any:
    root, key = get_root(path)
    if not isinstance(root[key], dict):
        root[key] = value


<<<<<<< HEAD
# Credit: Pandas config
# https://github.com/pandas-dev/pandas/blob/master/pandas/_config/config.py
=======
>>>>>>> b568645aa1b8c24349f7614c23f0fc32efad3e8c
class Options:
    def __init__(self, config: dict, path: str = ""):
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "path", path)

    def __setattr__(self, key: str, value: Any):
        path = object.__getattribute__(self, "path")
        if path:
            path += "."
        path += key

        set_option(path, value)

    def __getattr__(self, key: str) -> Any:
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
        return "Data Describe configuration options"

    def __repr__(self):
        return f"{self.path}\n{self.config}"


options = Options(_global_config)
