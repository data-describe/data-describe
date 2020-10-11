import sys
import inspect

import pandas

from data_describe.compat._dependency import _compat

_DATAFRAME_BACKENDS = {
    "<class 'pandas.core.frame.DataFrame'>": ["pandas"],
    "<class 'pandas.core.series.Series'>": ["pandas"],
    "<class 'modin.pandas.dataframe.DataFrame'>": ["modin", "pandas"],
    "<class 'modin.pandas.series.Series'>": ["modin", "pandas"],
}


def _is_dataframe(obj, module=None) -> bool:
    """Test if an object is a dataframe type.

    Checks for a DataFrame type from multiple providers (Pandas, Modin etc.).

    Args:
        obj: The object to test.
        module (str, optional): Specify a dataframe type of a particular module. Default
            value of None allows any module to match.

    Returns:
        bool: True if the type matches.
    """
    is_module_dataframe = {"pandas": isinstance(obj, pandas.DataFrame)}

    if "modin" in sys.modules.keys():
        is_module_dataframe["modin"] = isinstance(
            obj, _compat["modin.pandas"].DataFrame
        )
    else:
        # Modin not yet imported; use name checking to avoid expensive import
        # https://stackoverflow.com/questions/49577290/determine-if-object-is-of-type-foo-without-importing-type-foo
        is_module_dataframe["modin"] = False
        for cls in inspect.getmro(type(obj)):
            try:
                if "modin" in cls.__module__ and cls.__name__ == "DataFrame":
                    is_module_dataframe["modin"] = True
            except AttributeError:
                pass

    if module is None:
        return any(is_module_dataframe.values())
    else:
        try:
            return is_module_dataframe.get(module, False)
        except KeyError:
            return False


def _is_series(obj, module=None) -> bool:
    """Test if an object is a series type.

    Checks for a Series type from multiple providers (Pandas, Modin etc.).

    Args:
        obj: The object to test.
        module (str, optional): Specify a series type of a particular module. Default
            value of None allows any module to match.

    Returns:
        bool: True if the type matches.
    """
    is_module_series = {"pandas": isinstance(obj, pandas.Series)}

    if "modin" in sys.modules.keys():
        is_module_series["modin"] = isinstance(obj, _compat["modin.pandas"].Series)
    else:
        # Modin not yet imported; use name checking to avoid expensive import
        # https://stackoverflow.com/questions/49577290/determine-if-object-is-of-type-foo-without-importing-type-foo
        is_module_series["modin"] = False
        for cls in inspect.getmro(type(obj)):
            try:
                if "modin" in cls.__module__ and cls.__name__ == "Series":
                    is_module_series["modin"] = True
            except AttributeError:
                pass

    if module is None:
        return any(is_module_series.values())
    else:
        try:
            return is_module_series.get(module, False)
        except KeyError:
            return False
