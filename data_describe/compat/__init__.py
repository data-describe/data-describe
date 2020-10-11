"""data-describe compatibility and checks.

data-describe makes use of a wide range of Python dependencies that
are not applicable (and thus not required) for all users. The logic for
handling these optional dependencies (e.g. lazy-loading) is implemented here.

Note:
    This subpackage is typically not used by end users of data-describe.
"""
from data_describe.compat._dependency import _compat, _requires  # noqa: F401
from data_describe.compat._dataframe import (  # noqa: F401
    is_dataframe,
    is_series,
    _DATAFRAME_BACKENDS,
)
from data_describe.compat._notebook import _IN_NOTEBOOK  # noqa: F401
