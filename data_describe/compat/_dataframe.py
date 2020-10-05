from collections import namedtuple

import pandas

from data_describe.compat._dependency import _compat

_DATAFRAME_BACKENDS = {
    "<class 'pandas.core.frame.DataFrame'>": ["pandas"],
    "<class 'pandas.core.series.Series'>": ["pandas"],
    "<class 'modin.pandas.dataframe.DataFrame'>": ["modin", "pandas"],
    "<class 'modin.pandas.series.Series'>": ["modin", "pandas"],
}

if _compat.check_install("modin"):
    compute = namedtuple("compute", ["pandas", "modin"])
    _SERIES_TYPE = compute(
        pandas=pandas.Series, modin=_compat["modin.pandas"].Series  # type: ignore
    )
    _DATAFRAME_TYPE = compute(
        pandas=pandas.DataFrame, modin=_compat["modin.pandas"].DataFrame  # type: ignore
    )
else:
    _SERIES_TYPE = pandas.Series
    _DATAFRAME_TYPE = pandas.DataFrame
