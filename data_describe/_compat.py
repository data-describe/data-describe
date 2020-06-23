try:
    import modin.pandas
    import pandas

    # _PACKAGE_INSTALLED["modin"] = True
    _MODIN_INSTALLED = True
    _FRAME_TYPE = (pandas.DataFrame, modin.pandas.DataFrame)
    _SERIES_TYPE = (pandas.Series, modin.pandas.Series)
except (ImportError, ModuleNotFoundError):
    import pandas

    # _PACKAGE_INSTALLED["modin"] = False
    _MODIN_INSTALLED = False
    _FRAME_TYPE = pandas.DataFrame
    _SERIES_TYPE = pandas.Series