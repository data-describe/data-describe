from data_describe.utilities.contextmanager import _context_manager
from data_describe.compat import _PACKAGE_INSTALLED
from data_describe.backends._backends import _get_compute_backend

if _PACKAGE_INSTALLED["modin"]:
    import modin.pandas as modin


@_context_manager
def data_summary(data, context=None, compute_backend=None):
    """ Summary statistics and data description
    Args:
        data: A Pandas data frame
        modin: A boolean flag for whether or not the data is a Modin Series or DataFrame
        context: The context
    Returns:
        Pandas data frame with metrics in rows
    """
    return _get_compute_backend(compute_backend).compute_data_summary(data)


def agg_zero(series):
    """ Count of zero values in a pandas series
    Args:
        series: A Pandas series
    Returns:
        Number of zeros
    """
    return (series == 0).sum()


def agg_null(series):
    """ Count of null values in a pandas series
    Args:
        series: A Pandas series
    Returns:
        Number of null values
    """
    return series.isnull().sum()


def most_frequent(series):
    """ Percent of most frequent value, per column, in a pandas data frame
    Args:
        data: A Pandas data frame
    Returns:
        Percent of most frequent value per column
    """
    top = series.mode().iloc[0]
    m_freq = round(series.isin([top]).sum() / series.shape[0] * 100, 2)
    return m_freq


def cardinality(series):
    """ Number of unique values in a series
    Args:
        series: A Pandas series
    Returns:
        Number of unique values
    """
    series = series.dropna().values
    return len(set(series))
