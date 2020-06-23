from data_describe.utilities.contextmanager import _context_manager
from data_describe._compat import _MODIN_INSTALLED, _SERIES_TYPE, _FRAME_TYPE

if _MODIN_INSTALLED:
    import modin.pandas as frame
else:
    import pandas as frame


@_context_manager
def data_summary(data, context=None):
    """ Summary statistics and data description
    Args:
        data: A Pandas data frame
        modin: A boolean flag for whether or not the data is a Modin Series or DataFrame
        context: The context
    Returns:
        Pandas data frame with metrics in rows
    """
    if isinstance(data, _SERIES_TYPE):
        data = frame.DataFrame(data)

    if not isinstance(data, _FRAME_TYPE):
        raise ValueError("Data must be a Pandas (or Modin) DataFrame")

    # Save column order
    columns = data.columns

    dtypes = data.agg([lambda x: x.dtype])

    moments = data.agg(["mean", "std", "median"])

    # Non-numerical columns given NaN values for min/max and zeros
    minmax = data.select_dtypes('number').agg(["min", "max"]).reindex(columns=columns)

    zeros = data.select_dtypes('number').agg([agg_zero]).reindex(columns=columns)

    null_summary = data.agg([agg_null])

    freq_summary = data.agg([most_frequent])

    summary = (
        dtypes
        .append(moments, ignore_index=True)
        .append(minmax, ignore_index=True)
        .append(zeros, ignore_index=True)
        .append(null_summary, ignore_index=True)
        .append(freq_summary, ignore_index=True)
    )
    summary = summary[columns]
    summary.index = [
        "Data Type",
        "Mean",
        "Standard Deviation",
        "Median",
        "Min",
        "Max",
        "# Zeros",
        "# Nulls",
        "% Most Frequent Value",
    ]

    # Removing NaNs
    summary.fillna("", inplace=True)
    return summary


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
