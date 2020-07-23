from data_describe.backends._backends import _get_compute_backend


def data_summary(data, compute_backend=None):
    """Summary statistics and data description.

    Args:
        data: The dataframe

    Returns:
        The dataframe with metrics in rows
    """
    return _get_compute_backend(backend=compute_backend, df=data).compute_data_summary(
        data
    )


def agg_zero(series):
    """Count of zero values in a Pandas or Modin series.

    Args:
        series: The Pandas or Modin series

    Returns:
        Number of zeros
    """
    return (series == 0).sum()


def agg_null(series):
    """Count of null values in a Pandas or Modin series.

    Args:
        series: The Pandas or Modin series

    Returns:
        Number of null values
    """
    return series.isnull().sum()


def most_frequent(series):
    """Percent of most frequent value, per column, in a Pandas or Modin data frame.

    Args:
        data: The Pandas or Modin dataframe

    Returns:
        Percent of most frequent value per column
    """
    top = series.mode().iloc[0]
    m_freq = round(series.isin([top]).sum() / series.shape[0] * 100, 2)
    return m_freq
