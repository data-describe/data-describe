import pandas as pd

from data_describe.compat import _SERIES_TYPE, _DATAFRAME_TYPE, requires, _compat
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
    counts = series.value_counts()
    if counts.shape[0] == 0:
        return None
    return round(counts.iloc[0] / series.shape[0] * 100, 2)


def _pandas_compute_data_summary(data):
    """Perform computation for summary statistics and data description.

    Args:
        data: The dataframe

    Returns:
        The Pandas dataframe with metrics in rows
    """
    if isinstance(data, _SERIES_TYPE):
        data = pd.DataFrame(data, columns=[data.name])

    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Data must be a Pandas DataFrame")

    # Save column order
    columns = data.columns
    dtypes = data.agg([lambda x: x.dtype])
    moments = data.agg(["mean", "std", "median"])
    minmax = data.select_dtypes("number").agg(["min", "max"]).reindex(columns=columns)
    zeros = data.select_dtypes("number").agg([agg_zero]).reindex(columns=columns)
    null_summary = data.agg([agg_null])
    freq_summary = data.agg([most_frequent])

    summary = (
        dtypes.append(moments, ignore_index=True)
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
    try:
        return summary._to_pandas()
    except AttributeError:
        return summary


@requires("modin")
def _modin_compute_data_summary(data):
    """Perform computation for summary statistics and data description.

    Args:
        data: The dataframe

    Returns:
        The Modin dataframe with metrics in rows
    """
    if isinstance(data, _SERIES_TYPE):
        data = _compat["modin.pandas"].DataFrame(data)

    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Data must be a Modin DataFrame")

    # Save column order
    columns = data.columns
    dtypes = data.agg([lambda x: x.dtype])
    moments = data.agg(["mean", "std", "median"])
    minmax = data.select_dtypes("number").agg(["min", "max"]).reindex(columns=columns)
    zeros = data.select_dtypes("number").agg([agg_zero]).reindex(columns=columns)
    null_summary = data.agg([agg_null])
    freq_summary = data.agg([most_frequent])

    summary = (
        dtypes.append(moments, ignore_index=True)
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
