import pandas as pd

from data_describe._widget import BaseWidget
from data_describe.compat import _requires, _compat, _is_series, _is_dataframe
from data_describe.backends._backends import _get_compute_backend


class SummaryWidget(BaseWidget):
    """Container for data summary.

    This class (object) is returned from the ``data_summary`` function. The
    attributes documented below can be accessed or extracted.

    Attributes:
        input_data: The input data.
        summary (DataFrame): The summary statistics.
    """

    def __init__(
        self,
        input_data=None,
        summary=None,
        **kwargs,
    ):
        """Data heatmap.

        Args:
            input_data: The input data.
            summary: The summary statistics.
        """
        super(SummaryWidget, self).__init__(**kwargs)
        self.input_data = input_data
        self.summary = summary

    def __str__(self):
        return "data-describe Summary Widget"

    def __repr__(self):
        return "data-describe Summary Widget"

    def show(self, viz_backend=None, **kwargs):
        """The default display for this output.

        Returns the summary data as a Pandas dataframe.

        Args:
            viz_backend: The visualization backend.
            **kwargs: Keyword arguments.

        Raises:
            ValueError: Summary data is missing.

        Returns:
            The correlation matrix plot.
        """
        if self.summary is None:
            raise ValueError("Could not find data to visualize.")

        return self.summary


def data_summary(data, compute_backend=None):
    """Summary statistics and data description with metrics in rows and original fields as columns.

    Args:
        data: The dataframe
        compute_backend: The compute backend.

    Returns:
        SummaryWidget
    """
    return _get_compute_backend(backend=compute_backend, df=data).compute_data_summary(
        data
    )


def _count_zeros(series):
    """Count of zero values in a Pandas or Modin series.

    Args:
        series: The Pandas or Modin series

    Returns:
        Number of zeros
    """
    return (series == 0).sum()


def _count_nulls(series):
    """Count of null values in a Pandas or Modin series.

    Args:
        series: The Pandas or Modin series

    Returns:
        Number of null values
    """
    return series.isnull().sum()


def _most_frequent(series):
    """Percent of most frequent value, per column, in a Pandas or Modin data frame.

    Args:
        series: The Pandas or Modin series

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

    Raises:
        ValueError: Invalid input data type.

    Returns:
        The Pandas dataframe with metrics in rows
    """
    if _is_series(data):
        data = pd.DataFrame(data, columns=[data.name])

    if not _is_dataframe(data):
        raise ValueError("Data must be a Pandas DataFrame")

    # Save column order
    columns = data.columns
    dtypes = data.agg([lambda x: x.dtype])
    moments = data.agg(["mean", "std", "median"])
    minmax = data.select_dtypes("number").agg(["min", "max"]).reindex(columns=columns)
    zeros = data.select_dtypes("number").agg([_count_zeros]).reindex(columns=columns)
    null_summary = data.agg([_count_nulls])
    freq_summary = data.agg([_most_frequent])

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

    return SummaryWidget(data, summary)


@_requires("modin")
def _modin_compute_data_summary(data):
    """Perform computation for summary statistics and data description.

    Args:
        data: The dataframe

    Raises:
        ValueError: Invalid input data type.

    Returns:
        The Modin dataframe with metrics in rows
    """
    if _is_series(data):
        data = _compat["modin.pandas"].DataFrame(data)

    if not _is_dataframe(data):
        raise ValueError("Data must be a Modin DataFrame")

    # Save column order
    columns = data.columns
    dtypes = data.agg([lambda x: x.dtype])
    moments = data.agg(["mean", "std", "median"])
    minmax = data.select_dtypes("number").agg(["min", "max"]).reindex(columns=columns)
    zeros = data.select_dtypes("number").agg([_count_zeros]).reindex(columns=columns)
    null_summary = data.agg([_count_nulls])
    freq_summary = data.agg([_most_frequent])

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

    return SummaryWidget(data, summary)
