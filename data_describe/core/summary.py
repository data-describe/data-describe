from typing import Dict, Callable

import pandas as pd
from pandas.core.dtypes.common import is_float
from pandas.io.formats.info import _sizeof_fmt
import numpy as np

from data_describe._widget import BaseWidget
from data_describe.compat import _requires, _compat, _is_series, _is_dataframe
from data_describe.backends._backends import _get_compute_backend


class SummaryWidget(BaseWidget):
    """Container for data summary.

    This class (object) is returned from the ``data_summary`` function. The
    attributes documented below can be accessed or extracted.

    Attributes:
        input_data: The input data.
        info_data: Information about the data shape and size.
        summary_data (DataFrame): The summary statistics.
        as_percentage (bool): If True, display counts as percentage over total
        auto_float (bool): If True, apply formatting to float values
    """

    def __init__(
        self,
        input_data=None,
        info_data=None,
        summary_data=None,
        as_percentage: bool = False,
        auto_float: bool = True,
        **kwargs,
    ):
        """Data heatmap.

        Args:
            input_data: The input data.
            info_data: Information about the data shape and size.
            summary_data: The summary statistics.
            as_percentage (bool): If True, display counts as percentage over total
            auto_float (bool): If True, apply formatting to float values
        """
        super(SummaryWidget, self).__init__(**kwargs)
        self.input_data = input_data
        self.info_data = info_data
        self.summary_data = summary_data
        self.as_percentage = as_percentage
        self.auto_float = auto_float

    def __str__(self):
        return "data-describe Summary Widget"

    def __repr__(self):
        return "data-describe Summary Widget"

    def show(
        self, viz_backend=None, as_percentage: bool = False, auto_float=True, **kwargs
    ):
        """The default display for this output.

        Displays the summary information.

        Args:
            viz_backend: The visualization backend.
            as_percentage (bool): If True, display counts as percentage over total
            auto_float (bool): If True, apply formatting to float values
            **kwargs: Keyword arguments.

        Returns:
            The correlation matrix plot.
        """
        try:
            from IPython.display import display

            view = display
        except ImportError:
            view = print

        view(self.info_data)

        summary_data = self.summary_data
        format_dict: Dict[str, Callable] = {}

        as_percentage = as_percentage or self.as_percentage
        if as_percentage:
            for col in ["Zeros", "Nulls", "Top Frequency"]:
                summary_data[col] = summary_data[col] / self.input_data.shape[0]
                format_dict[col] = "{:.1%}".format

        summary_data.fillna("", inplace=True)
        auto_float = auto_float or self.auto_float
        if auto_float:
            for col in summary_data.columns:
                if col not in format_dict.keys():
                    format_dict[col] = _value_formatter

        view(summary_data.style.format(format_dict))


def data_summary(
    data, as_percentage: bool = False, auto_float: bool = True, compute_backend=None
):
    """Summary statistics and data description.

    Args:
        data: The dataframe
        as_percentage (bool): If True, display counts as percentage over total
        auto_float (bool): If True, apply formatting to float values
        compute_backend: The compute backend.

    Returns:
        The dataframe with metrics in rows
    """
    widget = _get_compute_backend(
        backend=compute_backend, df=data
    ).compute_data_summary(data)
    widget.as_percentage = as_percentage
    widget.auto_float = auto_float
    return widget


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


def mode1(x):
    """Mode (counts only) by Warren Weckesser.

    https://stackoverflow.com/questions/46365859/what-is-the-fastest-way-to-get-the-mode-of-a-numpy-array

    Args:
        x: Input array
    """
    _, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return counts[m]


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

    info_data = pd.DataFrame(
        {
            "Info": [
                data.shape[0],
                data.shape[1],
                _sizeof_fmt(data.memory_usage().sum(), ""),
            ]
        },
        index=["Rows", "Columns", "Size in Memory"],
    )

    columns = data.columns
    val = data.values
    num_columns = data.select_dtypes("number").columns
    num_ind = np.nonzero([c in num_columns for c in columns])[0]
    date_columns = data.select_dtypes(["datetime", "datetimetz"]).columns
    date_ind = np.nonzero([c in date_columns for c in columns])[0]
    other_columns = data.select_dtypes(
        exclude=["number", "datetime", "datetimetz"]
    ).columns
    other_ind = np.nonzero([c in other_columns for c in columns])[0]
    order = np.concatenate([num_ind, date_ind, other_ind], axis=0)

    dtypes = data.dtypes[order]
    s_mean = np.pad(
        np.mean(val[:, num_ind], axis=0),
        (0, len(data.columns) - num_ind.size),
        constant_values=np.nan,
    )
    s_sd = np.pad(
        np.std(val[:, num_ind].astype(np.float), axis=0),
        (0, len(data.columns) - num_ind.size),
        constant_values=np.nan,
    )
    s_med = np.pad(
        np.median(val[:, num_ind], axis=0),
        (0, len(data.columns) - num_ind.size),
        constant_values=np.nan,
    )
    s_min = np.pad(
        np.min(val[:, np.concatenate([num_ind, date_ind])], axis=0),
        (0, len(data.columns) - num_ind.size - date_ind.size),
        constant_values=np.nan,
    )
    s_max = np.pad(
        np.max(val[:, np.concatenate([num_ind, date_ind])], axis=0),
        (0, len(data.columns) - num_ind.size - date_ind.size),
        constant_values=np.nan,
    )
    s_zero = data[data == 0].fillna(0).sum().astype(int)[order]
    s_null = data.isnull().sum().astype(int)[order]
    s_unique = data.nunique()[order]
    s_freq = np.apply_along_axis(mode1, 0, val.astype("str"))[order]

    summary_data = pd.DataFrame(
        np.vstack(
            [
                dtypes,
                s_null,
                s_zero,
                s_min,
                s_med,
                s_max,
                s_mean,
                s_sd,
                s_unique,
                s_freq,
            ]
        ).transpose()[np.argsort(order), :],
        columns=[
            "Data Type",
            "Nulls",
            "Zeros",
            "Min",
            "Median",
            "Max",
            "Mean",
            "Standard Deviation",
            "Unique",
            "Top Frequency",
        ],
        index=data.columns,
    )

    return SummaryWidget(data, info_data, summary_data)


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
        "Zeros",
        "Nulls",
        "Top Frequency",
    ]

    return SummaryWidget(data, summary)


def _get_precision(x, margin: int = 1) -> int:
    """Get the minimum precision for a value.

    Used to determine how to display and format floats.

    Args:
        x: Input value
        margin (int): Added to the calculated precision

    Returns:
        int: Number of decimal places
    """
    try:
        x = (x - np.trunc(x)).astype(np.float)
        magnitude = np.log10(x[x != 0])
        return int(np.ceil(np.abs(magnitude)))
    except TypeError:
        return 0


def _value_formatter(x, precision=None):
    """Formatter for displaying values of mixed type and precision.

    Args:
        x: Input value
        precision (int, optional): Precision for floats.
    """
    if is_float(x):
        precision = precision or _get_precision(x)
        try:
            return f"{{:.{precision}f}}".format(x)
        except ValueError:
            pass
    else:
        return x
