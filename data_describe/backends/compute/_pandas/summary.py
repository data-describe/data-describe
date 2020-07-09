import pandas as pd

from data_describe.compat import _SERIES_TYPE, _DATAFRAME_TYPE
from data_describe.core.summary import agg_null, agg_zero, most_frequent


def compute_data_summary(data, context=None):
    if isinstance(data, _SERIES_TYPE):
        data = pd.DataFrame(data, columns=[data.name])

    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Data must be a Pandas DataFrame")

    # Save column order
    columns = data.columns

    dtypes = data.agg([lambda x: x.dtype])

    moments = data.agg(["mean", "std", "median"])

    # Non-numerical columns given NaN values for min/max and zeros
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
