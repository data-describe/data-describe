import modin.pandas as modin
from sklearn.decomposition import IncrementalPCA

from data_describe.core.summary import agg_null, agg_zero, most_frequent, cardinality


def summary(data, context=None):
    """ Summary statistics and data description
    Args:
        data: A Pandas data frame
        modin: A boolean flag for whether or not the data is a Modin Series or DataFrame
        context: The context
    Returns:
        Pandas data frame with metrics in rows
    """
    if isinstance(data, modin.pandas.Series):
        data = modin.DataFrame(data)

    if not isinstance(data, modin.pandas.DataFrame):
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
    return summary


def pca_type(data, n_components, column_names):
    pca = IncrementalPCA(n_components)
    reduc = pca.fit_transform(data)
    reduc_df = modin.DataFrame(reduc, columns=column_names)
    return reduc_df, pca
