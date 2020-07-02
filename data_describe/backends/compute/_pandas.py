import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA

from data_describe.compat import _SERIES_TYPE, _DATAFRAME_TYPE
from data_describe.core.summary import agg_null, agg_zero, most_frequent


def compute_data_summary(data, context=None):
    """ Summary statistics and data description
    Args:
        data: A Pandas data frame
        modin: A boolean flag for whether or not the data is a Modin Series or DataFrame
        context: The context
    Returns:
        Pandas data frame with metrics in rows
    """
    if isinstance(data, _SERIES_TYPE):
        data = pd.DataFrame(data)

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
    return summary


def compute_run_pca(data, n_components, column_names):
    pca = PCA(n_components)
    reduc = pca.fit_transform(data)
    reduc_df = pd.DataFrame(reduc, columns=column_names)
    return reduc_df, pca


def compute_run_ipca(data, n_components, column_names):
    ipca = IncrementalPCA(n_components)
    reduc = ipca.fit_transform(data)
    reduc_df = pd.DataFrame(reduc, columns=column_names)
    return reduc_df, ipca


def compute_run_tsne(reduc):
    return pd.DataFrame(reduc, columns=["ts1", "ts2"])


def compute_run_tsvd(reduc, column_names):
    return pd.DataFrame(reduc, columns=column_names)
