import pandas as pd


def data_summary(data):
    """ Summary statistics and data description

    Args:
        data: A Pandas data frame


    Returns:
        Pandas data frame with metrics in rows
    """
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    if isinstance(data, pd.DataFrame):
        # Save column order
        columns = data.columns

        dtypes = data.dtypes
        dtypes.name = "Data Type"
        dtypes = pd.DataFrame(dtypes).transpose()

        moments = data.select_dtypes(["number", "datetime"])
        if moments.shape[1] > 0:
            moments_summary = moments.agg(["mean", "std", "median"])
        else:
            moments_summary = pd.DataFrame([], columns=data.columns)

        minmax = data.select_dtypes(["number", "datetime", "bool"])
        if minmax.shape[1] > 0:
            minmax_summary = minmax.agg(["min", "max"])
        else:
            minmax_summary = pd.DataFrame([], columns=data.columns)

        zeros = data.select_dtypes(["number", "bool"])
        if zeros.shape[1] > 0:
            zeros_summary = zeros.agg([agg_zero])
        else:
            zeros_summary = pd.DataFrame([], columns=data.columns)

        null_summary = data.agg([agg_null])

        freq_summary = most_frequent(data)
        freq_summary = pd.DataFrame(freq_summary).transpose()

        summary = pd.concat(
            [
                dtypes,
                moments_summary,
                minmax_summary,
                zeros_summary,
                null_summary,
                freq_summary,
            ],
            sort=True,
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
    else:
        raise ValueError("Data must be a Pandas DataFrame")


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


def most_frequent(data):
    """ Percent of most frequent value, per column, in a pandas data frame

    Args:
        data: A Pandas data frame

    Returns:
        Percent of most frequent value per column
    """
    top = data.mode().head(1)
    if isinstance(data, pd.Series):
        m_freq = round((data == top[0]).sum() / data.shape[0] * 100, 2)
    elif isinstance(data, pd.DataFrame):
        m_freq = round(
            data.apply(lambda x: x == top[x.name][0]).sum(axis=0) / data.shape[0] * 100,
            2,
        )
    else:
        raise ValueError("Data must be a Pandas Series or Dataframe.")
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
