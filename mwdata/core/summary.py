import pandas as pd
import numpy as np
from mwdata.utilities.contextmanager import _context_manager


@_context_manager
def data_summary(data, context=None):
    """ Summary statistics and data description

    Args:
        data: A Pandas data frame
        context: The context

    Returns:
        Pandas data frame with metrics in rows
    """
    if isinstance(data, pd.Series):
        dtypes = pd.Series([data.dtype])
        dtypes.name = "Data Type"

        if data.dtype in [np.int64, np.float64]:
            num_summ = data.agg(['mean', 'std', 'median', 'min', 'max', zeros])
        else:
            nan_array = np.empty(6, )
            nan_array[:] = np.nan
            num_summ = pd.Series(nan_array)

        null_summ = data.agg([null])

        freq_summ = pd.Series([most_frequent(data)])
        freq_summ.name = "% Most Frequent Value"
        freq_summ = pd.Series(freq_summ).transpose()

        summary = pd.concat([dtypes, num_summ, null_summ, freq_summ], sort=True)
        summary.index = ['Data Type', 'Mean', 'Standard Deviation',
                         'Median', 'Min', 'Max', '# Zeros',
                         '# Nulls', '% Most Frequent Value']
        summary.name = data.name
        summary = pd.DataFrame(summary, columns=[data.name])

    elif isinstance(data, pd.DataFrame):
        # Save column order
        columns = data.columns

        dtypes = data.dtypes
        dtypes.name = "Data Type"
        dtypes = pd.DataFrame(dtypes).transpose()

        num_df = data.select_dtypes(['number'])
        num_summ = num_df.agg(['mean', 'std', 'median', 'min', 'max', zeros])

        null_summ = data.agg([null])

        freq_summ = most_frequent(data)
        freq_summ.name = "% Most Frequent Value"
        freq_summ = pd.DataFrame(freq_summ).transpose()

        summary = pd.concat([dtypes, num_summ, null_summ, freq_summ], sort=True)
        summary = summary[columns]
        summary.index = ['Data Type', 'Mean', 'Standard Deviation',
                         'Median', 'Min', 'Max', '# Zeros',
                         '# Nulls', '% Most Frequent Value']

    # Removing NaNs
    summary.fillna("", inplace=True)
    return summary


def zeros(series):
    """ Count of zero values in a pandas series

    Args:
        series: A Pandas series

    Returns:
        Number of zeros
    """
    return (series == 0).sum()


def null(series):
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
        m_freq = round(data.apply(lambda x: x == top[x.name][0]).sum(axis=0) / data.shape[0] * 100, 2)
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
