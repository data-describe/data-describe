from sklearn.preprocessing import StandardScaler

from data_describe.compat import _DATAFRAME_TYPE


def process_data_heatmap(data, missing=False, **kwargs):
    """Pre-processes data for the data heatmap

    Values are standardized (removing the mean and scaling to unit variance).
    If `missing` is set to True, the dataframe flags missing records using 1/0.

    Args:
        data: The dataframe
        missing: If True, uses missing values instead
        kwargs: Not implemented

    Returns:
        (dataframe, column_names)
    """
    if isinstance(data, _DATAFRAME_TYPE):
        data = data.select_dtypes(["number"])
        colnames = data.columns.values
    else:
        raise ValueError("Unsupported input data type")

    if missing:
        data = data.isna().astype(int)
    else:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    return data, colnames
