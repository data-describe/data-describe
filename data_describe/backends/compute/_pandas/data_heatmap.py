from sklearn.preprocessing import StandardScaler

from data_describe.compat import _DATAFRAME_TYPE


def process_data_heatmap(data, missing=False, **kwargs):
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
