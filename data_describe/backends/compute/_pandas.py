import pandas as pd
from sklearn.preprocessing import StandardScaler


def process_data_heatmap(data, missing=False, **kwargs):
    if isinstance(data, pd.DataFrame):  # TODO: compat frame type
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
