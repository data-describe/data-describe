from sklearn.preprocessing import StandardScaler
from pyscagnostics import scagnostics

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


def process_scatter_plot(data, mode, sample, threshold, **kwargs):
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")

    data = data.select_dtypes(["number"])
    if mode == "diagnostic":
        diagnostics = scagnostics(data)
        return data, diagnostics
    else:
        return data, None
