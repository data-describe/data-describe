from pyscagnostics import scagnostics

from data_describe.compat import _DATAFRAME_TYPE


def compute_scatter_plot(data, mode, sample, threshold, **kwargs):
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Unsupported input data type")

    data = data.select_dtypes(["number"])
    if mode == "diagnostic":
        diagnostics = scagnostics(data)
        return data, diagnostics
    else:
        return data, None
