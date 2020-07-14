from pyscagnostics import scagnostics


def compute_scatter_plot(data, mode, sample, threshold, **kwargs):
    data = data.select_dtypes(["number"])
    if mode == "diagnostic":
        diagnostics = scagnostics(data)
        return data, diagnostics
    else:
        return data, None
