import matplotlib
import pytest
import plotly

import data_describe as dd
from data_describe.core.heatmap import HeatmapWidget

matplotlib.use("Agg")


@pytest.mark.base
def test_heatmap_matplotlib(compute_backend_df):
    w = dd.data_heatmap(compute_backend_df)
    assert isinstance(w, HeatmapWidget), "Not a Heatmap Widget"
    assert isinstance(w.show(), matplotlib.axes.Axes)


@pytest.mark.base
def test_heatmap_missing(compute_backend_df):
    w = dd.data_heatmap(compute_backend_df, missing=True)
    assert isinstance(w, HeatmapWidget)
    assert w.missing is True, "Missing param was not assigned to the widget"
    assert w.missing_data is not None, "`Missing dataframe` is missing"


@pytest.mark.base
def test_heatmap_plotly(compute_backend_df):
    w = dd.data_heatmap(compute_backend_df)
    assert isinstance(w.show(viz_backend="plotly"), plotly.graph_objs.Figure)


@pytest.mark.base
def test_heatmap_invalid(compute_backend_df):
    data = [1, 2, 3, 4]
    with pytest.raises((ValueError, ModuleNotFoundError)):
        dd.data_heatmap(data)
