import matplotlib
import pytest
import plotly.graph_objects as go

import data_describe as dd

matplotlib.use("Agg")


@pytest.mark.base
def test_heatmap_matplotlib(compute_backend_df):
    fig = dd.data_heatmap(compute_backend_df)
    assert isinstance(fig, matplotlib.artist.Artist)


@pytest.mark.base
def test_heatmap_missing(compute_backend_df):
    fig = dd.data_heatmap(compute_backend_df, missing=True)
    assert isinstance(fig, matplotlib.artist.Artist)


@pytest.mark.base
def test_heatmap_plotly(compute_backend_df):
    fig = dd.data_heatmap(compute_backend_df, viz_backend="plotly")
    assert isinstance(fig, go.Figure)


@pytest.mark.base
def test_heatmap_invalid(compute_backend_df):
    data = [1, 2, 3, 4]
    with pytest.raises((ValueError, ModuleNotFoundError)):
        dd.data_heatmap(data)
