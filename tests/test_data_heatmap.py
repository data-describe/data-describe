import matplotlib
import pytest
import plotly.graph_objects as go

import data_describe as dd

matplotlib.use("Agg")


@pytest.mark.base
def test_heatmap_matplotlib(data, compute_backend):
    fig = dd.data_heatmap(data)
    assert isinstance(fig, matplotlib.artist.Artist)


@pytest.mark.base
def test_heatmap_missing(data, compute_backend):
    fig = dd.data_heatmap(data, missing=True)
    assert isinstance(fig, matplotlib.artist.Artist)


@pytest.mark.base
def test_heatmap_plotly(data, compute_backend):
    fig = dd.data_heatmap(data, viz_backend="plotly")
    assert isinstance(fig, go.Figure)


@pytest.mark.base
def test_heatmap_invalid(compute_backend):
    data = [1, 2, 3, 4]
    with pytest.raises((ValueError, ModuleNotFoundError)):
        dd.data_heatmap(data)
