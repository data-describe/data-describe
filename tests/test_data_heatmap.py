import matplotlib
import pytest

import data_describe as mw
from ._test_data import DATA

matplotlib.use("Agg")


@pytest.fixture
def data():
    return DATA


def test_heatmap(data):
    with pytest.raises(OSError):
        mw.data_heatmap(data)


def test_heatmap_static(data):
    fig = mw.data_heatmap(data, interactive=False)
    assert isinstance(fig, matplotlib.artist.Artist)


def test_heatmap_missing(data):
    fig = mw.data_heatmap(data, missing=True, interactive=False)
    assert isinstance(fig, matplotlib.artist.Artist)


def test_heatmap_numpy(data):
    data = data.select_dtypes(["number"]).to_numpy()
    fig = mw.data_heatmap(data, interactive=False)
    assert isinstance(fig, matplotlib.artist.Artist)


def test_heatmap_invalid():
    data = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        mw.data_heatmap(data)
