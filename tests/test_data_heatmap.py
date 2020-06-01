import matplotlib
import pytest
import pandas as pd
import numpy as np

import mwdata as mw

matplotlib.use("Agg")


@pytest.fixture
def data():
    df = pd.DataFrame(
        {
            "a": np.random.normal(2, 1.2, size=250),
            "b": np.random.normal(3, 1.5, size=250),
            "c": np.random.choice(["x", "y"], size=250),
        }
    )
    return df

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
