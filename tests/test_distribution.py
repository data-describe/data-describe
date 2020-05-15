import mwdata as mw
import pandas as pd
import numpy as np
import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture
def data():
    df = pd.read_csv("data/weatherAUS.csv")
    df = df[["Location", "Rainfall", "RainToday"]]
    return df


def test_distribution(data):
    fig = mw.distribution(data)
    assert isinstance(fig, list)
    assert isinstance(fig[0], matplotlib.artist.Artist)


def test_distribution_all(data):
    fig = mw.distribution(data, plot_all=True)
    assert isinstance(fig, list)
    assert isinstance(fig[0], matplotlib.artist.Artist)


def test_distribution_cats(data):
    fig = mw.distribution(data, max_categories=None)
    assert isinstance(fig, list)
    assert isinstance(fig[0], matplotlib.artist.Artist)


def test_distribution_nonimplemented():
    err_type = np.array([1, 2, 3])
    with pytest.raises(NotImplementedError):
        mw.distribution(err_type)
