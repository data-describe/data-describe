import plotly
import pytest

import data_describe as mw
from ._test_data import DATA


@pytest.fixture
def data():
    return DATA


def test_figure(data):
    fig = mw.correlation_matrix(data)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_figure_no_cluster(data):
    fig = mw.correlation_matrix(data, cluster=False)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_figure_categorical_no_cluster(data):
    fig = mw.correlation_matrix(data, categorical=True, cluster=False)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_figure_categorical_cluster(data):
    fig = mw.correlation_matrix(data, cluster=True, categorical=True)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_corr_values(data):
    corr_values = mw.correlation_matrix(data, return_values=True)
    num_cols = data.select_dtypes(["number"]).shape[1]
    assert corr_values.shape == (num_cols, num_cols)


def test_categorical_data_only(data):
    num_data = data.select_dtypes(["number"])
    cat_data = data[[c for c in data.columns if c not in num_data.columns]]
    corr_values = mw.correlation_matrix(cat_data, categorical=True, return_values=True)
    cat_cols = cat_data.shape[1]
    assert corr_values.shape == (cat_cols, cat_cols)


def test_categorical_data_only_but_specified_numeric(data):
    num_data = data.select_dtypes(["number"])
    cat_data = data[[c for c in data.columns if c not in num_data.columns]]
    with pytest.raises(ValueError):
        mw.correlation_matrix(cat_data, return_values=True)


def test_numeric_data_only_but_specified_categorical(data):
    num_data = data.select_dtypes(["number"])
    with pytest.warns(UserWarning):
        mw.correlation_matrix(num_data, categorical=True, return_values=True)
