import plotly
import pytest
import pandas as pd
import numpy as np

import mwdata as mw


@pytest.fixture
def load_data():
    df = pd.DataFrame(
        {
            "a": np.random.normal(2, 1.2, size=250),
            "b": np.random.normal(3, 1.5, size=250),
            "c": np.random.choice(["x", "y"], size=250),
        }
    )
    return df


def test_figure(load_data):
    fig = mw.correlation_matrix(load_data)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_figure_no_cluster(load_data):
    fig = mw.correlation_matrix(load_data, cluster=False)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_figure_categorical_no_cluster(load_data):
    fig = mw.correlation_matrix(load_data, categorical=True, cluster=False)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_figure_categorical_cluster(load_data):
    fig = mw.correlation_matrix(load_data, cluster=True, categorical=True)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_corr_values(load_data):
    corr_values = mw.correlation_matrix(load_data, return_values=True)
    num_cols = load_data.select_dtypes(["number"]).shape[1]
    assert corr_values.shape == (num_cols, num_cols)


def test_categorical_data_only(load_data):
    num_data = load_data.select_dtypes(["number"])
    cat_data = load_data[[c for c in load_data.columns if c not in num_data.columns]]
    corr_values = mw.correlation_matrix(cat_data, categorical=True, return_values=True)
    cat_cols = cat_data.shape[1]
    assert corr_values.shape == (cat_cols, cat_cols)


def test_categorical_data_only_but_specified_numeric(load_data):
    num_data = load_data.select_dtypes(["number"])
    cat_data = load_data[[c for c in load_data.columns if c not in num_data.columns]]
    with pytest.raises(ValueError):
        mw.correlation_matrix(cat_data, return_values=True)


def test_numeric_data_only_but_specified_categorical(load_data):
    num_data = load_data.select_dtypes(["number"])
    with pytest.warns(UserWarning):
        mw.correlation_matrix(num_data, categorical=True, return_values=True)
