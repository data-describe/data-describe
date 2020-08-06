import plotly
import pytest
import matplotlib
from matplotlib.axes import Axes as mpl_plot
from pandas._testing import assert_frame_equal

import data_describe as dd
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.core.correlation_matrix import CorrelationMatrixWidget

matplotlib.use("Agg")


def test_not_df():
    with pytest.raises(ValueError):
        dd.correlation_matrix("this_is_a_string")


def test_cluster_widget():
    cr = CorrelationMatrixWidget()
    assert hasattr(cr, "association_matrix"), "Cluster Widget missing cluster labels"
    assert hasattr(cr, "viz_backend"), "Cluster Widget missing default viz backend"
    assert hasattr(cr, "cluster_matrix"), "Cluster Widget missing cluster matrix"
    assert hasattr(cr, "categorical"), "Cluster Widget missing categorical flag"
    assert hasattr(cr, "viz_data"), "Cluster Widget missing viz_data"
    assert hasattr(cr, "__repr__"), "Cluster Widget missing __repr__ method"
    assert hasattr(cr, "_repr_html_"), "Cluster Widget missing _repr_html_ method"
    assert hasattr(cr, "show"), "Cluster Widget missing show method"


def test_figure_default(data):
    cr = dd.correlation_matrix(data)
    assert isinstance(cr.show(viz_backend="plotly"), plotly.graph_objs._figure.Figure)
    assert isinstance(cr.show(), mpl_plot)
    assert isinstance(cr.association_matrix, _DATAFRAME_TYPE)
    assert isinstance(cr.viz_data, _DATAFRAME_TYPE)
    assert isinstance(cr, CorrelationMatrixWidget)
    assert_frame_equal(cr.viz_data, cr.association_matrix)
    assert data.select_dtypes(["number"]).shape[1] == cr.association_matrix.shape[1]
    assert data.select_dtypes(["number"]).shape[1] == cr.association_matrix.shape[0]
    assert data.select_dtypes(["number"]).shape[1] == cr.viz_data.shape[1]
    assert data.select_dtypes(["number"]).shape[1] == cr.viz_data.shape[0]
    assert isinstance(cr.cluster_matrix, type(None))


def test_figure_categorical_cluster(data):
    cr = dd.correlation_matrix(data, cluster=True, categorical=True)
    assert isinstance(cr.show(viz_backend="plotly"), plotly.graph_objs._figure.Figure)
    assert isinstance(cr.show(), mpl_plot)
    assert isinstance(cr.association_matrix, _DATAFRAME_TYPE)
    assert isinstance(cr.association_matrix, _DATAFRAME_TYPE)
    assert isinstance(cr.viz_data, _DATAFRAME_TYPE)
    assert isinstance(cr, CorrelationMatrixWidget)
    assert_frame_equal(cr.viz_data, cr.cluster_matrix)


def test_cluster_no_categorical_figure(data):
    cr = dd.correlation_matrix(data, cluster=True)
    assert isinstance(cr.show(viz_backend="plotly"), plotly.graph_objs._figure.Figure)
    assert isinstance(cr.show(), mpl_plot)
    assert isinstance(cr.association_matrix, _DATAFRAME_TYPE)
    assert isinstance(cr.viz_data, _DATAFRAME_TYPE)
    assert isinstance(cr, CorrelationMatrixWidget)
    assert_frame_equal(cr.viz_data, cr.cluster_matrix)
    assert data.select_dtypes(["number"]).shape[1] == cr.association_matrix.shape[1]
    assert data.select_dtypes(["number"]).shape[1] == cr.association_matrix.shape[0]
    assert data.select_dtypes(["number"]).shape[1] == cr.viz_data.shape[1]
    assert data.select_dtypes(["number"]).shape[1] == cr.viz_data.shape[0]


def test_categorical_data_only(data):
    cr = dd.correlation_matrix(data, categorical=True, cluster=False)
    assert isinstance(cr.show(viz_backend="plotly"), plotly.graph_objs._figure.Figure)
    assert isinstance(cr.show(), mpl_plot)
    assert isinstance(cr.association_matrix, _DATAFRAME_TYPE)
    assert isinstance(cr.viz_data, _DATAFRAME_TYPE)
    assert isinstance(cr, CorrelationMatrixWidget)
    assert_frame_equal(cr.viz_data, cr.association_matrix)
    assert data.shape[1] == cr.association_matrix.shape[1]
    assert data.shape[1] == cr.association_matrix.shape[0]
    assert data.shape[1] == cr.viz_data.shape[1]
    assert data.shape[1] == cr.viz_data.shape[0]
    assert isinstance(cr.cluster_matrix, type(None))


def test_categorical_data_only_but_specified_numeric(data):
    num_data = data.select_dtypes(["number"])
    cat_data = data[[c for c in data.columns if c not in num_data.columns]]
    with pytest.raises(ValueError):
        dd.correlation_matrix(cat_data)


def test_numeric_data_only_but_specified_categorical(data):
    num_data = data.select_dtypes(["number"])
    with pytest.warns(UserWarning):
        dd.correlation_matrix(num_data, categorical=True)
