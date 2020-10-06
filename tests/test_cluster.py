import numpy as np
import pytest
import matplotlib
from matplotlib.axes import Axes as mpl_plot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly

import data_describe as dd
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.core.clustering import (
    ClusterWidget,
    KmeansClusterWidget,
    HDBSCANClusterWidget,
    _pandas_compute_cluster,
    _run_kmeans,
    _find_clusters,
    _fit_kmeans,
    _run_hdbscan,
)

matplotlib.use("Agg")


@pytest.mark.base
def test_not_df():
    with pytest.raises(ValueError):
        dd.cluster("this_is_a_string")


@pytest.mark.base
def test_method_not_implemented(numeric_data):
    with pytest.raises(ValueError):
        dd.cluster(numeric_data, method="unimplemented")


@pytest.mark.base
def test_cluster_widget():
    cl = ClusterWidget()
    assert hasattr(cl, "clusters"), "Cluster Widget missing cluster labels"
    assert hasattr(cl, "viz_backend"), "Cluster Widget missing default viz backend"
    assert hasattr(cl, "__repr__"), "Cluster Widget missing __repr__ method"
    assert hasattr(cl, "_repr_html_"), "Cluster Widget missing _repr_html_ method"
    assert hasattr(cl, "show"), "Cluster Widget missing show method"


@pytest.mark.base
def test_kmeans_cluster_widget():
    kcl = KmeansClusterWidget()
    assert isinstance(
        kcl, ClusterWidget
    ), "Kmeans Cluster Widget not a subclass of Cluster Widget"
    assert hasattr(kcl, "method"), "Kmeans Cluster Widget missing `method` attribute"
    assert (
        kcl.method == "kmeans"
    ), "Kmeans Cluster Widget has the wrong cluster method attribute"
    assert hasattr(
        kcl, "estimator"
    ), "Kmeans Cluster Widget missing estimator attribute"
    assert hasattr(
        kcl, "n_clusters"
    ), "Kmeans Cluster Widget missing `n_clusters` attribute"
    assert hasattr(kcl, "search"), "Kmeans Cluster Widget missing `search` attribute"
    assert hasattr(
        kcl, "cluster_range"
    ), "Kmeans Cluster Widget `cluster_range` attribute"
    assert hasattr(kcl, "metric"), "Kmeans Cluster Widget missing `metric` attribute"
    assert hasattr(
        kcl, "estimator"
    ), "Kmeans Cluster Widget missing `estimator` attribute"
    assert hasattr(kcl, "scores"), "Kmeans Cluster Widget missing `scores` attribute"


def test_hdbscan_cluster_widget(_hdbscan):
    hcl = HDBSCANClusterWidget()
    assert isinstance(
        hcl, ClusterWidget
    ), "HDBSCAN Cluster Widget not a subclass of Cluster Widget"
    assert hasattr(hcl, "method"), "HDBSCANCluster Widget missing `method` attribute"
    assert (
        hcl.method == "hdbscan"
    ), "HDBSCAN Cluster Widget has the wrong cluster method attribute"
    assert hasattr(
        hcl, "estimator"
    ), "HDBSCAN Cluster Widget missing `estimator` attribute"


@pytest.fixture
def kmeans_default(numeric_data):
    return dd.cluster(numeric_data, method="kmeans")


@pytest.mark.base
def test_kmeans_default(kmeans_default):
    cl = kmeans_default
    assert isinstance(
        cl.show(), matplotlib.axes.Axes
    ), "Default show() didn't return a mpl Axes object"
    assert isinstance(cl, KmeansClusterWidget)
    assert isinstance(cl.estimator, KMeans), "Saved cluster estimator was not KMeans"
    assert hasattr(cl, "input_data"), "Widget does not have input data"
    assert isinstance(cl.input_data, _DATAFRAME_TYPE), "Input data is not a data frame"
    assert hasattr(cl, "scaled_data"), "Widget does not have standardized data"
    assert isinstance(
        cl.scaled_data, _DATAFRAME_TYPE
    ), "Scaled data is not a data frame"
    assert hasattr(cl, "viz_data"), "Widget does not have visualization (reduced) data"
    assert isinstance(cl.viz_data, _DATAFRAME_TYPE), "Viz data is not a data frame"
    assert cl.clusters is not None, "Widget is missing cluster labels"
    assert cl.n_clusters == 19, "Expected number of clusters found to be 19"
    assert isinstance(cl.cluster_range, tuple), "Widget is missing cluster range tuple"
    assert len(cl.cluster_range) == 2, "Cluster range tuple had the wrong shape"
    assert isinstance(cl.cluster_range[0], int) and isinstance(
        cl.cluster_range[1], int
    ), "Cluster range tuple does not contain integers"
    assert (
        cl.cluster_range[0] < cl.cluster_range[1]
    ), "Cluster range had an invalid default; the maximum is <= the minimum"
    assert (
        cl.metric == "silhouette_score"
    ), "Default search metric was not silhouette_score"
    assert cl.scores is not None, "Widget is missing cluster search scores"
    assert cl.search, "Widget is missing boolean attribute 'search' for cluster search"
    assert isinstance(cl.show(), mpl_plot)
    assert hasattr(cl, "cluster_search_plot")
    assert isinstance(cl.cluster_search_plot(), mpl_plot)
    assert hasattr(cl, "reductor")


@pytest.mark.base
def test_kmeans_plotly(kmeans_default):
    figure = kmeans_default.show(viz_backend="plotly")
    assert isinstance(figure, plotly.graph_objs._figure.Figure)


@pytest.fixture
def monkeypatch_KMeans(monkeypatch):
    class mock_kmeans(KMeans):
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def predict(self, X):
            return np.ones(X.shape[0])

    monkeypatch.setattr("sklearn.cluster.KMeans", mock_kmeans)


@pytest.fixture
def monkeypatch_HDBSCAN(_hdbscan, monkeypatch):
    class mock_hdbscan(_hdbscan.HDBSCAN):
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, X, **kwargs):
            return np.ones(X.shape[0])

    monkeypatch.setattr("hdbscan.HDBSCAN", mock_hdbscan)


@pytest.mark.base
def test_pandas_compute_cluster(numeric_data, monkeypatch_KMeans):
    widget = _pandas_compute_cluster(numeric_data, method="kmeans")
    assert isinstance(
        widget.clusters, np.ndarray
    ), "Cluster labels was not a numpy array"
    assert isinstance(
        widget, KmeansClusterWidget
    ), "Fit object was not a KmeansClusterWidget"
    assert hasattr(widget, "scaler"), "Missing sklearn StandardScaler"
    assert isinstance(
        widget.scaler, StandardScaler
    ), "Scaler attribute is not an instance of sklearn StandardScaler"
    assert hasattr(widget, "input_data"), "Missing input data"
    assert hasattr(widget, "scaled_data"), "Missing scaled data"


def test_pandas_compute_cluster_hdbscan(_hdbscan, numeric_data, monkeypatch_HDBSCAN):
    widget = _pandas_compute_cluster(numeric_data, method="hdbscan")
    assert isinstance(
        widget.clusters, np.ndarray
    ), "Cluster labels was not a numpy array"
    assert isinstance(
        widget, HDBSCANClusterWidget
    ), "Fit object was not a HDBSCANClusterWidget"
    assert hasattr(widget, "scaler"), "Missing sklearn StandardScaler"
    assert isinstance(
        widget.scaler, StandardScaler
    ), "Scaler attribute is not an instance of sklearn StandardScaler"
    assert hasattr(widget, "input_data"), "Missing input data"
    assert hasattr(widget, "scaled_data"), "Missing scaled data"


@pytest.mark.base
def test_pandas_compute_cluster_invalid_method(numeric_data):
    with pytest.raises(ValueError):
        _pandas_compute_cluster(numeric_data, method="unimplemented")


@pytest.mark.base
def test_pandas_run_kmeans_default(numeric_data, monkeypatch_KMeans):
    widget = _run_kmeans(numeric_data)
    assert isinstance(
        widget.clusters, np.ndarray
    ), "Cluster labels was not a numpy array"
    assert isinstance(
        widget, KmeansClusterWidget
    ), "Fit object was not a KmeansClusterWidget"
    assert hasattr(widget, "n_clusters"), "Missing `n_clusters` attribute"
    assert (
        widget.search
    ), "Cluster search did not occur when n_clusters is None (default)"


@pytest.mark.base
def test_pandas_run_kmeans_specified_cluster(numeric_data, monkeypatch_KMeans):
    widget = _run_kmeans(numeric_data, n_clusters=2)
    assert isinstance(
        widget.clusters, np.ndarray
    ), "Cluster labels was not a numpy array"
    assert isinstance(
        widget, KmeansClusterWidget
    ), "Fit object was not a KmeansClusterWidget"
    assert (
        widget.n_clusters == 2
    ), "n_clusters on the widget does not match expected value"
    assert not widget.search, "`search` equals True although n_cluster is specified"


@pytest.mark.base
def test_pandas_find_clusters_default(numeric_data, monkeypatch_KMeans):
    widget = _find_clusters(numeric_data)
    assert isinstance(widget.clusters, np.ndarray)
    assert isinstance(widget, KmeansClusterWidget)
    assert widget.cluster_range == (2, 20)
    assert widget.search, "`search` attribute was not true on widget for cluster search"
    assert (
        widget.metric == "silhouette_score"
    ), "`metric` did not default to `silhouette_score`"
    assert isinstance(widget.scores, list), "metric `scores` on widget is not a list"


@pytest.mark.base
def test_pandas_find_clusters_param(numeric_data, monkeypatch_KMeans):
    widget = _find_clusters(numeric_data, cluster_range=(2, 4))
    assert widget.cluster_range == (2, 4)
    with pytest.raises(ValueError):
        _find_clusters(numeric_data, metric="adjusted_rand_score")
    widget = _find_clusters(
        numeric_data, metric="adjusted_rand_score", target=numeric_data.iloc[:, 0]
    )
    assert hasattr(
        widget, "target"
    ), "Supervised clustering did not record `target` on widget"


@pytest.mark.base
def test_pandas_fit_kmeans(numeric_data, monkeypatch_KMeans):
    widget = _fit_kmeans(numeric_data, 2)
    assert isinstance(
        widget.clusters, np.ndarray
    ), "Cluster labels was not a numpy array"
    assert isinstance(
        widget, KmeansClusterWidget
    ), "Fit object was not a KmeansClusterWidget"
    assert (
        widget.n_clusters == 2
    ), "n_clusters on the widget does not match expected value"
    assert isinstance(widget.estimator, KMeans), "Estimator is not KMeans"


def test_pandas_run_hdbscan_default(_hdbscan, numeric_data, monkeypatch_HDBSCAN):
    widget = _run_hdbscan(numeric_data, min_cluster_size=10)
    assert isinstance(
        widget.clusters, np.ndarray
    ), "Cluster labels was not a numpy array"
    assert isinstance(
        widget, HDBSCANClusterWidget
    ), "Fit object was not a HDBSCANClusterWidget"
    assert (
        widget.n_clusters == 1
    ), "n_clusters on the widget does not match expected value"
    print(repr(widget))
    assert isinstance(widget.estimator, _hdbscan.HDBSCAN), "Estimator is not HDBSCAN"
