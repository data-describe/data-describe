import pytest
import matplotlib
from matplotlib.axes import Axes as mpl_plot
from sklearn.cluster import KMeans

import data_describe as dd
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.core.cluster import (
    ClusterWidget,
    KmeansClusterWidget,
    HDBSCANClusterWidget,
)


matplotlib.use("Agg")


def test_not_df():
    with pytest.raises(ValueError):
        dd.cluster("this_is_a_string")


def test_cluster_widget():
    cl = ClusterWidget()
    assert hasattr(cl, "clusters"), "Cluster Widget missing cluster labels"
    assert hasattr(cl, "viz_backend"), "Cluster Widget missing default viz backend"
    assert hasattr(cl, "__repr__"), "Cluster Widget missing __repr__ method"
    assert hasattr(cl, "_repr_html_"), "Cluster Widget missing _repr_html_ method"
    assert hasattr(cl, "show"), "Cluster Widget missing show method"


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


def test_hdbscan_cluster_widget():
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


def test_kmeans_default(numeric_data):
    cl = dd.cluster(numeric_data, method="kmeans")
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
    assert cl.n_clusters == 19, "Expected number of clusters found to be 3"
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


def test_kmeans_specify_cluster(numeric_data):
    pass


def test_kmeans_search_metrics(numeric_data):
    pass


def test_kmeans_cluster_range(numeric_data):
    pass


def test_kmeans_supervised(numeric_data):
    pass


def test_hdbscan_default(numeric_data):
    pass


def test_hdbscan_with_kwargs(numeric_data):
    pass
