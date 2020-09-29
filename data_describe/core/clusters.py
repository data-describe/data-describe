from typing import List
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_describe._widget import BaseWidget
from data_describe.compat import _DATAFRAME_TYPE, _compat, requires
from data_describe.backends import _get_viz_backend, _get_compute_backend
from data_describe.dimensionality_reduction.dimensionality_reduction import dim_reduc


def cluster(
    data,
    method="kmeans",
    dim_method="pca",
    compute_backend=None,
    viz_backend=None,
    **kwargs,
):
    """Cluster analysis.

    Performs clustering on the data and visualizes the results on a 2-d plot.

    Args:
        data (DataFrame): The data.
        method (str, optional): {'kmeans', 'hdbscan'} The clustering method. Defaults to "kmeans".
        dim_method (str, optional): The method to use for dimensionality reduction. Defaults to "pca".
        compute_backend (str, optional): The compute backend.
        viz_backend (str, optional): The visualization backend.

    Raises:
        ValueError: Data frame required
        ValueError: Clustering method not implemented

    Returns:
        ClusterWidget
    """
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Data frame required")

    if method not in ["kmeans", "hdbscan"]:
        raise ValueError(f"{method} not implemented")

    data = data.select_dtypes("number")

    clusterwidget = _get_compute_backend(compute_backend, data).compute_cluster(
        data=data, method=method, **kwargs
    )

    viz_data, reductor = dim_reduc(clusterwidget.scaled_data, 2, dim_method=dim_method)
    viz_data.columns = ["x", "y"]
    viz_data["clusters"] = clusterwidget.clusters

    clusterwidget.viz_data = viz_data
    clusterwidget.reductor = reductor

    if dim_method == "pca":
        var_explained = np.round(reductor.explained_variance_ratio_[:2], 2) * 100
        clusterwidget.xlabel = f"Component 1 ({var_explained[0]}% variance explained)"
        clusterwidget.ylabel = f"Component 2 ({var_explained[1]}% variance explained)"
    else:
        clusterwidget.xlabel = "Dimension 1"
        clusterwidget.ylabel = "Dimension 2"

    clusterwidget.viz_backend = viz_backend

    return clusterwidget


class ClusterWidget(BaseWidget):
    """Interface for collecting additional information about the clustering."""

    def __init__(
        self,
        clusters: List[int] = None,
        method: str = None,
        estimator=None,
        input_data=None,
        scaled_data=None,
        viz_data=None,
        dim_method: str = None,
        reductor=None,
        xlabel: str = None,
        ylabel: str = None,
        **kwargs,
    ):
        """Cluster Analysis.

        Args:
            clusters (List[int], optional): The predicted cluster labels
            method (str): The clustering algorithm
            estimator: The clustering estimator
            input_data: The input data
            scaled_data: The data after applying standardization
            viz_data: The data used for the default visualization i.e. reduced to 2 dimensions
            dim_method (str): The algorithm used for dimensionality reduction
            reductor: The dimensionality reduction estimator
            xlabel (str): The x-axis label for the cluster plot
            ylabel (str): The y-axis label for the cluster plot
        """
        super(ClusterWidget, self).__init__(**kwargs)
        self.clusters = clusters
        self.method = method
        self.estimator = estimator
        self.input_data = input_data
        self.scaled_data = scaled_data
        self.viz_data = viz_data
        self.dim_method = dim_method
        self.reductor = reductor
        self.xlabel = xlabel
        self.ylabel = ylabel

    def __str__(self):
        return "data-describe Cluster Widget"

    def show(self, viz_backend=None, **kwargs):
        """Show the cluster plot."""
        backend = viz_backend or self.viz_backend

        if self.viz_data is None:
            raise ValueError("Could not find data to visualize.")

        return _get_viz_backend(backend).viz_cluster(
            self.viz_data,
            method=self.method,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            **kwargs,
        )


class KmeansClusterWidget(ClusterWidget):
    """Interface for collecting additional information about the k-Means clustering."""

    def __init__(
        self,
        n_clusters=None,
        search=False,
        cluster_range=None,
        metric=None,
        scores=None,
        **kwargs,
    ):
        """Mandatory parameters.

        Args:
            clusters (List[int], optional): The predicted cluster labels.
            estimator (optional): The cluster estimator object.
            n_clusters (int, optional): The number of clusters (k) used in the final clustering fit.
            search (bool, optional): If True, a search was performed for optimal n_clusters.
            cluster_range (Tuple[int, int], optional): The range of clusters searched as (min_cluster, max_cluster).
            metric (str, optional): The metric used to evaluate the cluster search.
            scores: The metric scores in cluster search.
        """
        super(KmeansClusterWidget, self).__init__(**kwargs)
        self.method = "kmeans"
        self.n_clusters = n_clusters
        self.search = search
        self.cluster_range = cluster_range
        self.metric = metric
        self.scores = scores

    def cluster_search_plot(self, viz_backend=None, **kwargs):
        """Shows the results of cluster search.

        Cluster search attempts to find an optimal n_clusters by maximizing on some criterion.
        This plot shows a line plot of each n_cluster that was attempted and its score.

        Args:
            viz_backend: The visualization backend.
            **kwargs: Additional keyword arguments to pass to the visualization backend.

        Returns:
            The plot
        """
        if not self.search:
            raise ValueError(
                "Cluster search plot is not applicable when n_cluster is explicitly selected"
            )

        return _get_viz_backend(viz_backend).viz_cluster_search_plot(
            self.cluster_range, self.scores, self.metric, **kwargs
        )


class HDBSCANClusterWidget(ClusterWidget):
    """Interface for collecting additional information about the HDBSCAN clustering."""

    def __init__(self, **kwargs):
        """Mandatory parameters.

        Args:
            clusters (List[int], optional): The predicted cluster labels.
            estimator (optional): The HDBSCAN estimator object.
        """
        super(HDBSCANClusterWidget, self).__init__(**kwargs)
        self.method = "hdbscan"


def _pandas_compute_cluster(data, method: str, **kwargs):
    """Backend implementation of cluster.

    Args:
        data (DataFrame): The data
        method (str): {"kmeans", "hdbscan} The clustering algorithm

    Raises:
        NotImplementedError: If method is not implemented

    Returns:
        (clusters, ClusterFit)

        clusters: The predicted cluster labels
        ClusterFit: A class containing additional information about the fit
    """
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    if method == "kmeans":
        clusterwidget = _run_kmeans(scaled_data, **kwargs)
    elif method == "hdbscan":
        clusterwidget = _run_hdbscan(scaled_data)
    else:
        raise ValueError(f"{method} not implemented")

    clusterwidget.scaler = scaler
    clusterwidget.input_data = data
    clusterwidget.scaled_data = scaled_data

    return clusterwidget


def _run_kmeans(
    data,
    n_clusters: Optional[int] = None,
    cluster_range: Tuple[int, int] = None,
    metric: str = "silhouette_score",
    target=None,
    **kwargs,
):
    """Run K-Means clustering.

    Args:
        data (DataFrame): The data.
        n_clusters (Optional[int], optional): The number of clusters.
        cluster_range (Tuple[int, int], optional): A tuple of the minimum and
        maximum cluster search range. Defaults to (2, 20).
        metric (str, optional): The metric to optimize (from sklearn.metrics).
        target: (For supervised clustering) The labels, as a 1-D array.
        **kwargs: Keyword arguments to be passed into the K-Means estimator.

    Returns:
        [type]: [description]
    """
    if n_clusters is None:
        clusterwidget = _find_clusters(
            data=data,
            cluster_range=cluster_range,
            metric=metric,
            target=target,
            **kwargs,
        )
    else:
        clusterwidget = _fit_kmeans(data, n_clusters, **kwargs)

    return clusterwidget


def _find_clusters(
    data,
    cluster_range: Tuple[int, int] = None,
    metric: str = "silhouette_score",
    target=None,
    **kwargs,
):
    """Finds the optimal number of clusters for K-Means clustering using the selected metric.

    Args:
        data: The data.
        cluster_range: A tuple of the minimum and
        maximum cluster search range. Defaults to (2, 20).
        metric: The metric to optimize (from sklearn.metrics).
        target: (For supervised clustering) The labels, as a 1-D array.
        **kwargs: Keyword arguments to be passed into the K-Means estimator.

    Returns:
        clusters, KmeansFit
    """
    cluster_range = cluster_range or (2, 20)
    if not cluster_range[0] < cluster_range[1]:
        raise ValueError(
            "cluster_range expected to be (min_cluster, max_cluster), but the min was >= the max"
        )
    unsupervised_metrics = [
        "silhouette_score",
        "davies_bouldin_score",
        "calinski_harabasz_score",
    ]

    scores = []
    widgets = []
    for n in range(*cluster_range):
        clusterwidget = _fit_kmeans(data, n, **kwargs)
        analysis_func = getattr(sklearn.metrics, metric)
        if metric in unsupervised_metrics:
            score = analysis_func(data, clusterwidget.clusters)
        else:
            if target is None:
                raise ValueError("'target' must be specified for supervised clustering")
            score = analysis_func(target, clusterwidget.clusters)
        scores.append(score)
        widgets.append(clusterwidget)

    best_idx = np.argmax(scores)
    clusterwidget = widgets[best_idx]
    clusterwidget.search = True
    clusterwidget.cluster_range = cluster_range
    clusterwidget.metric = metric
    clusterwidget.scores = scores
    if target is not None:
        clusterwidget.target = target

    return clusterwidget


def _fit_kmeans(data, n_clusters, **kwargs):
    """Fits the K-Means estimator.

    Args:
        data: Data frame
        n_clusters: Number of clusters for K-means
        **kwargs: Keyword arguments to be passed into the K-Means estimator

    Returns:
        clusters, KmeansFit
    """
    default_kmeans_kwargs = {"random_state": 0, "n_clusters": n_clusters}
    kmeans_kwargs = {**default_kmeans_kwargs, **(kwargs or {})}
    kmeans = KMeans(**kmeans_kwargs)
    kmeans.fit(data)
    cluster_labels = kmeans.predict(data)

    clusterwidget = KmeansClusterWidget(
        clusters=cluster_labels, estimator=kmeans, n_clusters=n_clusters
    )
    return clusterwidget


@requires("hdbscan")
def _run_hdbscan(data, min_cluster_size=15, **kwargs):
    """Run HDBSCAN clustering.

    Args:
        data (DataFrame): The data
        min_cluster_size (int, optional): Minimum cluster size. Defaults to 15.
        **kwargs: Additional keyword arguments to be passed to HDBSCAN

    Returns:
        clusters, HDBSCANFit
    """
    default_hdbscan_kwargs = {"min_cluster_size": min_cluster_size}
    hdbscan_kwargs = {**default_hdbscan_kwargs, **(kwargs or {})}
    hdb = _compat["hdbscan"].HDBSCAN(**hdbscan_kwargs)
    clusters = hdb.fit_predict(data)
    clusterwidget = HDBSCANClusterWidget(
        clusters=clusters, method="hdbscan", estimator=hdb
    )
    clusterwidget.n_clusters = len(np.unique(clusters))
    return clusterwidget
