from typing import Optional, Tuple

import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_describe import _compat
from data_describe.compat import requires
import data_describe.core.clusters as ddcluster


def compute_cluster(data, method: str, **kwargs):
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

    clusterwidget = ddcluster.KmeansClusterWidget(
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
    clusterwidget = ddcluster.HDBSCANClusterWidget(
        clusters=clusters, method="hdbscan", estimator=hdb
    )
    clusterwidget.n_clusters = len(np.unique(clusters))
    return clusterwidget
