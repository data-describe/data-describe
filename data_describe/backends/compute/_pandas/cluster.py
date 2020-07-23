from typing import Optional, Tuple

import numpy as np
import sklearn
from sklearn.cluster import KMeans

from data_describe import compat
from data_describe.compat import requires
import data_describe.core.cluster as ddc


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
    data = data.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    if method == "kmeans":
        clusters, fit = _run_kmeans(data, **kwargs)
    elif method == "hdbscan":
        clusters, fit = _run_hdbscan(data)
    else:
        raise NotImplementedError(f"{method} not implemented")

    return clusters, fit


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
        data: Data frame
        n_clusters: (Optional) Number of clusters (k)
        metric: The metric to optimize (from sklearn.metrics)
            Includes: silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,
            completeness_score, v_measure_score, homogeneity_completeness_v_measure, fowlkes_mallows_score,
            davies_bouldin_score
        target: (For supervised clustering) The labels, as a 1-D array
        **kwargs: Keyword arguments to be passed to the specific clustering method

    Returns:
        (clusters, KMeansFit)

        clusters: The predicted cluster labels
        KMeansFit: A class containing additional information about the fit
    """
    if n_clusters is None:
        clusters, fit = _find_clusters(
            data=data,
            cluster_range=cluster_range,
            metric=metric,
            target=target,
            **kwargs,
        )
    else:
        clusters, fit = _fit_kmeans(data, n_clusters, **kwargs)

    return clusters, fit


def _find_clusters(
    data,
    cluster_range: Tuple[int, int] = None,
    metric: str = "silhouette_score",
    target=None,
    **kwargs,
):
    """Finds the optimal number of clusters for K-Means clustering using the selected metric.

    Args:
        data: Data frame
        cluster_range: A tuple of the minimum and maximum cluster search range. Defaults to (2, 20)
        metric: The metric to optimize (from sklearn.metrics)
            Includes: silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,
            completeness_score, v_measure_score, homogeneity_completeness_v_measure, fowlkes_mallows_score,
            davies_bouldin_score
        target: (For supervised clustering) The labels, as a 1-D array
        **kwargs: Keyword arguments to be passed into the K-Means estimator

    Returns:
        clusters, KmeansFit
    """
    cluster_range = cluster_range or (2, 20)
    unsupervised_metrics = ["silhouette_score", "davies_bouldin_score"]

    scores = []
    fits = []
    for n in range(*cluster_range):
        clusters, fit = _fit_kmeans(data, n, **kwargs)
        analysis_func = getattr(sklearn.metrics, metric)
        if metric in unsupervised_metrics:
            score = analysis_func(data, clusters)
        else:
            if target is None:
                raise ValueError("'target' must be specified for supervised")
            score = analysis_func(target, clusters)
        scores.append(score)
        fits.append(fit)

    best_idx = np.argmax(scores)
    fit = fits[best_idx]
    fit.search = True
    fit.cluster_range = cluster_range
    fit.metric = metric
    fit.scores = scores

    return fit.clusters, fit


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

    fit = ddc.KmeansFit(
        clusters=cluster_labels, estimator=kmeans, n_clusters=n_clusters
    )
    return cluster_labels, fit


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
    hdb = compat.hdbscan.HDBSCAN(**hdbscan_kwargs)
    clusters = hdb.fit_predict(data)
    fit = ddc.HDBSCANFit(clusters, method="hdbscan", estimator=hdb)
    return clusters, fit
