from abc import ABC
from typing import List

from IPython.display import display

from data_describe.compat import _DATAFRAME_TYPE
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

    clusters, fit = _get_compute_backend(compute_backend, data).compute_cluster(
        data=data, method=method, **kwargs
    )

    viz_data, reductor = dim_reduc(fit.scaled_data, 2, dim_method=dim_method)
    viz_data.columns = ["x", "y"]
    viz_data["clusters"] = clusters

    fit.viz_data = viz_data
    fit.reductor = reductor

    # TODO (haishiro): Set x/y labels with explained variance if using PCA
    # x
    # + " "
    # + "({}% variance explained)".format(
    #     str(round(truncator.explained_variance_ratio_[0] * 100, 2))
    # )

    fit.viz_backend = viz_backend

    return fit


class ClusterWidget(ABC):
    """Interface for collecting additional information about the clustering."""

    def __init__(self, clusters: List[int] = None, **kwargs):
        """Mandatory parameters.

        Args:
            clusters (List[int], optional): The predicted cluster labels.
        """
        self.viz_backend = None
        self.clusters = clusters
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __repr__(self):
        return "Data Describe Cluster Widget"

    def _repr_html_(self):
        return display(self.show())

    def show(self, viz_backend=None):
        """Show the default output."""
        try:
            viz_data = self.viz_data
        except AttributeError:
            raise ValueError(
                "Could not find the expected data to visualize on this widget"
            )

        try:
            backend = self.viz_backend
        except AttributeError:
            backend = viz_backend

        return _get_viz_backend(backend).viz_cluster(viz_data, method=self.method)


class KmeansClusterWidget(ClusterWidget):
    """Interface for collecting additional information about the k-Means clustering."""

    def __init__(
        self,
        clusters=None,
        estimator=None,
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
        super(ClusterWidget, self).__init__(**kwargs)
        self.clusters = clusters
        self.method = "kmeans"
        self.estimator = estimator
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

    def __init__(self, clusters: List[int] = None, estimator=None, **kwargs):
        """Mandatory parameters.

        Args:
            clusters (List[int], optional): The predicted cluster labels.
            estimator (optional): The HDBSCAN estimator object.
        """
        super(ClusterWidget, self).__init__(**kwargs)
        self.clusters = clusters
        self.method = "hdbscan"
        self.estimator = estimator
