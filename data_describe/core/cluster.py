# import hdbscan
# import sklearn
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from numpy import count_nonzero  # type: ignore
# from sklearn.cluster import KMeans
from abc import ABC
from typing import List

from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends import _get_viz_backend, _get_compute_backend
from data_describe.dimensionality_reduction.dimensionality_reduction import dim_reduc

# from data_describe.dimensionality_reduction.dimensionality_reduction import dim_reduc


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
        Cluster plot
    """
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Data frame required")

    if method not in ["kmeans", "hdbscan"]:
        raise ValueError(f"{method} not implemented")

    data = data.select_dtypes("number")

    clusters, fit = _get_compute_backend(compute_backend, data).compute_cluster(
        data=data, method=method, **kwargs
    )

    xy_data, _ = dim_reduc(data, 2, dim_method=dim_method)
    xy_data.columns = ["x", "y"]
    xy_data["clusters"] = clusters

    # TODO (haishiro): Set x/y labels with explained variance if using PCA
    # x
    # + " "
    # + "({}% variance explained)".format(
    #     str(round(truncator.explained_variance_ratio_[0] * 100, 2))
    # )

    return _get_viz_backend(viz_backend).viz_cluster(xy_data, method, **kwargs)


class ClusterFit(ABC):
    """Interface for collecting additional information about the clustering."""

    def __init__(self, clusters: List[int] = None, method: str = None):
        """Mandatory parameters.

        Args:
            clusters (List[int], optional): The predicted cluster labels.
            method (str, optional): {'kmeans', 'hdbscan'}. The specified cluster method.
        """
        self.clusters = clusters
        self.method = method

    def __repr__(self):
        return "Cluster fit information"


class KmeansFit(ClusterFit):
    """Interface for collecting additional information about the k-Means clustering."""

    def __init__(
        self,
        clusters=None,
        method=None,
        estimator=None,
        n_clusters=None,
        search=False,
        cluster_range=None,
        metric=None,
    ):
        """Mandatory parameters.

        Args:
            clusters (List[int], optional): The predicted cluster labels.
            method (str, optional): The specified cluster method, "kmeans".
            estimator (optional): The cluster estimator object.
            n_clusters (int, optional): The number of clusters (k) used in the final clustering fit.
            search (bool, optional): If True, a search was performed for optimal n_clusters.
            cluster_range (Tuple[int, int], optional): The range of clusters searched as (min_cluster, max_cluster).
            metric (str, optional): The metric used to evaluate the cluster search.
        """
        self.clusters = clusters
        self.method = method
        self.estimator = estimator
        self.n_clusters = n_clusters
        self.search = search
        self.cluster_range = cluster_range
        self.metric = metric

    def elbow_plot(self, viz_backend=None, **kwargs):
        """Elbow plot."""
        if not self.search:
            raise ValueError(
                "Elbow plot is not applicable when n_cluster is explicitly selected"
            )

        return _get_viz_backend(viz_backend).viz_elbow_plot(
            self.cluster_range, self.scores, self.metric ** kwargs
        )


class HDBSCANFit(ClusterFit):
    """Interface for collecting additional information about the HDBSCAN clustering."""

    def __init__(
        self, clusters: List[int] = None, method: str = None, estimator=None,
    ):
        """Mandatory parameters.

        Args:
            clusters (List[int], optional): The predicted cluster labels.
            method (str, optional): The specified cluster method, "hdbscan".
            estimator (optional): The HDBSCAN estimator object.
        """
        self.clusters = clusters
        self.method = method
        self.estimator = estimator

    # if return_value is None:
    #     try:
    #         return iplot(viz)
    #     except Exception:  # TODO: This should be handled by backend routing
    #         return viz
    # elif return_value == "plot":
    #     return viz


#     if interactive is False:
#         return (
#             static_plot(
#                 data=reduc_df,
#                 method="KMeans",
#                 x=reduc_df.columns[0],
#                 y=reduc_df.columns[1],
#                 truncator=truncator,
#             ),
#             reduc_df,
#         )
#     else:
#         return (
#             interactive_plot(
#                 df=reduc_df,
#                 method="KMeans",
#                 x=reduc_df.columns[0],
#                 y=reduc_df.columns[1],
#                 color="cluster",
#                 truncator=truncator,
#             ),
#             reduc_df,
#         )


# TODO (haishiro): Move to dimensionality_reduction
# def truncate_data(data):
#     """ Reduces the number of dimensions for t-SNE to speed up computation time and reduce noise

#         Args:
#             data: Pandas data frame
#         Returns:
#             reduc_df: reduced data frame
#             truncator: Instance of a dimensionality reduction method
#     """
#     if data.shape[1] > 50:
#         data = data.to_numpy()
#         sparsity = 1.0 - (count_nonzero(data) / float(data.size))
#         if sparsity >= 0.5:
#             reduc, truncator = dim_reduc(data, n_components=50, dim_method="tsvd")
#         else:
#             reduc, truncator = dim_reduc(data, n_components=50, dim_method="pca")
#         reduc_df, truncator = dim_reduc(reduc, n_components=2, dim_method="tsne")
#         return reduc_df, truncator
#     reduc_df, truncator = dim_reduc(data, n_components=2, dim_method="tsne")
#     return reduc_df, truncator


# def interactive_plot(df, x, y, method, color, truncator=None):
#     """ Creates interactive scatter plot using plotly

#     Args:
#         df: Pandas data frame
#         x: x-axis column
#         y: y-axis column
#         method: Method for creating a title for the plot
#         color: Feature from df that determines color for data points
#         truncator: Instance of a dimensionality reduction method

#     Returns:
#         fig: Plotly scatter plot or plotly object
#     """
#
