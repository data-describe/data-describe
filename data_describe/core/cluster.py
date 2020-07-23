# import hdbscan
# import sklearn
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from numpy import count_nonzero  # type: ignore
# from sklearn.cluster import KMeans
from abc import ABC

from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends import _get_viz_backend, _get_compute_backend
from data_describe.dimensionality_reduction.dimensionality_reduction import dim_reduc

# from data_describe.dimensionality_reduction.dimensionality_reduction import dim_reduc


def cluster(
    data,
    method="kmeans",
    target=None,
    dim_method="pca",
    elbow=False,
    compute_backend=None,
    viz_backend=None,
    **kwargs,
):
    """ Creates cluster visualization

    Args:
        df: Pandas data frame
        method: Clustering method: kmeans or hdbscan
        interactive: If False, creates a seaborn plot
                    If True, create plotly interactive plot
        elbow: If true, create an elbow plot for the optimal number of clusters
        dim_method: Select method to reduce the data to two dimensions for visualization
                    Note: Only pca, tsne, and tsvd are supported
        analysis: Metric to choose the optimal number of clusters (metrics from sklearn.metrics)
                    Includes: silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,
                    completeness_score, v_measure_score, homogeneity_completeness_v_measure, fowlkes_mallows_score,
                    davies_bouldin_score
        return_value: Specifies the object that will be returned by the function. (None=Emit the plot, plot=Return
                    plotly object if interactive is selected, data=Return the original data with cluster labels,
                    reduc=Return the 2-dimensional data with cluster labels)

        kwargs: Key word arguments for clustering methods


    Returns:
        viz: Seaborn scatter plot or Plotly scatter plot
        reduc_df: reduced data frame with cluster labels
        df: original data frame with cluster labels
        elbow_plot: elbow plot if elbow parameter is True

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

    return _get_viz_backend(viz_backend).viz_cluster(
        xy_data, method, **kwargs
    )


class ClusterFit(ABC):
    """Interface for collecting additional information about the clustering"""

    def __init__(self, clusters=None, method=None):
        """Mandatory parameters

        Args:
            clusters: The predicted clusters
            method: The clustering algorithm
        """
        self.clusters = clusters
        self.method = method

    def __repr__(self):
        return "Cluster fit information"


class KmeansFit(ClusterFit):
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
        self.clusters = clusters
        self.method = method
        self.estimator = estimator
        self.n_clusters = n_clusters
        self.search = search
        self.cluster_range = cluster_range
        self.metric = metric

    def elbow_plot(self, viz_backend=None, **kwargs):
        if not self.search:
            raise ValueError(
                "Elbow plot is not applicable when n_cluster is explicitly selected"
            )

        return _get_viz_backend(viz_backend).viz_elbow_plot(
            self.cluster_range, self.scores, self.metric ** kwargs
        )


class HDBSCANFit(ClusterFit):
    def __init__(
        self, clusters=None, method=None, estimator=None,
    ):
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
