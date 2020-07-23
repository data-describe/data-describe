from typing import Tuple, List, Union

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from data_describe.config._config import get_option


def viz_cluster(data, method: str, xlabel: str = None, ylabel: str = None, **kwargs):
    """Visualize clusters using Seaborn.

    Args:
        data (DataFrame): The data
        method (str): The clustering method, to be used as the plot title
        xlabel (str, optional): The x-axis label. Defaults to "Reduced Dimension 1".
        ylabel (str, optional): The y-axis label. Defaults to "Reduced Dimension 2".

    Returns:
        Seaborn plot
    """
    xlabel = xlabel or "Reduced Dimension 1"
    ylabel = ylabel or "Reduced Dimension 2"
    plt.figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    unique_labels = len(np.unique(data["clusters"]))
    pal = sns.set_palette("tab10", n_colors=unique_labels + 1)
    ax = sns.scatterplot(
        data=data, x="x", y="y", hue="clusters", palette=pal, legend="brief",
    )
    sns.set_context("talk")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right", ncol=1)
    plt.title(method + " Cluster")
    return ax


def viz_elbow_plot(
    cluster_range: Tuple[int, int],
    scores: List[Union[int, float]],
    metric: str,
    **kwargs
):
    """Visualize the elbow plot for K-means clusters.

    Args:
        cluster_range (Tuple[int, int]): The range of n_clusters (k) searched as (min_cluster, max_cluster)
        scores (List[Union[int, float]]): The scores from the evaluation metric used to determine the "optimal" n_clusters
        metric (str): The evaluation metric used

    Returns:
        Seaborn plot
    """
    n_clusters = list(range(cluster_range))
    plt.figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    ax = sns.lineplot(n_clusters, scores)
    ax.set_title("Optimal Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average {}".format(" ".join(metric.split("_"))))
    return ax
