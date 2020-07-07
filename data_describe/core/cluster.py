import hdbscan
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from numpy import count_nonzero  # type: ignore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from plotly.offline import init_notebook_mode, iplot
from IPython import get_ipython

from data_describe.dimensionality_reduction.dimensionality_reduction import dim_reduc


def cluster(
    df,
    method="KMeans",
    target=None,
    interactive=True,
    elbow=False,
    dim_method="pca",
    analysis="silhouette_score",
    return_value=None,
    kwargs=None,
):
    """ Creates cluster visualization

    Args:
        df: Pandas data frame
        method: Clustering method: KMeans or HDBSCAN
        target: Name of the column that contains the response
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
    if isinstance(df, pd.DataFrame):
        num_df = df.select_dtypes(["number"]).astype("float64")
    else:
        raise NotImplementedError("Pandas data frame required")

    if method not in ["KMeans", "HDBSCAN"]:
        raise ValueError("{} model is not supported".format(method))

    scaler = StandardScaler()
    data = pd.DataFrame(
        scaler.fit_transform(num_df), index=num_df.index, columns=num_df.columns
    ).dropna()

    if target is not None:
        target_df = data.drop([target], axis=1)
    else:
        target_df = data

    if dim_method == "tsne":
        reduc_df, truncator = truncate_data(data=target_df)
    else:
        reduc_df, truncator = dim_reduc(
            data=target_df, n_components=2, dim_method=dim_method
        )

    if method == "KMeans":
        viz, reduc_df = kmeans_cluster(
            data=data,
            reduc_df=reduc_df,
            target=target,
            analysis=analysis,
            interactive=interactive,
            elbow=elbow,
            truncator=truncator,
            kwargs=kwargs,
        )
        df["cluster"] = reduc_df["cluster"]

    elif method == "HDBSCAN":
        viz, reduc_df = hdbscan_cluster(
            data=data,
            reduc_df=reduc_df,
            interactive=interactive,
            truncator=truncator,
            kwargs=kwargs,
        )
        df["cluster"] = reduc_df["cluster"]

    if return_value is None:
        try:
            return iplot(viz)
        except Exception:  # TODO: This should be handled by backend routing
            return viz
    elif return_value == "plot":
        return viz
    elif return_value == "data":
        return df
    elif return_value == "reduc":
        return reduc_df
    else:
        raise ValueError("{} is not supported".format(return_value))


def kmeans_cluster(
    data,
    reduc_df,
    analysis="silhouette_score",
    n_clusters=None,
    cluster_min=2,
    cluster_max=20,
    interactive=True,
    truncator=None,
    target=None,
    elbow=False,
    kwargs=None,
):
    """Function to create K-Means clustering visualization

    Args:
        data: Pandas data frame
        reduc_df: Reduced data frame
        analysis: Metric to choose the optimal number of clusters (metrics from sklearn.metrics)
                Includes: silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,
                completeness_score, v_measure_score, homogeneity_completeness_v_measure, fowlkes_mallows_score,
                davies_bouldin_score
        n_clusters: Number of clusters
        cluster_min: Minimum number of clusters to be generated
        cluster_max: Maximum number of clusters to be generated
        interactive: If False, creates a seaborn plot
                    If True, create plotly interactive plot
        truncator: Instance of a dimensionality reduction method
        target: Name of the column that contains the response
        elbow: If true, create an elbow plot for the optimal number of clusters
        kwargs: Key word arguments to be passed into K-Means cluster

    Returns:
        Seaborn plot or Plotly interactive scatter plot
        elbow_plot: elbow plot
        reduc_df: reduced data frame

    """
    if n_clusters is None:
        n_clusters, cluster_range, scores = find_clusters(
            data=data,
            cluster_min=cluster_min,
            cluster_max=cluster_max,
            analysis=analysis,
            target=target,
        )

    labels, kmeans_model = apply_kmeans(data, n_clusters, kwargs)
    reduc_df["cluster"] = pd.Series(labels).astype("str")
    if elbow is True:
        # plt.figure(figsize=(context.fig_width.fig_height)) # TODO (haishiro): Replace with get_option
        elbow_plot = sns.lineplot(cluster_range, scores)
        elbow_plot.set_title("Optimal Number of Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Average {}".format(" ".join(analysis.split("_"))))
        plt.show()

    if interactive is False:
        return (
            static_plot(
                data=reduc_df,
                method="KMeans",
                x=reduc_df.columns[0],
                y=reduc_df.columns[1],
                truncator=truncator,
            ),
            reduc_df,
        )
    else:
        return (
            interactive_plot(
                df=reduc_df,
                method="KMeans",
                x=reduc_df.columns[0],
                y=reduc_df.columns[1],
                color="cluster",
                truncator=truncator,
            ),
            reduc_df,
        )


def find_clusters(
    data,
    cluster_min,
    cluster_max,
    analysis="silhouette_score",
    target=None,
    kwargs=None,
):
    """Finds the optimal number of clusters for K-Means clustering using the selected analysis

    Args:
        data: Pandas data frame
        cluster_min: Minimum number of clusters to be generated
        cluster_max: Maximum number of clusters to be generated
        analysis: Metric to choose the optimal number of clusters (metrics from sklearn.metrics)
                Includes: silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,
                completeness_score, v_measure_score, homogeneity_completeness_v_measure, fowlkes_mallows_score,
                davies_bouldin_score
        target: Name of the column that contains the response
        kwargs: Key word arguments to be passed into K-Means cluster

    Returns:
        n_cluster: Optimal number of clusters for kmeans and/or elbow plot
    """
    scores = []
    cluster_range = range(cluster_min, cluster_max)
    unsupervised_metrics = ["silhouette_score", "davies_bouldin_score"]
    if analysis not in unsupervised_metrics:
        try:
            actual_labels = data[target]
            data.drop([target], axis=1, inplace=True)
        except KeyError:
            raise ValueError("target must not be None")
        for n in cluster_range:
            predicted_labels, kmeans = apply_kmeans(data, n, kwargs)
            analysis_func = getattr(sklearn.metrics, analysis)
            score = analysis_func(actual_labels, predicted_labels)
            scores.append(score)
        n_clusters = max(enumerate(scores, start=cluster_min), key=lambda x: x[1])[0]
    else:
        try:
            data.drop([target], axis=1, inplace=True)
        except KeyError:
            pass
        for n in cluster_range:
            predicted_labels, kmeans = apply_kmeans(data, n, kwargs)
            analysis_func = getattr(sklearn.metrics, analysis)
            score = analysis_func(data, predicted_labels)
            scores.append(score)
        n_clusters = max(enumerate(scores, start=cluster_min), key=lambda x: x[1])[0]
    return n_clusters, cluster_range, scores


def apply_kmeans(data, n_clusters=None, kwargs=None):
    """ Fits and predicts data using K-Means

     Args:
        data: Pandas data frame
        n_clusters: Number of clusters
        kwargs: Keyword arguments for kmeans

    Returns:
        n_clusters: Number of clusters
        y_kmeans: Predicted labels
    """
    if kwargs is None:
        kwargs = {"random_state": 0, "n_clusters": n_clusters}
    kmeans = KMeans(**kwargs)
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)
    return y_kmeans, kmeans


def hdbscan_cluster(
    data, reduc_df, truncator=None, interactive=True, min_cluster_size=15, kwargs=None,
):
    """Function to create a HDBSCAN clustering visualization

    Args:
        data: Pandas data frame
        reduc_df: Reduced data frame
        truncator: Instance of a dimensionality reduction method
        interactive: If False, create a seaborn plot
                    If True, create plotly interactive plot
        min_cluster_size: Minimum size of grouping to be considered a cluster
        kwargs: Key word arguments for HDBSCAN

    Returns:
        interactive plot: ploty scatter plot
        static_plot: Matplotlib scatter plott
        reduc_df: Reduced data frame
    """
    if kwargs is None:
        kwargs = {"min_cluster_size": min_cluster_size}
    elif "min_cluster_size" not in kwargs.keys():
        kwargs["min_cluster_size"] = min_cluster_size
    hdb = hdbscan.HDBSCAN(**kwargs)
    pred = hdb.fit_predict(data)
    reduc_df["cluster"] = pd.Series(pred).astype(str)

    if interactive is False:
        return (
            static_plot(
                data=reduc_df,
                method="HDBSCAN",
                x=reduc_df.columns[0],
                y=reduc_df.columns[1],
                truncator=truncator,
            ),
            reduc_df,
        )
    else:
        return (
            interactive_plot(
                df=reduc_df,
                method="HDBSCAN",
                x=reduc_df.columns[0],
                y=reduc_df.columns[1],
                color="cluster",
                truncator=truncator,
            ),
            reduc_df,
        )


def truncate_data(data):
    """ Reduces the number of dimensions for t-SNE to speed up computation time and reduce noise

        Args:
            data: Pandas data frame
        Returns:
            reduc_df: reduced data frame
            truncator: Instance of a dimensionality reduction method
    """
    if data.shape[1] > 50:
        data = data.to_numpy()
        sparsity = 1.0 - (count_nonzero(data) / float(data.size))
        if sparsity >= 0.5:
            reduc, truncator = dim_reduc(data, n_components=50, dim_method="tsvd")
        else:
            reduc, truncator = dim_reduc(data, n_components=50, dim_method="pca")
        reduc_df, truncator = dim_reduc(reduc, n_components=2, dim_method="tsne")
        return reduc_df, truncator
    reduc_df, truncator = dim_reduc(data, n_components=2, dim_method="tsne")
    return reduc_df, truncator


def interactive_plot(df, x, y, method, color, truncator=None):
    """ Creates interactive scatter plot using plotly

    Args:
        df: Pandas data frame
        x: x-axis column
        y: y-axis column
        method: Method for creating a title for the plot
        color: Feature from df that determines color for data points
        truncator: Instance of a dimensionality reduction method

    Returns:
        fig: Plotly scatter plot or plotly object
    """
    labels = df[color].unique()
    df_copy = df.copy()
    try:
        df_copy["cluster"] = df["cluster"]
        df_copy[
            "{} ({}% variance explained)".format(
                x, str(round(truncator.explained_variance_ratio_[0] * 100, 2))
            )
        ] = df[x]
        df_copy[
            "{} ({}% variance explained)".format(
                y, str(round(truncator.explained_variance_ratio_[1] * 100, 2))
            )
        ] = df[y]
        x = df_copy.columns[-2]
        y = df_copy.columns[-1]
    except AttributeError:
        pass

    trace_list = []
    for i in labels:
        if int(i) < 0:
            trace = go.Scatter(
                x=df_copy[df_copy[color] == i][x],
                y=df_copy[df_copy[color] == i][y],
                name="Noise",
                mode="markers",
                marker=dict(size=10, color="grey", line=dict(width=1)),
            )
            trace_list.append(trace)

        else:
            trace = go.Scatter(
                x=df_copy[df_copy[color] == i][x],
                y=df_copy[df_copy[color] == i][y],
                name="Cluster {}".format(i),
                mode="markers",
                marker=dict(size=10, colorscale="earth", line=dict(width=1)),
            )
            trace_list.append(trace)

    layout = dict(
        yaxis=dict(zeroline=False, title=y),
        xaxis=dict(zeroline=False, title=x),
        autosize=False,
        # width=1000, # context.viz_size # TODO (haishiro): Replace with get_option
        # height=1000, # context.viz_size # TODO (haishiro): Replace with get_option
        title={"text": "{} Cluster".format(method), "font": {"size": 25}},
    )

    fig = go.Figure(dict(data=trace_list, layout=layout))

    if get_ipython() is not None:
        init_notebook_mode(connected=True)
        return fig
    return fig


def static_plot(data, x, y, method, truncator=None):
    """ Creates a plot using seaborn's lmplot

    Args:
        data: Pandas data frame
        x: x-axis column
        y: y-axis column
        method: Method for creating a title for the plot
        truncator: Instance of a dimensionality reduction method

    Returns:
        fig: Matplotlib scatter plot
    """
    unique_labels = len(data["cluster"].unique())
    p = sns.set_palette("tab10", n_colors=unique_labels + 1)
    fig = sns.lmplot(
        data=data,
        x=x,
        y=y,
        hue="cluster",
        palette=p,
        fit_reg=False,
        # height=10 ,  # context.fig_height, # TODO (haishiro): Replace with get_option
        # aspect=1 ,  # context.fig_width / 10 ,  # context.fig_height, # TODO (haishiro): Replace with get_option # TODO (haishiro): Replace with get_option
        legend=False,
    )
    sns.set_context("talk")
    plt.legend(bbox_to_anchor=(1.1, 1), loc="upper right", ncol=1, title="Cluster")
    plt.title(method + " Cluster")
    try:
        plt.xlabel(
            x
            + " "
            + "({}% variance explained)".format(
                str(round(truncator.explained_variance_ratio_[0] * 100, 2))
            )
        )
        plt.ylabel(
            y
            + " "
            + "({}% variance explained)".format(
                str(round(truncator.explained_variance_ratio_[1] * 100, 2))
            )
        )
    except AttributeError:
        plt.xlabel(x)
        plt.xlabel(y)
    return fig
