import pytest
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import plotly
import matplotlib

import data_describe as dd
from data_describe.core.cluster import apply_kmeans, truncate_data, find_clusters


matplotlib.use("Agg")


def test_not_df():
    with pytest.raises(NotImplementedError):
        dd.cluster("this_is_a_string")


def test_find_clusters(numeric_data):
    n_clusters, cluster_range, scores = find_clusters(
        numeric_data,
        cluster_min=2,
        cluster_max=3,
        analysis="adjusted_rand_score",
        target="c",
    )
    assert isinstance(n_clusters, int)
    assert isinstance(cluster_range, range)
    assert isinstance(scores, list)


def test_apply_kmeans(numeric_data):
    y_kmeans, kmeans = apply_kmeans(numeric_data, n_clusters=2)
    assert y_kmeans.shape[0] == numeric_data.shape[0]
    assert isinstance(y_kmeans, np.ndarray)


def test_cluster_kmean(numeric_data):
    viz = dd.cluster(
        df=numeric_data, interactive=True, return_value="plot", kwargs={"n_clusters": 2}
    )
    assert isinstance(viz, plotly.graph_objs._figure.Figure)
    df = dd.cluster(
        df=numeric_data, return_value="reduc", kwargs={"n_clusters": 2}, target="c"
    )
    assert isinstance(df, pd.core.frame.DataFrame)
    assert df.shape[1] == 3
    df = dd.cluster(
        df=numeric_data, return_value="data", kwargs={"n_clusters": 2}, elbow=True
    )

    assert isinstance(df, pd.core.frame.DataFrame)
    assert df.shape[1] == numeric_data.shape[1]
    viz = dd.cluster(
        df=numeric_data, dim_method="tsne", kwargs={"n_clusters": 2}, interactive=False
    )
    assert isinstance(viz, sns.axisgrid.FacetGrid)
    viz = dd.cluster(df=numeric_data, dim_method="tsne", interactive=False)
    assert isinstance(viz, sns.axisgrid.FacetGrid)


def test_cluster_hdbscan(numeric_data):
    viz = dd.cluster(df=numeric_data, method="HDBSCAN", return_value="plot")
    assert isinstance(viz, plotly.graph_objs._figure.Figure)
    viz = dd.cluster(df=numeric_data, method="HDBSCAN", interactive=False)
    assert isinstance(viz, sns.axisgrid.FacetGrid)


def test_cluster_unsupported(numeric_data):
    with pytest.raises(ValueError):
        dd.cluster(df=numeric_data, method="random_model")
    with pytest.raises(ValueError):
        dd.cluster(df=numeric_data, return_value="unsupported_return_value")
    with pytest.raises(ValueError):
        find_clusters(
            data=numeric_data,
            analysis="adjusted_rand_score",
            cluster_min=2,
            cluster_max=3,
        )


def test_cluster_args(numeric_data):
    dd.cluster(
        df=numeric_data, interactive=False, method="HDBSCAN", kwargs={"alpha": 3.0}
    )


def test_truncate_data(numeric_data):
    reduc_df, truncator = truncate_data(numeric_data)
    assert reduc_df.shape[1] == 2
    assert isinstance(reduc_df, pd.core.frame.DataFrame)
    assert isinstance(truncator, sklearn.manifold.TSNE)
