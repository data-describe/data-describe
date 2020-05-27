# import numpy
# import pytest
# import pandas as pd
# # import numpy as np
# import seaborn as sns
# import plotly
# # import sklearn
# from sklearn.datasets import load_wine
# import matplotlib

# import mwdata.core.cluster as mw

# matplotlib.use("Agg")


# @pytest.fixture
# def data_loader():
#     data = load_wine()
#     df = pd.DataFrame(data=data.data, columns=data.feature_names)
#     df = pd.concat([pd.Series(data.target), df], axis=1)
#     df = df.rename({0: "Target"}, axis=1)
#     return df.sample(n=50, replace=True, random_state=1)


# def test_not_df():
#     with pytest.raises(NotImplementedError):
#         mw.cluster("this_is_a_string")


# def test_find_clusters(data_loader):
#     n_clusters, cluster_range, scores = mw.find_clusters(
#         data_loader,
#         cluster_min=2,
#         cluster_max=3,
#         analysis="adjusted_rand_score",
#         target="Target",
#     )
#     assert isinstance(n_clusters, int)
#     assert isinstance(cluster_range, range)
#     assert isinstance(scores, list)


# def test_apply_kmeans(data_loader):
#     y_kmeans, kmeans = mw.apply_kmeans(data_loader, n_clusters=2)
#     assert y_kmeans.shape[0] == data_loader.shape[0]
#     assert isinstance(y_kmeans, numpy.ndarray)


# def test_cluster_kmean(data_loader):
#     viz = mw.cluster(
#         df=data_loader, interactive=True, return_value="plot", kwargs={"n_clusters": 2}
#     )
#     assert isinstance(viz, plotly.graph_objs._figure.Figure)
#     df = mw.cluster(
#         df=data_loader, return_value="reduc", kwargs={"n_clusters": 2}, target="Target"
#     )
#     assert isinstance(df, pd.core.frame.DataFrame)
#     assert df.shape[1] == 3
#     df = mw.cluster(
#         df=data_loader, return_value="data", kwargs={"n_clusters": 2}, elbow=True
#     )
#     assert isinstance(df, pd.core.frame.DataFrame)
#     assert df.shape[1] == data_loader.shape[1]
#     viz = mw.cluster(
#         df=data_loader, dim_method="tsne", kwargs={"n_clusters": 2}, interactive=False
#     )
#     assert isinstance(viz, sns.axisgrid.FacetGrid)
#     viz = mw.cluster(df=data_loader, dim_method="tsne", interactive=False)
#     assert isinstance(viz, sns.axisgrid.FacetGrid)


# def test_cluster_hdbscan(data_loader):
#     viz = mw.cluster(df=data_loader, method="HDBSCAN", return_value="plot")
#     assert isinstance(viz, plotly.graph_objs._figure.Figure)
#     viz = mw.cluster(df=data_loader, method="HDBSCAN", interactive=False)
#     assert isinstance(viz, sns.axisgrid.FacetGrid)


# def test_cluster_unsupported(data_loader):
#     with pytest.raises(ValueError):
#         mw.cluster(df=data_loader, method="random_model")
#     with pytest.raises(ValueError):
#         mw.cluster(df=data_loader, return_value="unsupported_return_value")
#     with pytest.raises(ValueError):
#         mw.find_clusters(
#             data=data_loader,
#             analysis="adjusted_rand_score",
#             cluster_min=2,
#             cluster_max=3,
#         )


# def test_cluster_args(data_loader):
#     mw.cluster(df=data_loader, method="HDBSCAN", kwargs={"alpha": 3.0})


# def test_truncate_data(data_loader):
#     data = pd.DataFrame(np.zeros((52, 52))).dropna()
#     reduc_df, truncator = mw.truncate_data(data)
#     assert reduc_df.shape[1] == 2

#     data = pd.DataFrame(np.ones((52, 52))).dropna()
#     reduc_df, truncator = mw.truncate_data(data)
#     assert reduc_df.shape[1] == 2

#     reduc_df, truncator = mw.truncate_data(data_loader)
#     assert isinstance(reduc_df, pd.core.frame.DataFrame)
#     assert isinstance(truncator, sklearn.manifold.t_sne.TSNE)
