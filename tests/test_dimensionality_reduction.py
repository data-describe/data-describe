import pytest
import pandas as pd
import sklearn
from sklearn.datasets import load_wine

from mwdata.dimensionality_reduction.dimensionality_reduction import dim_reduc


@pytest.fixture
def data_loader():
    data = load_wine()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df = pd.concat([pd.Series(data.target), df], axis=1)
    df = df.rename({0: "Target"}, axis=1)
    return df.sample(n=50, replace=True, random_state=1)


def test_error():
    with pytest.raises(NotImplementedError):
        dim_reduc(data=data_loader, n_components=2, dim_method="test_dim_method")


def test_pca(data_loader):
    x = dim_reduc(data=data_loader, n_components=2, dim_method="pca")
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], sklearn.decomposition.pca.PCA)


def test_tsne(data_loader):
    x = dim_reduc(data=data_loader, n_components=2, dim_method="tsne")
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], sklearn.manifold.t_sne.TSNE)


def test_tsvd(data_loader):
    x = dim_reduc(data=data_loader, n_components=2, dim_method="tsvd")
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], sklearn.decomposition.truncated_svd.TruncatedSVD)
