import pytest
import sklearn
import pandas as pd

from data_describe.dimensionality_reduction.dimensionality_reduction import dim_reduc


def test_error(data):
    with pytest.raises(NotImplementedError):
        dim_reduc(data=data, n_components=2, dim_method="test_dim_method")


def test_pca(data):
    x = dim_reduc(data=data, n_components=2, dim_method="pca")
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], sklearn.decomposition.PCA)


def test_tsne(data):
    x = dim_reduc(data=data, n_components=2, dim_method="tsne")
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], sklearn.manifold.TSNE)


def test_tsvd(data):
    x = dim_reduc(data=data, n_components=2, dim_method="tsvd")
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], sklearn.decomposition.TruncatedSVD)
