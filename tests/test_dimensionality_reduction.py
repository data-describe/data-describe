import pytest
import sklearn
import pandas as pd
import modin.pandas as modin

from data_describe.compat import _DATAFRAME_TYPE
from data_describe.dimensionality_reduction.dimensionality_reduction import dim_reduc


def test_error(compute_backend_df):
    with pytest.raises(NotImplementedError):
        dim_reduc(data=compute_backend_df, n_components=2, dim_method="test_dim_method")


def test_pca(compute_numeric_backend_df):
    x = dim_reduc(data=compute_numeric_backend_df, n_components=2, dim_method="pca")
    assert isinstance(x, tuple)
    assert isinstance(x[0], _DATAFRAME_TYPE)
    assert isinstance(x[1], sklearn.decomposition.PCA)


def test_ipca(compute_numeric_backend_df):
    x = dim_reduc(data=compute_numeric_backend_df, n_components=2, dim_method="ipca")
    assert isinstance(x, tuple)
    assert isinstance(x[0], _DATAFRAME_TYPE)
    assert isinstance(x[1], sklearn.decomposition.IncrementalPCA)


def test_tsne(compute_numeric_backend_df):
    x = dim_reduc(data=compute_numeric_backend_df, n_components=2, dim_method="tsne")
    y = dim_reduc(data=compute_numeric_backend_df, n_components=2, apply_tsvd=False, dim_method='tsne')
    assert isinstance(x, tuple)
    assert isinstance(x[0], _DATAFRAME_TYPE)
    assert isinstance(x[1], sklearn.manifold.TSNE)
    assert not x[0].equals(y[0])


def test_tsvd(compute_numeric_backend_df):
    x = dim_reduc(data=compute_numeric_backend_df, n_components=2, dim_method="tsvd")
    assert isinstance(x, tuple)
    assert isinstance(x[0], _DATAFRAME_TYPE)
    assert isinstance(x[1], sklearn.decomposition.TruncatedSVD)
