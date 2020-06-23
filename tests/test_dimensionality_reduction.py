import pytest
import sklearn
import pandas as pd
import numpy as np
import modin.pandas as modin

from mwdata.dimensionality_reduction.dimensionality_reduction import dim_reduc
from ._test_data import DATA


@pytest.fixture
def data():
    return DATA.select_dtypes(np.number)

@pytest.fixture
def modin_data():
    return modin.DataFrame(DATA).select_dtypes(np.number)


def test_error():
    with pytest.raises(NotImplementedError):
        dim_reduc(data=data, n_components=2, dim_method="test_dim_method")


def test_pca(data):
    x = dim_reduc(data=data, n_components=2, dim_method="pca")
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], sklearn.decomposition.PCA)

def test_pca_modin(modin_data):
    x = dim_reduc(data=modin_data, n_components=2, dim_method="pca", df_type='modin')
    assert isinstance(x, tuple)
    assert isinstance(x[0], modin.dataframe.DataFrame)
    assert isinstance(x[1], sklearn.decomposition.PCA)

def test_tsne(data):
    x = dim_reduc(data=data, n_components=2, dim_method="tsne")
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], sklearn.manifold.TSNE)

def test_tsne_modin(modin_data):
    x = dim_reduc(data=modin_data, n_components=2, dim_method="tsne",  df_type='modin')
    assert isinstance(x, tuple)
    assert isinstance(x[0], modin.dataframe.DataFrame)
    assert isinstance(x[1], sklearn.manifold.TSNE)

def test_tsvd(data):
    x = dim_reduc(data=data, n_components=2, dim_method="tsvd")
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], sklearn.decomposition.TruncatedSVD)

def test_tsvd_modin(modin_data):
    x = dim_reduc(data=modin_data, n_components=2, dim_method="tsvd", df_type='modin')
    assert isinstance(x, tuple)
    assert isinstance(x[0], modin.dataframe.DataFrame)
    assert isinstance(x[1], sklearn.decomposition.TruncatedSVD)
