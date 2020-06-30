import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA

from data_describe.utilities.compat import requires, _PACKAGE_INSTALLED
from data_describe.backends._backends import _get_df_backend

if _PACKAGE_INSTALLED["modin"]:
    import modin.pandas as modin


def dim_reduc(data, n_components, dim_method, df_backend=None):
    """Calls various dimensionality reduction methods

    Args:
        data: Pandas data frame
        n_components: Desired dimensionality for the data set prior to modeling
        dim_method: Dimensionality reduction method. Only pca, tsne, and
        tsvd are supported.

    Returns:
        Reduced data frame and reduction object
    """
    if dim_method == "pca":
        reduc_df, reductor = run_pca(data, n_components, df_backend)
    elif dim_method == "tsne":
        reduc_df, reductor = run_tsne(data, n_components, df_backend)
    elif dim_method == "tsvd":
        reduc_df, reductor = run_tsvd(data, n_components, df_backend)
    else:
        raise NotImplementedError("{} is not supported".format(dim_method))
    return reduc_df, reductor


def run_pca(data, n_components, df_backend=None):
    """Reduces the number of dimensions using PCA

        Args:
            data: Pandas data frame
            n_components: Desired dimensionality for the data set prior
            to modeling

        Returns:
            reduc_df: Reduced data frame
            pca: PCA object
    """
    fname = []
    for i in range(1, n_components + 1):
        fname.append("component_" + str(i))
    return _get_df_backend(df_backend).pca_type(data, n_components, column_names=fname)


def run_tsne(data, n_components, df_backend=None):
    """Reduces the number of dimensions using t-SNE

        Args:
            data: Pandas data frame
            n_components: Desired dimensionality for the data set prior
            to modeling

        Returns:
            reduc_df: Reduced data frame
            tsne: tsne object
    """
    tsne = TSNE(n_components, random_state=0)
    reduc = tsne.fit_transform(data)
    return _get_df_backend(df_backend).tsne_type(reduc), tsne


def run_tsvd(data, n_components, df_backend=None):
    """Reduces the number of dimensions using TSVD

        Args:
            data: Pandas data frame
            n_components: Desired dimensionality for the data set prior
            to modeling

        Returns:
            reduc_df: Reduced data frame
            t_svd: tsvd object
    """
    fname = []
    with np.errstate(invalid="ignore"):
        for i in range(1, n_components + 1):
            fname.append("component_" + str(i))
        t_svd = TruncatedSVD(n_components, random_state=0)
        reduc = t_svd.fit_transform(data)
        return _get_df_backend(df_backend).tsvd_type(reduc, column_names=fname), t_svd
