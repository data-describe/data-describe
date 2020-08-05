import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

from data_describe import compat
from data_describe.compat import requires


@requires("modin")
def compute_run_pca(data, n_components, column_names):
    """Performs PCA on the provided dataset.

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the output dataset
        column_names: Names for the columns in the output dataset

    Returns:
        The dimensionally-reduced Modin dataframe
        pca: The applied PCA object
    """
    pca = PCA(n_components)
    reduc = pca.fit_transform(data)
    return compat.modin.DataFrame(reduc, columns=column_names), pca


@requires("modin")
def compute_run_ipca(data, n_components, column_names):
    """Performs Incremental PCA on the provided dataset.

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the output dataset
        column_names: Names for the columns in the output dataset

    Returns:
        The dimensionally-reduced Modin dataframe
        ipca: The applied IncrementalPCA object
    """
    ipca = IncrementalPCA(n_components)
    reduc = ipca.fit_transform(data)
    return compat.modin.DataFrame(reduc, columns=column_names), ipca


@requires("modin")
def compute_run_tsne(data, n_components, apply_tsvd):
    """Performs dimensionality reduction using t-SNE on the provided dataset.

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the output dataset
        apply_tsvd: If True, TSVD will be run before t-SNE. This is highly recommended when running t-SNE

    Returns:
        The dimensionally-reduced Modin dataframe
        tsne: The applied t-SNE object
    """
    if apply_tsvd:
        fname = ["component_{}".format(i) for i in range(1, n_components + 1)]
        data = compute_run_tsvd(data, n_components, fname)[0]
    tsne = TSNE(n_components, random_state=0)
    reduc = tsne.fit_transform(data)
    return compat.modin.DataFrame(reduc, columns=["ts1", "ts2"]), tsne


@requires("modin")
def compute_run_tsvd(data, n_components, column_names):
    """Performs dimensionality reduction using TSVD on the provided dataset.

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the output dataset
        column_names: Names for the columns in the output dataset

    Returns:
        The dimensionally-reduced Modin dataframe
        t_svd: The applied TSVD object
    """
    with np.errstate(invalid="ignore"):
        t_svd = TruncatedSVD(n_components, random_state=0)
        reduc = t_svd.fit_transform(data)
    return compat.modin.DataFrame(reduc, columns=column_names), t_svd
