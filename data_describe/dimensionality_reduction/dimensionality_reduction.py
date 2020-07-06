import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends._backends import _get_compute_backend


def dim_reduc(
    data,
    n_components: int,
    dim_method: str,
    apply_tsvd: bool = True,
    compute_backend=None,
):
    """Reduces the number of dimensions of the input data
    Args:
        data: Data frame
        n_components: Desired dimensionality for the data set prior to modeling
        dim_method: {'pca', 'ipca', 'tsne', 'tsvd'}
            - pca: Principal Component Analysis
            - ipca: Incremental Principal Component Analysis. Highly suggested for very large datasets
            - tsne: T-distributed Stochastic Neighbor Embedding
            - tsvd: Truncated Singular Value Decomposition
        apply_tsvd: If True, TSVD will be run before t-SNE. This is highly recommended when running t-SNE
    Returns:
        Dimensionally-reduced data frame and reduction object
    """
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Data must be a Pandas (or Modin) DataFrame")

    if dim_method == "pca":
        reduc_df, reductor = run_pca(data, n_components, compute_backend)
    elif dim_method == "ipca":
        reduc_df, reductor = run_ipca(data, n_components, compute_backend)
    elif dim_method == "tsne":
        reduc_df, reductor = run_tsne(data, n_components, apply_tsvd, compute_backend)
    elif dim_method == "tsvd":
        reduc_df, reductor = run_tsvd(data, n_components, compute_backend)
    else:
        raise NotImplementedError("{} is not supported".format(dim_method))
    return reduc_df, reductor


def run_pca(data, n_components, compute_backend=None):
    """Reduces the number of dimensions using PCA
        Args:
            data: Data frame
            n_components: Desired dimensionality for the data set prior
            to modeling
        Returns:
            reduc_df: Reduced data frame
            pca: PCA object
    """
    fname = []
    for i in range(1, n_components + 1):
        fname.append("component_" + str(i))
    return _get_compute_backend(compute_backend, data).compute_run_pca(
        data, n_components, column_names=fname
    )


def run_ipca(data, n_components, compute_backend=None):
    """Reduces the number of dimensions using Incremental PCA
        Args:
            data: Data frame
            n_components: Desired dimensionality for the data set prior
            to modeling
        Returns:
            reduc_df: Reduced data frame
            ipca: PCA object
    """
    fname = []
    for i in range(1, n_components + 1):
        fname.append("component_" + str(i))
    return _get_compute_backend(compute_backend, data).compute_run_ipca(
        data, n_components, column_names=fname
    )


def run_tsne(data, n_components, apply_tsvd=True, compute_backend=None):
    """Reduces the number of dimensions using t-SNE
        Args:
            data: Data frame
            n_components: Desired dimensionality for the data set prior
            to modeling
            apply_tsvd: If True, TSVD will be run before t-SNE. This is highly recommended when running t-SNE
        Returns:
            reduc_df: Reduced data frame
            tsne: tsne object
    """
    if apply_tsvd:
        data = run_tsvd(data, n_components, compute_backend)[0]
    tsne = TSNE(n_components, random_state=0)
    reduc = tsne.fit_transform(data)
    return _get_compute_backend(compute_backend, data).compute_run_tsne(reduc), tsne


def run_tsvd(data, n_components, compute_backend=None):
    """Reduces the number of dimensions using TSVD
        Args:
            data: Data frame
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
        return (
            _get_compute_backend(compute_backend, data).compute_run_tsvd(
                reduc, column_names=fname
            ),
            t_svd,
        )
