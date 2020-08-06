from data_describe._compat import _DATAFRAME_TYPE
from data_describe.backends._backends import _get_compute_backend


def dim_reduc(
    data,
    n_components: int,
    dim_method: str,
    apply_tsvd: bool = True,
    compute_backend=None,
):
    """Reduces the number of dimensions of the input data.

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the data set prior to modeling
        dim_method: {'pca', 'ipca', 'tsne', 'tsvd'}
        - pca: Principal Component Analysis
        - ipca: Incremental Principal Component Analysis. Highly suggested for very large datasets
        - tsne: T-distributed Stochastic Neighbor Embedding
        - tsvd: Truncated Singular Value Decomposition
        apply_tsvd: If True, TSVD will be run before t-SNE. This is highly recommended when running t-SNE

    Returns:
        The dimensionally-reduced dataframe and reduction object
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
    """Reduces the number of dimensions of the input data using PCA.

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the data set prior to modeling

    Returns:
        reduc_df: The dimensionally-reduced dataframe
        pca: The applied PCA object
    """
    fname = ["component_{}".format(i) for i in range(1, n_components + 1)]
    return _get_compute_backend(compute_backend, data).compute_run_pca(
        data, n_components, column_names=fname
    )


def run_ipca(data, n_components, compute_backend=None):
    """Reduces the number of dimensions of the input data using Incremental PCA.

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the data set prior to modeling

    Returns:
        reduc_df: The dimensionally-reduced dataframe
        ipca: The applied IncrementalPCA object
    """
    fname = ["component_{}".format(i) for i in range(1, n_components + 1)]
    return _get_compute_backend(compute_backend, data).compute_run_ipca(
        data, n_components, column_names=fname
    )


def run_tsne(data, n_components, apply_tsvd=True, compute_backend=None):
    """Reduces the number of dimensions of the input data using t-SNE.

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the output dataset
        apply_tsvd: If True, TSVD will be run before t-SNE. This is highly recommended when running t-SNE

    Returns:
        reduc_df: The dimensionally-reduced dataframe
        tsne: The applied t-SNE object
    """
    return _get_compute_backend(compute_backend, data).compute_run_tsne(data, n_components, apply_tsvd)


def run_tsvd(data, n_components, compute_backend=None):
    """Reduces the number of dimensions of the input data using TSVD.

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the output dataset

    Returns:
        reduc_df: The dimensionally-reduced dataframe
        tsne: The applied TSVD object
    """
    fname = ["component_{}".format(i) for i in range(1, n_components + 1)]
    return _get_compute_backend(compute_backend, data).compute_run_tsvd(data, n_components, column_names=fname)
