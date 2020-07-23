import modin.pandas as modin
from sklearn.decomposition import PCA, IncrementalPCA


def compute_run_pca(data, n_components, column_names):
    """Performs PCA on the provided dataset

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the output dataset
        column_names: Names for the columns in the output dataset

    Returns:
        reduc_df: The dimensionally-reduced Modin dataframe
        pca: The applied PCA object
    """
    pca = PCA(n_components)
    reduc = pca.fit_transform(data)
    reduc_df = modin.DataFrame(reduc, columns=column_names)
    return reduc_df, pca


def compute_run_ipca(data, n_components, column_names):
    """Performs Incremental PCA on the provided dataset

    Args:
        data: The dataframe
        n_components: Desired dimensionality for the output dataset
        column_names: Names for the columns in the output dataset

    Returns:
        reduc_df: The dimensionally-reduced Modin dataframe
        ipca: The applied IncrementalPCA object
    """
    ipca = IncrementalPCA(n_components)
    reduc = ipca.fit_transform(data)
    reduc_df = modin.DataFrame(reduc, columns=column_names)
    return reduc_df, ipca


def compute_run_tsne(reduc):
    """Transforms the dimensionally reduced array into a Modin dataframe

    Args:
        reduc: An array of the dimensionally reduced dataset

    Returns:
        The dimensionally-reduced Modin dataframe
    """
    return modin.DataFrame(reduc, columns=["ts1", "ts2"])


def compute_run_tsvd(reduc, column_names):
    """Transforms the dimensionally reduced array into a Modin dataframe

    Args:
        reduc: An array of the dimensionally reduced dataset
        columns_names: Names for the columns in the output dataset

    Returns:
        The dimensionally-reduced Modin dataframe
    """
    return modin.DataFrame(reduc, columns=column_names)
