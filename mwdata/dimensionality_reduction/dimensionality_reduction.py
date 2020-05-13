import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, PCA


def dim_reduc(data, n_components, dim_method):
    """Calls various dimensionality reduction methods

    Args:
        data: Pandas data frame
        n_components: Desired dimensionality for the data set prior to modeling
        dim_method: Dimensionality reduction method. Only pca, tsne, and tsvd are supported.

    Returns:
        Reduced data frame and reduction object
    """
    if dim_method == 'pca':
        reduc_df, reductor = run_pca(data, n_components)
    elif dim_method == 'tsne':
        reduc_df, reductor = run_tsne(data, n_components)
    elif dim_method == 'tsvd':
        reduc_df, reductor = run_tsvd(data, n_components)
    else:
        raise NotImplementedError('{} is not supported'.format(dim_method))
    return reduc_df, reductor


def run_pca(data, n_components):
    """Reduces the number of dimensions using PCA

        Args:
            data: Pandas data frame
            n_components: Desired dimensionality for the data set prior to modeling

        Returns:
            reduc_df: Reduced data frame
            pca: PCA object
        """
    fname = []
    for i in range(1, n_components + 1):
        fname.append('component_' + str(i))
    pca = PCA(n_components, random_state=0)
    reduc = pca.fit_transform(data)
    reduc_df = pd.DataFrame(reduc, columns=fname)
    return reduc_df, pca


def run_tsne(data, n_components):
    """Reduces the number of dimensions using t-SNE

        Args:
            data: Pandas data frame
            n_components: Desired dimensionality for the data set prior to modeling

        Returns:
            reduc_df: Reduced data frame
            tsne: tsne object
    """
    tsne = TSNE(n_components, random_state=0)
    reduc = tsne.fit_transform(data)
    reduc_df = pd.DataFrame(reduc, columns=['ts1', 'ts2'])
    return reduc_df, tsne


def run_tsvd(data, n_components):
    """Reduces the number of dimensions using TSVD

        Args:
            data: Pandas data frame
            n_components: Desired dimensionality for the data set prior to modeling

        Returns:
            reduc_df: Reduced data frame
            t_svd: tsvd object
    """
    fname = []
    with np.errstate(invalid='ignore'):
        for i in range(1, n_components + 1):
            fname.append('component_' + str(i))
        t_svd = TruncatedSVD(n_components, random_state=0)
        reduc = t_svd.fit_transform(data)
        reduc_df = pd.DataFrame(reduc, columns=fname)
        return reduc_df, t_svd
