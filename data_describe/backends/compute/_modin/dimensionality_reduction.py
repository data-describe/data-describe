import modin.pandas as modin
from sklearn.decomposition import PCA, IncrementalPCA


def compute_run_pca(data, n_components, column_names):
    pca = PCA(n_components)
    reduc = pca.fit_transform(data)
    reduc_df = modin.DataFrame(reduc, columns=column_names)
    return reduc_df, pca


def compute_run_ipca(data, n_components, column_names):
    ipca = IncrementalPCA(n_components)
    reduc = ipca.fit_transform(data)
    reduc_df = modin.DataFrame(reduc, columns=column_names)
    return reduc_df, ipca


def compute_run_tsne(reduc):
    return modin.DataFrame(reduc, columns=["ts1", "ts2"])


def compute_run_tsvd(reduc, column_names):
    return modin.DataFrame(reduc, columns=column_names)
