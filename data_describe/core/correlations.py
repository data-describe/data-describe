import warnings

import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
from sklearn.metrics import matthews_corrcoef
from scipy.stats import chi2_contingency

from data_describe._widget import BaseWidget
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends import _get_viz_backend, _get_compute_backend


def correlation_matrix(
    data,
    cluster=False,
    categorical=False,
    compute_backend=None,
    viz_backend=None,
    **kwargs,
):
    """Correlation matrix of numeric variables.

    Args:
        data (DataFrame): A data frame
        cluster (bool): If True, use clustering to reorder similar columns together

        categorical (bool): If True, calculate categorical associations using Cramer's V, Correlation Ratio, and
            Point-biserial coefficient (aka Matthews correlation coefficient). All associations (including Pearson
            correlation) are in the range [0, 1].

        compute_backend: Select computing backend. Defaults to None (pandas).
        viz_backend: The visualization backend. Only 'plotly' is supported. Defaults to plotly.

    Returns:
        CorrelationMatrixWidget
    """
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Data frame required")

    corrwidget = _get_compute_backend(compute_backend, data).compute_correlation_matrix(
        data, cluster=cluster, categorical=categorical, **kwargs
    )

    corrwidget.viz_backend = viz_backend

    return corrwidget


class CorrelationMatrixWidget(BaseWidget):
    """Interface for collecting additional information about the correlation matrix."""

    def __init__(
        self,
        association_matrix=None,
        cluster_matrix=None,
        categorical=None,
        viz_data=None,
        **kwargs,
    ):
        """Correlation matrix.

        Args:
            association_matrix (DataFrame): The association matrix. Defaults to None.
            cluster_matrix (DataFrame, optional): The clustered association matrix. Defaults to None.
            categorical (bool, optional): True if association matrix contains categorical values. Defaults to None.
            viz_data (DataFrame): The data to be visualized. Defaults to None.
        """
        super(CorrelationMatrixWidget, self).__init__(**kwargs)
        self.association_matrix = association_matrix
        self.cluster_matrix = cluster_matrix
        self.viz_data = viz_data
        self.categorical = categorical

    def __str__(self):
        return "data-describe Correlation Matrix Widget"

    def show(self, viz_backend=None, **kwargs):
        """Show the Correlation Matrix plot."""
        backend = viz_backend or self.viz_backend

        if self.viz_data is None:
            raise ValueError("Could not find data to visualize.")

        return _get_viz_backend(backend).viz_correlation_matrix(self.viz_data, **kwargs)


def _pandas_compute_correlation_matrix(data, cluster=False, categorical=False):
    """Correlation matrix of numeric variables.

    Args:
        data (DataFrame): The data frame
        cluster (bool): If True, use clustering to reorder similar columns together

        categorical (bool): If True, calculate categorical associations using Cramer's V, Correlation Ratio, and
            Point-biserial coefficient (aka Matthews correlation coefficient). All associations (including Pearson
            correlation) are in the range [0, 1]

    Returns:
        CorrelationMatrixWidget
    """
    numeric = data.select_dtypes(["number"])
    categoric = data[[col for col in data.columns if col not in numeric.columns]]

    has_categoric = categoric.shape[1] > 0
    has_numeric = numeric.shape[1] > 0

    if categorical and not has_categoric:
        warnings.warn(
            UserWarning(
                "Categorical associations were requested, but no categorical features were found. "
                "Defaulting to Pearson Correlation."
            )
        )
        categorical = False

    if categorical:
        if has_numeric:
            association_numeric = np.abs(numeric.corr())
            association_cramers = cramers_v_matrix(categoric)
            association_cr = correlation_ratio_matrix(numeric, categoric)

            association_matrix = pd.concat(
                [
                    pd.concat([association_numeric, association_cr], axis=1),
                    pd.concat([association_cr.T, association_cramers], axis=1),
                ],
                axis=0,
            )
        else:
            association_matrix = cramers_v_matrix(categoric)

        association_matrix.fillna(0, inplace=True)

        if cluster:
            cluster_matrix = reorder_by_cluster(association_matrix)
        else:
            association_matrix = reorder_by_original(association_matrix, data)

    else:
        if has_numeric:
            association_matrix = numeric.corr()
        else:
            raise ValueError(
                "No numerical features were found. Could not compute correlation."
            )

        association_matrix.fillna(0, inplace=True)

        if cluster:
            cluster_matrix = reorder_by_cluster(association_matrix)
        else:
            pass  # Pearson Correlation does not need reordering

    return CorrelationMatrixWidget(
        association_matrix=association_matrix,
        cluster_matrix=cluster_matrix if cluster else None,
        viz_data=cluster_matrix if cluster else association_matrix,
    )


def cramers_v_matrix(df):
    """Computes Cramer's V for all column pairs.

    Adapted from https://github.com/shakedzy/dython/blob/master/dython/nominal.py

    Args:
        df (DataFrame): A data frame containing only categorical features

    Returns:
        A pandas data frame
    """
    index = df.columns.values
    cramers_matrix = pd.DataFrame(
        [[cramers_v(df[x], df[y]) for x in index] for y in index]
    )

    # Cramer's V can be NaN when there are not enough instances in a category
    cramers_matrix.fillna(0)

    # Assign column and row labels
    cramers_matrix.columns = index
    cramers_matrix.set_index(index, inplace=True)

    return cramers_matrix


def cramers_v(x, y):
    """Calculates Cramer's V statistic for categorical-categorical association.

    Adapted from https://github.com/shakedzy/dython/blob/master/dython/nominal.py

    Args:
        x: A list, numpy array, or pandas series of categorical measurements
        y: A list, numpy array, or pandas series of categorical measurements

    Returns:
        Cramer's V value (float)
    """
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape[0] == 2 and confusion_matrix.shape[1] == 2:
        # phi coefficient
        return matthews_corrcoef(x.astype(str), y.astype(str))
    else:
        # Cramer's V
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        if n == k or n == r:
            return 1.0
        else:
            chi2 = chi2_contingency(confusion_matrix)[0]
            phi2 = chi2 / n
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def correlation_ratio_matrix(num_df, cat_df):
    """Computes correlation ratio for all numeric-categoric pairs of columns.

    Args:
        num_df (DataFrame): A dataframe containing only numeric features
        cat_df (DataFrame)): A dataframe containing only categorical features

    Returns:
        A pandas data frame
    """
    num_index = num_df.columns.values
    cat_index = cat_df.columns.values
    corr_ratio_mat = pd.DataFrame(
        [
            [correlation_ratio(cat_df[x], num_df[y]) for x in cat_index]
            for y in num_index
        ]
    )

    # Assign column and row labels
    corr_ratio_mat.columns = cat_index
    corr_ratio_mat.set_index(num_index, inplace=True)
    return corr_ratio_mat


def correlation_ratio(categorical, numeric):
    """Computes correlation ratio between a categorical column and numeric column.

    Args:
        categorical (Series): A Series of categorical values
        numeric (Series): A Series of numeric values

    Returns:
        Correlation Ratio value (float)
    """
    df = pd.concat([categorical, numeric], axis=1)
    n = numeric.shape[0]
    ybar = np.mean(numeric)

    # Get average and count for each category level
    agg = df.groupby(categorical.name).agg(["count", "mean"])

    # Remove the Multilevel Index
    agg.columns = agg.columns.droplevel(0)

    category_variance = (agg["mean"] - ybar) ** 2
    weighted_category_variance = (
        np.sum(np.multiply(agg["count"], category_variance)) / n
    )

    total_variance = np.sum((numeric - ybar) ** 2) / n

    eta = np.sqrt(weighted_category_variance / total_variance)

    return eta


def reorder_by_cluster(association_matrix):
    """Reorder an association matrix by cluster distances.

    Args:
        association_matrix (DataFrame): A matrix of associations (similarity)
        data (DataFrame): The original data frame

    Returns:
        A Pandas data frame
    """
    distance = 1 - association_matrix

    # Determine padding dimensions, if non-square
    max_dim = max(distance.shape[0], distance.shape[1])
    pad_distance = np.pad(
        distance,
        (
            (0, min(1, max_dim - distance.shape[0])),
            (0, min(1, max_dim - distance.shape[1])),
        ),
        mode="constant",
        constant_values=0,
    )

    if distance.shape[0] == distance.shape[1]:
        y = np.array(pad_distance)[np.triu_indices_from(pad_distance, 1)]
    else:
        # Non-square matrix
        y = np.array(pad_distance)[np.tril_indices_from(pad_distance, 1)]

    # Use hierarchical clustering to get order
    link = hierarchy.linkage(y)
    dendrogram = hierarchy.dendrogram(link, get_leaves=True, no_plot=True)
    new_order = [x for x in dendrogram["leaves"] if x < max_dim]

    reorder_corr = pd.DataFrame(
        [row[new_order] for row in association_matrix.values[new_order]]
    )

    # Assign column and row labels
    reorder_corr.columns = [association_matrix.columns.values[i] for i in new_order]
    reorder_corr.set_index(
        np.array([association_matrix.columns.values[i] for i in new_order]),
        inplace=True,
    )

    return reorder_corr


def reorder_by_original(association_matrix, original_df):
    """Reorder the matrix to the original order.

    Args:
        association_matrix ((DataFrame): The square matrix of correlations/associations
        original_df (DataFrame): The original data frame

    Returns:
        A Pandas Data frame
    """
    # Get the original column order
    order = [
        np.where(association_matrix.columns.values == label)[0][0]
        for label in original_df.columns.values
        if label in association_matrix.columns.values
    ]

    reorder_matrix = pd.DataFrame(
        [row[order] for row in association_matrix.values[order]]
    )

    # Assign column and row labels
    reorder_matrix.columns = [association_matrix.columns.values[i] for i in order]
    reorder_matrix.set_index(
        np.array([association_matrix.columns.values[i] for i in order]), inplace=True
    )

    return reorder_matrix
