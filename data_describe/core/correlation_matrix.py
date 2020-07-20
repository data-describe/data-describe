import warnings

import pandas as pd
import numpy as np
from IPython import get_ipython
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from scipy.cluster import hierarchy
from sklearn.metrics import matthews_corrcoef
from scipy.stats import chi2_contingency, percentileofscore

from data_describe.utilities.contextmanager import _context_manager
from data_describe.utilities.colorscale import color_fade, rgb_to_str

warnings.filterwarnings(
    "error",
    category=DeprecationWarning,
    module="scipy",
    message="`scipy.sparse.sparsetools` is deprecated!",
)


@_context_manager
def correlation_matrix(
    data, cluster=False, categorical=False, return_values=False, context=None
):
    """Correlation matrix of numeric variables.

    Args:
        data: A pandas data frame
        cluster: If True, use clustering to reorder similar columns together
        categorical: If True, calculate categorical associations using Cramer's V, Correlation Ratio, and
            Point-biserial coefficient (aka Matthews correlation coefficient). All associations (including Pearson
            correlation) are in the range [0, 1]
        return_values: If True, return the correlation/association values
        context: The context manager

    Returns:
        Matplotlib graph or Pandas data frame
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
            association_matrix = reorder_by_cluster(association_matrix)
        else:
            association_matrix = reorder_by_original(association_matrix, data)

        if return_values:
            return association_matrix
        else:
            return plot_heatmap(association_matrix, context=context)
    else:
        if has_numeric:
            association_numeric = numeric.corr()
        else:
            raise ValueError(
                "No numerical features were found. Could not compute correlation."
            )

        association_numeric.fillna(0, inplace=True)

        if cluster:
            association_numeric = reorder_by_cluster(association_numeric)
        else:
            pass  # Pearson Correlation does not need reordering

        if return_values:
            return association_numeric
        else:
            return plot_heatmap(association_numeric, context=context)


def cramers_v_matrix(df):
    """Computes Cramer's V for all column pairs.

    Adapted from https://github.com/shakedzy/dython/blob/master/dython/nominal.py
    Args:
        df: A pandas data frame containing only categorical features

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
        num_df: A pandas dataframe containing only numeric features
        cat_df: A pandas dataframe containing only categorical features

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
        categorical: A Pandas Series of categorical values
        numeric: A Pandas Series of numeric values

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
        association_matrix: A matrix of associations (similarity)
        data: The original data frame

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
        association_matrix: The square matrix of correlations/associations
        original_df: The original data frame

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


def plot_heatmap(association_matrix, context=None):
    """Plot the heatmap for the association matrix.

    Args:
        association_matrix: The association matrix
        context: The context

    Returns:
        Seaborn figure
    """
    # Plot lower left triangle
    x_ind, y_ind = np.triu_indices(association_matrix.shape[0])
    corr = association_matrix.to_numpy()
    for x, y in zip(x_ind, y_ind):
        corr[x, y] = None

    # Set up the color scale
    blue_anchor = (65, 124, 167)
    white_anchor = (242, 242, 242)
    red_anchor = (217, 58, 70)

    vmin = min(corr.flatten()[~np.isnan(corr.flatten())])
    vmax = max(corr.flatten()[~np.isnan(corr.flatten())])

    if vmin > 0:
        cmin = color_fade(white_anchor, red_anchor, 1 - vmin)
        cmax = color_fade(white_anchor, red_anchor, 1 - vmax)
        cscale = [[0, rgb_to_str(cmin)], [1.0, rgb_to_str(cmax)]]
    elif vmax < 0:
        cmin = color_fade(blue_anchor, white_anchor, -vmin)
        cmax = color_fade(blue_anchor, white_anchor, -vmax)
        cscale = [[0, rgb_to_str(cmin)], [1.0, rgb_to_str(cmax)]]
    else:
        cmin = color_fade(blue_anchor, white_anchor, -vmin)
        cmax = color_fade(white_anchor, red_anchor, 1 - vmax)
        corr_values = corr[~np.isnan(corr)].flatten()
        z_val = percentileofscore(corr_values, 0.0) / 100.0
        cscale = [
            [0, rgb_to_str(cmin)],
            [z_val, rgb_to_str(white_anchor)],
            [1.0, rgb_to_str(cmax)],
        ]

    # Generate a custom diverging colormap
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=np.flip(corr, axis=0),
                x=association_matrix.columns.values,
                y=association_matrix.columns.values[::-1],
                connectgaps=False,
                xgap=2,
                ygap=2,
                colorscale=cscale,
                colorbar={"title": "Strength"},
            )
        ],
        layout=go.Layout(
            autosize=False,
            width=context.viz_size,
            height=context.viz_size,
            title={"text": "Correlation Matrix", "font": {"size": 25}},
            xaxis=go.layout.XAxis(
                automargin=True, tickangle=270, ticks="", showgrid=False
            ),
            yaxis=go.layout.YAxis(automargin=True, ticks="", showgrid=False),
            plot_bgcolor="rgb(0,0,0,0)",
            paper_bgcolor="rgb(0,0,0,0)",
        ),
    )

    if get_ipython() is not None:
        init_notebook_mode(connected=True)
        return iplot(fig, config={"displayModeBar": False})
    else:
        return fig
