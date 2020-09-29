import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from data_describe.config._config import get_option
from data_describe.misc.colors import get_p_RdBl_cmap


def viz_correlation_matrix(
    association_matrix, annot=False, xticks_rotation=90, yticks_rotation=0
):
    """Plot the heatmap for the association matrix.

    Args:
        association_matrix (DataFrame): The association matrix
        annot (bool): If True, add correlation values to plot. Defaluts to False.
        xticks_rotation (int): Degrees of rotation for the xticks. Defaults to 90.
        yticks_rotation (int): Degrees of rotation for the yticks. Defaults to 0.

    Returns:
        The Seaborn figure
    """
    mask = np.triu(np.ones_like(association_matrix, dtype=np.bool))
    corr = association_matrix.to_numpy()
    vmin = min(corr.flatten()[~np.isnan(corr.flatten())])
    vmax = max(corr.flatten()[~np.isnan(corr.flatten())])

    plt.figure(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )

    ax = sns.heatmap(
        data=association_matrix,
        vmin=vmin,
        vmax=vmax,
        cmap=get_p_RdBl_cmap(),
        annot=annot,
        mask=mask,
        center=0,
        linewidths=2,
        square=True,
    )

    plt.title("Correlation Matrix")
    plt.xticks(rotation=xticks_rotation)
    plt.yticks(rotation=yticks_rotation)
    return ax
