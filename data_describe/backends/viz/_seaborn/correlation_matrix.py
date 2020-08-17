import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from data_describe.config._config import get_option


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
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"),)
    )

    ax = sns.heatmap(
        association_matrix,
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",
        annot=annot,
        mask=mask,
        center=0,
        linewidths=2,
    )

    plt.title("Correlation Matrix")
    plt.xticks(rotation=xticks_rotation)
    plt.yticks(rotation=yticks_rotation)
    return ax
