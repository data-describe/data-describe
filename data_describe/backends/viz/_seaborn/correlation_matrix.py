import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from data_describe.config._config import get_option


def viz_plot_correlation_matrix(association_matrix):
    """Plot the heatmap for the association matrix.

    Args:
        association_matrix: The association matrix

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
    plt.xticks(rotation=90)
    ax = sns.heatmap(
        corr,
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",
        annot=True,
        mask=mask,
        center=0,
        linewidths=2,
    )
    plt.title("Correlation Matrix")
    return ax
