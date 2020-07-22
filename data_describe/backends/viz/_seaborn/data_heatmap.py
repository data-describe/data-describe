from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from data_describe.config._config import get_option


def viz_data_heatmap(data, colnames: List[str], missing: bool = False, **kwargs):
    """Plots the data heatmap

    Args:
        data: The dataframe
        colnames: The column names, used for tick labels
        missing: If True, plots missing values instead
        kwargs: Keyword arguments passed to seaborn.heatmap
    """
    plot_options = {
        "cmap": "PuRd" if missing else "viridis",
        "robust": True,
        "center": 0,
        "xticklabels": False,
        "yticklabels": colnames,
        "cbar_kws": {"shrink": 0.5},
    }

    plot_options.update(kwargs)

    plt.figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    heatmap = sns.heatmap(data, **plot_options)
    plt.title("Data Heatmap")
    plt.ylabel("Variable")
    plt.xlabel("Record #")

    return heatmap
