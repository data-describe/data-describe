import matplotlib.pyplot as plt
import seaborn as sns

from data_describe.config._config import fig_height, fig_width


def plot_data_heatmap(data, colnames, missing=False, **kwargs):
    """Plots the data heatmap

    Args:
        data: The dataframe
        missing: If True, plots only missing values
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

    plt.figure(figsize=(fig_width, fig_height))
    heatmap = sns.heatmap(data, **plot_options)
    plt.title("Data Heatmap")
    plt.ylabel("Variable")
    plt.xlabel("Record #")

    return heatmap
