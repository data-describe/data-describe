import seaborn as sns
import matplotlib.pyplot as plt

from data_describe.config._config import get_option


def viz_importance(importance_values, idx, cols):
    """Plot feature importances.

    Args:
        importance_values: The importances
        idx: The sorted indices
        cols: The columns

    Returns:
        fig: The figure
    """
    plt.figure(
        figsize=(get_option("display.fig_width"), get_option("display.fig_height"))
    )
    plt.xlabel("Permutation Importance Value")
    plt.ylabel("Features")

    fig = sns.barplot(
        y=cols[idx], x=importance_values[idx], palette="Blues_d"
    ).set_title("Feature Importance")
    return fig
