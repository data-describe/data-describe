from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from data_describe.config._config import get_option


def viz_elbow_plot(min_topics: int, max_topics: int, coherence_values: List[float]):
    """Creates an elbow plot displaying coherence values vs number of topics.

    Args:
        min_topics: Starting number of topics that were optimized for
        max_topics: Maximum number of topics that were optimized for
        coherence_values: A list of coherence values mapped from min_topics to max_topics

    Returns:
        Elbow plot showing coherence values vs number of topics
    """
    ax = sns.lineplot(
        x=[num for num in range(min_topics, max_topics + 1)], y=coherence_values,
    )
    ax.set_title("Coherence Values Across Topic Numbers")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Values")
    plt.figure(
        figsize=(get_option("display.matplotlib.fig_width"), get_option("display.matplotlib.fig_height"))
    )
    return ax
