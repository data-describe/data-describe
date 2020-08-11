import matplotlib.pyplot as plt
import seaborn as sns


def viz_plot_elbow(min_topics, max_topics, coherence_values):
    """Creates an elbow plot displaying coherence values vs number of topics.

    Args:

    Returns:
        fig: Elbow plot showing coherence values vs number of topics
    """
    # plt.figure(figsize=(context.fig_width.fig_height)) # TODO (haishiro): Replace with get_option

    fig = sns.lineplot(
        x=[num for num in range(min_topics, max_topics + 1)], y=coherence_values,
    )
    fig.set_title("Coherence Values Across Topic Numbers")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Values")
    return fig
