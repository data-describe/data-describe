import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def scatter_plots(
    data,
    plot_mode="matrix",
    threshold=None,
    joint_kws=None,
    scatter_kws=None,
    dist_kws=None,
):
    """Scatter plots

    Args:
        data:  A Pandas data frame
        plot_mode: Default = 'diagnostic"
            diagnostic: Choose plots by scatter plot diagnostics
            matrix: Generate the full scatter plot matrix
            all: Generate all individual scatter plots
        threshold: The scatter plot diagnostic threshold value [0,1] for returning a plot
            If a number: Returns all plots where at least one metric is above this threshold
            If a dictionary: Returns plots where the metric is above its threshold
            For example, {"Outlier": 0.9} returns plots with outlier metrics above 0.9
            The available metrics are: Outlier, Convex, Skinny, Skewed, Stringy, Straight, Monotonic, Clumpy, Striated
        joint_kws: Keywords to pass to seaborn.JointGrid
        scatter_kws: Keywords to pass to seaborn.scatterplot
        dist_kws: Keywords to pass to the seaborn.distplot

    Returns:
        Seaborn figure or list of figures
    """
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(["number"])
    else:
        raise NotImplementedError("Must be a Pandas data frame")

    if plot_mode == "matrix":
        fig = sns.pairplot(data)
        plt.show()
        plt.close()
        return fig
    elif plot_mode == "all":
        num_df = data.select_dtypes(["number"])
        pairs = [
            (i, j)
            for i in range(len(num_df.columns))
            for j in range(len(num_df.columns))
            if j > i
        ]
        fig = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r"Using a non-tuple sequence for multidimensional indexing is deprecated",
            )
            for p in pairs:
                x_col = num_df.columns.values[p[0]]
                y_col = num_df.columns.values[p[1]]
                fig.append(
                    scatter_plot(
                        data, x_col, y_col, joint_kws, scatter_kws, dist_kws
                    )
                )
            return fig

    elif plot_mode == "diagnostic":
        from data_describe.metrics.bivariate import Scagnostics

        scag = Scagnostics(data)
        metrics = scag.calculate()
        metrics = dict((k, v) for (k, v) in metrics.items() if len(v) > 0)

        if threshold is not None:
            metrics = filter_threshold(metrics, threshold)

        if len(metrics.items()) > 0:
            fig = []
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=r"Using a non-tuple sequence for multidimensional indexing",
                )
                for k in metrics.keys():
                    x_col = scag.names[k[0]]
                    y_col = scag.names[k[1]]
                    fig.append(
                        scatter_plot(
                            data,
                            x_col,
                            y_col,
                            joint_kws,
                            scatter_kws,
                            dist_kws,
                        )
                    )
        else:
            raise ValueError("No plots meeting threshold requirement.")
        return fig
    else:
        raise ValueError(f"Unknown plot mode: {plot_mode}")


def scatter_plot(
    data, x, y, joint_kws=None, scatter_kws=None, dist_kws=None
):
    """Generate one scatter (joint) plot

    Args:
        data: A Pandas data frame
        x: The x-axis column name
        y: The y-axis column name
        joint_kws: Keywords to pass to seaborn.JointGrid
        scatter_kws: Keywords to pass to seaborn.scatterplot
        dist_kws: Keywords to pass to the seaborn.distplot
        context: The context

    Returns:
        The Seaborn figure
    """
    if joint_kws is None:
        # joint_kws = {"height": max(context.fig_width, context.fig_height)}
        pass
    if scatter_kws is None:
        scatter_kws = {}
    if dist_kws is None:
        dist_kws = {"kde": False, "rug": False}

    g = sns.JointGrid(data[x], data[y], **joint_kws)
    g = g.plot_joint(sns.scatterplot, **scatter_kws)
    g = g.plot_marginals(sns.distplot, **dist_kws)
    plt.show()
    plt.close()
    return g


def filter_threshold(metrics, threshold=0.85):
    """Filter the plots by scatter plot diagnostic threshold

    Args:
        metrics: The metrics dictionary from Scagnostics
        threshold: The scatter plot diagnostic threshold value [0,1] for returning a plot
            If a number: Returns all plots where at least one metric is above this threshold
            If a dictionary: Returns plots where the metric is above its threshold
            For example, {"Outlier": 0.9} returns plots with outlier metrics above 0.9
            The available metrics are: Outlier, Convex, Skinny, Skewed, Stringy, Straight, Monotonic, Clumpy, Striated

    Returns:
        A dictionary of pairs that match the filter
    """
    if isinstance(threshold, dict):
        return dict(
            (k, v)
            for (k, v) in metrics.items()
            if all([v[m] >= threshold[m] for m in threshold.keys()])
        )
    else:
        return dict(
            (k, v) for (k, v) in metrics.items() if max(v.values()) >= threshold
        )
