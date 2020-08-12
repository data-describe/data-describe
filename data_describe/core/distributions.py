from data_describe._widget import BaseWidget
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends import _get_viz_backend, _get_compute_backend


def distribution(
    data, diagnostic=True, compute_backend=None, viz_backend=None, **kwargs
):
    """Distribution Plots.

    Args:
        data: Data Frame
        diagnostic: If True, will run diagnostics to select "interesting" plots.
        plot_all: If True, plot all features without filtering for "interesting" features
        max_categories: Maximum categories to show in violin plots. Additional categories will be combined into the "__OTHER__" contrast.
        spike: The factor threshold for identifying spikey histograms
        skew: The skew threshold for identifying skewed histograms
        hist_kwargs: Keyword arguments to be passed to seaborn.distplot
        violin_kwargs: Keyword arguments to be passed to seaborn.violinplot

    Returns:
        Matplotlib graphics
    """
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("DataFrame required.")

    widget = _get_compute_backend(compute_backend, data).compute_distribution(
        data, diagnostic=diagnostic, **kwargs
    )
    return widget


class DistributionWidget(BaseWidget):
    """Distribution Widget."""

    def __init__(
        self,
        input_data=None,
        num_data=None,
        cat_data=None,
        is_spikey=None,
        is_skewed=None,
        max_categories=None,
        label_name=None,
        spike_factor=None,
        skew_factor=None,
        categories_to_squash=None,
        viz_backend=None,
    ):
        """Distribution Plots."""
        self.input_data = input_data
        self.num_data = num_data
        self.cat_data = cat_data
        self.is_spikey = is_spikey
        self.is_skewed = is_skewed
        self.max_categories = max_categories
        self.label_name = label_name
        self.spike_factor = spike_factor
        self.skew_factor = skew_factor
        self.categories_to_squash = categories_to_squash
        self.viz_backend = viz_backend

    def show(self, viz_backend=None, **kwargs):
        """Show the default visualization.

        Args:
            viz_backend (str, optional): The visualization backend.
        """
        backend = viz_backend or self.viz_backend
        return _get_viz_backend(backend).viz_distribution_diagnostic(
            self.input_data, self.is_skewed, self.is_spikey, **kwargs
        )

    def plot_distribution(
        self, x: str = None, contrast: str = None, viz_backend=None, **kwargs
    ):
        """Generate distribution plot(s).

        Numeric features will be visualized using a histogram/violin plot, and any other types will be
        visualized using a categorical bar plot.

        Args:
            x (str, optional): The feature name to plot. If None, will plot all features.
            contrast (str, optional): The feature name to compare histograms by contrast.
            viz_backend (optional): The visualization backend.
            **kwargs: Additional keyword arguments for the visualization backend.

        Returns:
            Histogram plot(s).
        """
        backend = viz_backend or self.viz_backend

        if x is None:
            return _get_viz_backend(backend).viz_all_distributions(
                data=self.input_data.drop(contrast, axis=1),
                contrast=self.input_data[contrast],
                **kwargs
            )
        else:
            return _get_viz_backend(backend).viz_distribution(
                data=self.input_data, x=x, contrast=contrast, **kwargs
            )

    # def plot_histogram(
    #     self, x: str = None, contrast: str = None, viz_backend=None, **kwargs
    # ):
    #     """Generate histogram plot(s).

    #     Args:
    #         x (str, optional): The feature name to plot. If None, will plot all numeric features.
    #         contrast (str, optional): The feature name to compare histograms by contrast.
    #         viz_backend (optional): The visualization backend.
    #         **kwargs: Additional keyword arguments for the visualization backend.

    #     Returns:
    #         Histogram plot(s).
    #     """
    #     backend = viz_backend or self.viz_backend

    #     if x is None:
    #         return _get_viz_backend(backend).viz_all_histogram(
    #             data=self.input_data.drop(contrast, axis=1),
    #             contrast=self.input_data[contrast],
    #             **kwargs
    #         )
    #     else:
    #         return _get_viz_backend(backend).viz_histogram(
    #             data=self.input_data, x=x, contrast=contrast, **kwargs
    #         )

    # def plot_violin(
    #     self, x: str = None, contrast: str = None, viz_backend=None, **kwargs
    # ):
    #     """Generate violin plot(s).

    #     Args:
    #         x (str, optional): The feature name to plot. If None, will plot all numeric features.
    #         contrast (str, optional): The feature name to compare histograms by contrast.
    #         viz_backend (optional): The visualization backend.
    #         **kwargs: Additional keyword arguments for the visualization backend.

    #     Returns:
    #         Violin plot(s).
    #     """
    #     backend = viz_backend or self.viz_backend

    #     if x is None:
    #         return _get_viz_backend(backend).viz_all_violins(
    #             data=self.input_data.drop(contrast, axis=1),
    #             contrast=self.input_data[contrast],
    #             **kwargs
    #         )
    #     else:
    #         return _get_viz_backend(backend).viz_violin(
    #             data=self.input_data, x=x, contrast=contrast, **kwargs
    #         )

    # def plot_bar(self, x: str = None, contrast: str = None, viz_backend=None, **kwargs):
    #     """Generate bar plot(s).

    #     Args:
    #         x (str, optional): The feature name to plot. If None, will plot all categorical features.
    #         contrast (str, optional): The feature name to compare using stacked or side-by-side bars.
    #         viz_backend (optional): The visualization backend.
    #         **kwargs: Additional keyword arguments for the visualization backend.

    #     Returns:
    #         Violin plot(s).
    #     """
    #     backend = viz_backend or self.viz_backend

    #     if x is None:
    #         return _get_viz_backend(backend).viz_all_bar(
    #             data=self.input_data.drop(contrast, axis=1),
    #             contrast=self.input_data[contrast],
    #             **kwargs
    #         )
    #     else:
    #         return _get_viz_backend(backend).viz_bar(
    #             data=self.input_data, x=x, contrast=contrast, **kwargs
    #         )


# def plot_histograms(data, plot_all, spike=10, skew=6, hist_kwargs=None):
#     """Makes histogram plots.

#     Args:
#         data: A pandas data frame
#         plot_all: If True, plot all histograms without filtering for skew or spikes
#         spike: The factor threshold for identifying spikey histograms
#         skew: The skew threshold for identifying skewed histograms
#         hist_kwargs: Keyword arguments to be passed to seaborn.distplot

#     Returns:
#         Matplotlib graphics
#     """
#     fig = []
#     for column in data.columns:
#         with warnings.catch_warnings():
#             warnings.filterwarnings(
#                 "ignore",
#                 contrast=FutureWarning,
#                 module="scipy",
#                 message=r"Using a non-tuple sequence for multidimensional indexing is deprecated",
#             )
#             if plot_all:
#                 fig.append(plot_histogram(data[column].dropna(), hist_kwargs))
#             elif spikey(data[column].dropna(), factor=spike) or skewed(
#                 data[column].dropna(), threshold=skew
#             ):
#                 fig.append(plot_histogram(data[column].dropna(), hist_kwargs))
#         return fig


# def plot_violins(
#     data, num, cat, max_categories, plot_all=False, alpha=0.01, violin_kwargs=None,
# ):
#     """Makes violin plots.

#     Args:
#         data: A pandas data frame
#         num: The numeric feature names
#         cat: The categorical feature names
#         max_categories: Maximum categories to show in violin plots. Additional categories will be combined into
#         plot_all: If True, plot all violin plots without variation or heteroscedascity
#         alpha: The significance level for Levene's test and one-way ANOVA
#         violin_kwargs: Keyword arguments to be passed to seaborn.violinplot

#     Returns:
#         Matplotlib graphics
#     """
#     fig = []
#     for n in num:
#         for c in cat:
#             if plot_all:
#                 with warnings.catch_warnings():
#                     warnings.filterwarnings(
#                         "ignore",
#                         contrast=FutureWarning,
#                         module="scipy",
#                         message=r"Using a non-tuple sequence for multidimensional",
#                     )
#                     y, x = roll_up_categories(data[n], data[c], max_categories)
#                     fig.append(plot_violin(x, y, data, violin_kwargs,))
#             else:
#                 grp = split_by_contrast(data[[c, n]].dropna(), c, n)
#                 y, x = roll_up_categories(data[n], data[c], max_categories)
#                 with warnings.catch_warnings():
#                     warnings.filterwarnings(
#                         "ignore",
#                         contrast=FutureWarning,
#                         module="scipy",
#                         message=r"Using a non-tuple sequence for multidimensional",
#                     )
#                     if varying(grp, alpha=alpha) or heteroscedastic(grp, alpha=alpha):
#                         fig.append(plot_violin(x, y, data, violin_kwargs))
#     return fig


# def plot_violin(x, y, data, violin_kwargs=None):
#     """Make a single violin plot.

#     Args:
#         x: The x (categorical) data vector or name string
#         y: The y (numeric) data vector or name string
#         data: The data
#         violin_kwargs: Keyword arguments to be passed to seaborn.violinplot


#     Returns:
#         Matplotlib graphic
#     """
#     # plt.figure(figsize=(context.fig_width.fig_height)) # TODO (haishiro): Replace with get_option
#     if violin_kwargs is None:
#         violin_kwargs = {"cut": 0}
#     fig = sns.violinplot(x, y, data=data, **violin_kwargs)
#     fig.set_title("Violin Plot of " + y.name + " vs " + x.name)
#     plt.xticks(rotation=45, ha="right")
#     plt.show()
#     plt.close()
#     return fig


# def split_by_contrast(df, contrast, num):
#     """Splits the numeric feature `num` by the categorical feature `contrast`.

#     Args:
#         df: A pandas data frame
#         contrast: The name of the categorical feature, as a string
#         num: The name of the numeric feature, as a string

#     Returns:
#         A list of lists, where each list is the numeric data for a contrast
#     """
#     g = df.groupby(contrast)
#     return [x[1][num].to_numpy() for x in g]