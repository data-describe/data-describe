import numpy as np

from data_describe._widget import BaseWidget
from data_describe._compat import _DATAFRAME_TYPE
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
        spike_value=None,
        skew_value=None,
        spike_factor=None,
        skew_factor=None,
        viz_backend=None,
    ):
        """Distribution Plots."""
        self.input_data = input_data
        self.spike_value = spike_value
        self.skew_value = skew_value
        self.spike_factor = spike_factor
        self.skew_factor = skew_factor
        self.viz_backend = viz_backend

    def show(self, viz_backend=None, **kwargs):
        """Show the default visualization.

        Args:
            viz_backend (str, optional): The visualization backend.
        """
        summary_string = """Distribution Summary:
        Skew detected in {} columns.
        Spikey histograms detected in {} columns.

        Use the method plot_distribution("column_name") to view plots for each feature.

        Example:
            dist = DistributionWidget(data)
            dist.plot_distribution("column1")
        """.format(
            np.sum(self.skew_value > self.skew_factor),
            np.sum(self.spike_value > self.spike_factor),
        )
        print(summary_string)

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

        return _get_viz_backend(backend).viz_distribution(
            data=self.input_data, x=x, contrast=contrast, **kwargs
        )
