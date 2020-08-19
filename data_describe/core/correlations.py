from data_describe._widget import BaseWidget
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.backends import _get_viz_backend, _get_compute_backend


def correlation_matrix(
    data,
    cluster=False,
    categorical=False,
    compute_backend=None,
    viz_backend=None,
    **kwargs,
):
    """Correlation matrix of numeric variables.

    Args:
        data (DataFrame): A data frame
        cluster (bool): If True, use clustering to reorder similar columns together

        categorical (bool): If True, calculate categorical associations using Cramer's V, Correlation Ratio, and
            Point-biserial coefficient (aka Matthews correlation coefficient). All associations (including Pearson
            correlation) are in the range [0, 1].

        compute_backend: Select computing backend. Defaults to None (pandas).
        viz_backend: The visualization backend. Only 'plotly' is supported. Defaults to plotly.

    Returns:
        CorrelationMatrixWidget
    """
    if not isinstance(data, _DATAFRAME_TYPE):
        raise ValueError("Data frame required")

    corrwidget = _get_compute_backend(compute_backend, data).compute_correlation_matrix(
        data, cluster=cluster, categorical=categorical, **kwargs
    )

    corrwidget.viz_backend = viz_backend

    return corrwidget


class CorrelationMatrixWidget(BaseWidget):
    """Interface for collecting additional information about the correlation matrix."""

    def __init__(
        self,
        association_matrix=None,
        cluster_matrix=None,
        categorical=None,
        viz_data=None,
        **kwargs,
    ):
        """Correlation matrix.

        Args:
            association_matrix (DataFrame): The association matrix. Defaults to None.
            cluster_matrix (DataFrame, optional): The clustered association matrix. Defaults to None.
            categorical (bool, optional): True if association matrix contains categorical values. Defaults to None.
            viz_data (DataFrame): The data to be visualized. Defaults to None.
        """
        super(CorrelationMatrixWidget, self).__init__(**kwargs)
        self.association_matrix = association_matrix
        self.cluster_matrix = cluster_matrix
        self.viz_data = viz_data
        self.categorical = categorical

    def __str__(self):
        return "data-describe Correlation Matrix Widget"

    def show(self, viz_backend=None, **kwargs):
        """Show the Correlation Matrix plot."""
        backend = viz_backend or self.viz_backend

        if self.viz_data is None:
            raise ValueError("Could not find data to visualize.")

        return _get_viz_backend(backend).viz_correlation_matrix(self.viz_data, **kwargs)
