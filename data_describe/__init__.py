import subprocess
import logging

from data_describe.utilities.load_data import load_data  # noqa: F401
from data_describe.core.summary import data_summary  # noqa: F401
from data_describe.core.data_heatmap import data_heatmap  # noqa: F401
from data_describe.core.distribution import distribution  # noqa: F401
from data_describe.core.scatter_plot import scatter_plots  # noqa: F401
from data_describe.core.correlation_matrix import correlation_matrix  # noqa: F401
from data_describe.core.importance import importance  # noqa: F401
from data_describe.core.cluster import cluster  # noqa: F401
from data_describe.config._config import options


# Check for plotly extensions
try:  # TODO: Move to optional checks
    for extension in [
        "@jupyter-widgets/jupyterlab-manager",
        "jupyterlab-plotly",
        "plotlywidget",
    ]:
        p = subprocess.Popen(
            ["jupyter", "labextension", "check", extension],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()
        if "enabled" not in str(p[1]):
            raise FileNotFoundError(
                f'The extension "{extension}" was not found and is required for Plotly-based visualizations.'
            )
except FileNotFoundError:
    logging.warning(
        f'The extension "{extension}" was not found and is required for Plotly-based visualizations.'
    )
