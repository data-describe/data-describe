import subprocess
import logging
from mwdata.utilities.load_data import load_data  # noqa: F401
from mwdata.core.summary import data_summary  # noqa: F401
from mwdata.core.data_heatmap import data_heatmap  # noqa: F401
from mwdata.core.distribution import distribution  # noqa: F401
from mwdata.core.scatter_plot import scatter_plots  # noqa: F401
from mwdata.core.correlation_matrix import correlation_matrix  # noqa: F401
from mwdata.core.importance import importance  # noqa: F401
from mwdata.core.cluster import cluster  # noqa: F401


# Check for plotly extensions
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
        logging.warning(
            f'The extension "{extension}" was not found and is required for Plotly-based visualizations.'
        )
