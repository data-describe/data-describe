import subprocess
import logging
from mwdata.utilities.load_data import load_data
from mwdata.core.summary import data_summary
from mwdata.core.data_heatmap import data_heatmap
from mwdata.core.distribution import distribution
from mwdata.core.scatter_plot import scatter_plots
from mwdata.core.correlation_matrix import correlation_matrix
from mwdata.core.importance import importance
from mwdata.core.cluster import cluster


# Check for plotly extensions
for extension in [
    "@jupyter-widgets/jupyterlab-manager",
    "jupyterlab-plotly",
    "plotlywidget",
]:
    p = subprocess.Popen(
        ["jupyter", "labextension", "check", extension], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()
    if "enabled" not in str(p[1]):
        logging.warning(
            f'The extension "{extension}" was not found and is required for Plotly-based visualizations.'
        )
