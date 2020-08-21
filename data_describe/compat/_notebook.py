import subprocess
import warnings

from IPython import get_ipython


_IN_NOTEBOOK = get_ipython() is not None

if _IN_NOTEBOOK:
    for extension in ["jupyterlab-plotly"]:
        p = subprocess.Popen(
            ["jupyter", "labextension", "check", extension],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()
        if "enabled" not in str(p[1]):
            warnings.warn(
                f'The extension "{extension}" was not found and is required for Plotly-based visualizations.'
            )
