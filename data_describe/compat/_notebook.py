import subprocess
import warnings

_PLOTLY_EXTENSION_INSTALLED = False


def _in_notebook():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


class JupyterPlotlyWarning(UserWarning):
    """Warning for missing jupyter-plotly extension in Jupyter Lab."""

    pass


def _check_plotly_extension():
    p = subprocess.Popen(
        ["jupyter", "labextension", "check", "jupyterlab-plotly"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()
    if "enabled" not in str(p[1]):
        warnings.warn(
            'Are you running in Jupyter Lab? The extension "jupyterlab-plotly" was not found and is required for Plotly visualizations in Jupyter Lab.',
            JupyterPlotlyWarning,
        )
