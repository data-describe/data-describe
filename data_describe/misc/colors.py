from typing import List

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colors import colorConverter


def get_p_RdBl_cmap() -> LinearSegmentedColormap:
    """p_RdBl red-blue colormap."""
    cdict = {
        "red": [(0.0, 217, 217), (0.5, 242, 242), (1.0, 65, 65)],
        "green": [(0.0, 58, 58), (0.5, 242, 242), (1.0, 124, 124)],
        "blue": [(0.0, 70, 70), (0.5, 242, 242), (1.0, 167, 167)],
    }
    # Normalize
    n_cdict = {
        color: [(x[0], x[1] / 255.0, x[2] / 255.0) for x in scale]
        for color, scale in cdict.items()
    }
    return LinearSegmentedColormap("p_RdBl", n_cdict)


def mpl_to_plotly_cmap(cmap) -> List:
    """Convert a matplotlib cmap for use with Plotly.

    https://plotly.com/python/v3/matplotlib-colorscales/

    Args:
        cmap: The matplotlib colormap

    Returns:
        Plotly colorscale
    """
    pl_rgb = []
    norm = Normalize(vmin=0, vmax=255)
    for i in range(0, 255):
        k = colorConverter.to_rgb(cmap(norm(i)))
        pl_rgb.append(k)
    pl_entries = 255
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, "rgb" + str((C[0], C[1], C[2]))])
    return pl_colorscale
