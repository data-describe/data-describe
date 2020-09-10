from matplotlib.colors import LinearSegmentedColormap

from data_describe.misc.colors import get_p_RdBl_cmap, mpl_to_plotly_cmap


def test_get_p_RdBl_cmap():
    cmap = get_p_RdBl_cmap()
    assert isinstance(
        cmap, LinearSegmentedColormap
    ), "Colormap is not an instance of LinearSegmentedColormap"


def test_mpl_to_plotly_cmap():
    cmap = get_p_RdBl_cmap()
    pl_cmap = mpl_to_plotly_cmap(cmap)
    assert len(pl_cmap) == 255, "Length of colorscale is not 255"
    assert all(
        isinstance(x[0], float) for x in pl_cmap
    ), "First index of elements in plotly colorscale is not a float"
    assert all(
        "rgb(" in x[1] for x in pl_cmap
    ), "Second index of elements in plotly colorscale doesn't contain 'rgb('"
