from mwdata.utilities.colorscale import *


def test_colorfade():
    a = (0, 255, 255)
    b = (100, 201, 201)
    assert color_fade(a, b, 0.5) == (50, 228, 228)


def test_rgb_to_str():
    assert rgb_to_str((0, 0, 0)) == "rgb(0, 0, 0)"
