def color_fade(a, b, w):
<<<<<<< HEAD
    """Interpolates between two RGB colors.
=======
    """Interpolates between two RGB colors
>>>>>>> Clean up (Fixes #151) (#152)

    Args:
        a: The first color as an RGB tuple
        b: The second color as an RGB tuple
        w: The weight w in w * a + (1 - w) * b

    Returns:
        RGB tuple
    """
    new_color = tuple(w * i + (1 - w) * j for i, j in zip(a, b))
    return new_color


def rgb_to_str(rgb):
<<<<<<< HEAD
    """Represents the RGB tuple as a string "rgb(r, g, b)" to be used in Plotly colorscale.
=======
    """Represents the RGB tuple as a string "rgb(r, g, b)" to be used in Plotly colorscale
>>>>>>> Clean up (Fixes #151) (#152)

    Args:
        rgb: RGB tuple
    """
    return "rgb" + str(rgb)
