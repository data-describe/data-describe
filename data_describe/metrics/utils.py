import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/12951065/get-bins-coordinates-with-hexbin-in-matplotlib


def hexbin(x, y, **kwargs):
    """ Make a hexagonal grid in two dimensions

    Args:
        x: x data, as a 1-d array
        y: y data, as a 1-d array
        **kwargs: Keyword arguments to be passed to matplotlib.pyplot.hexbin

    Returns:
        A tuple of the counts in each bin, the bin centers, and the matplotlib.pyplot.hexbin object
    """
    fig = plt.figure()
    hexbin = plt.hexbin(x, y, mincnt=None, **kwargs)

    counts = hexbin.get_array().copy()
    centers = hexbin.get_offsets().copy()

    plt.close(fig.number)

    return counts, centers, hexbin
