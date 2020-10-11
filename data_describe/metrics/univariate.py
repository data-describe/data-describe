import numpy as np
from scipy.stats import skew

from data_describe.compat import is_series


def spikey(data):
    """Calculates the "spikey-ness" of the histogram.

    Spikeyness is the ratio between the tallest bin and the average bin height.

    Args:
        data: The 1-d data array

    Returns:
        Ratio of the tallest bin height and the average bin height.
    """
    if is_series(data):
        data = data.dropna()
    else:
        data = data[~np.isnan(data)]
    counts, bins = np.histogram(data, bins="sturges")
    return max(counts) / np.mean(counts)


def skewed(data):
    """Calculates skew.

    Utilizes scipy.stats.

    Args:
        data: The 1-d data array

    Returns:
        The data skew.
    """
    return skew(data)
