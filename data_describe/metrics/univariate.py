import numpy as np
from scipy.stats import skew


def spikey(data):
    """Calculates the "spikey-ness" of the histogram.

    Spikeyness is the ratio between the tallest bin and the average bin height.

    Args:
        data: The 1-d data array
    """
    counts, bins = np.histogram(data.dropna(), bins="sturges")
    return max(counts) / np.mean(counts)


def skewed(data):
    """Calculates skew.

    Utilizes scipy.stats.

    Args:
        data: The 1-d data array
    """
    return skew(data)
