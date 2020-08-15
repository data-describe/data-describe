import numpy as np
from scipy.stats import skew

from data_describe import compat


def spikey(data):
    """Calculates the "spikey-ness" of the histogram.

    Spikeyness is the ratio between the tallest bin and the average bin height.

    Args:
        data: The 1-d data array
    """
    if isinstance(data, compat._SERIES_TYPE):
        data = data.dropna()
    else:
        data = data[~np.isnan(data)]
    counts, bins = np.histogram(data.dropna(), bins="sturges")
    return max(counts) / np.mean(counts)


def skewed(data):
    """Calculates skew.

    Utilizes scipy.stats.

    Args:
        data: The 1-d data array
    """
    return skew(data)
