import numpy as np
from scipy.stats import skew


def spikey(data, factor=10):
    """Identifies 'spikey' histograms where the tallest bin is `factor` times the average bin count.

    Args:
        data: The 1-d data array
        factor: The factor

    Returns:
        True if statistically significant
    """
    counts, bins = np.histogram(data, bins="auto")
    return max(counts) >= factor * np.mean(counts)


def skewed(data, threshold=3):
    """Identifies skewed data as being over a threshold skew value.

    Args:
        data: The 1-d data array
        threshold: The skew value

    Returns:
        True if statistically significant
    """
    return skew(data) >= threshold
