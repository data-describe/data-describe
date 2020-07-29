from scipy.stats import f_oneway, levene


def varying(group, alpha=0.01):
    """Identifies varying box plots (i.e. with different means) using the one-way ANOVA test.

    Args:
        group: The groups from `split_by_category`
        alpha: The significance level

    Returns:
        True if statistically significant
    """
    F, p = f_oneway(*group)
    return p <= alpha


def heteroscedastic(group, alpha=0.01):
    """Identifies heteroscedasticity in box plots using the Brown-Forscythe test.

    Args:
        group: The groups from `split_by_category`
        alpha: The significance level

    Returns:
        True if statistically significant
    """
    W, p = levene(*group, center="median")
    return p <= alpha
