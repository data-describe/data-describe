import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from data_describe.utilities.preprocessing import preprocess
from data_describe.compat import _DATAFRAME_STATIC_TYPE


def compute_importance(
    data: _DATAFRAME_STATIC_TYPE,
    target: str,
    preprocess_func=None,
    estimator=None,
    truncate: bool = True,
    **kwargs
):
    """Computes importance using permutation importance.

    Uses Random Forest Classifier by default

    Args:
        data: A Pandas data frame
        target: Name of the response column, as a string
        preprocess_func: A custom preprocessing function that takes a Pandas dataframe and the target/response column
        as a string. Returns X and y as tuple
        estimator: A custom sklearn estimator. Default is Random Forest Classifier
        truncate: If True, negative importance values will be truncated (set to zero)
        **kwargs: Other arguments to be passed to the preprocess function

    Returns:
        importance_values: The importances
        idx: The sorted index of importance_values
        X.columns: The columns
    """
    if estimator is None:
        estimator = RandomForestClassifier(random_state=1)

    if preprocess_func is None:
        X, y = preprocess(data, target, **kwargs)
    else:
        X, y = preprocess_func(data, target, **kwargs)

    estimator.fit(X, y)
    pi = permutation_importance(estimator, X, y, n_repeats=5, random_state=1)

    importance_values = np.array(
        [max(0, x) if truncate else x for x in pi.importances_mean]
    )
    idx = importance_values.argsort()[::-1]
    return importance_values, idx, X.columns
