import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from data_describe.misc.preprocessing import preprocess
from data_describe.backends import _get_viz_backend, _get_compute_backend


def importance(
    data,
    target: str,
    preprocess_func=None,
    estimator=None,
    return_values=False,
    truncate=True,
    compute_backend=None,
    viz_backend=None,
    **kwargs
):
    """Variable importance chart.

    Uses Random Forest Classifier by default

    Args:
        data: A Pandas data frame
        target: Name of the response column, as a string
        preprocess_func: A custom preprocessing function that takes a Pandas dataframe and the target/response column
        as a string. Returns X and y as tuple
        estimator: A custom sklearn estimator. Default is Random Forest Classifier
        return_values: If True, only the importance values as a numpy array
        truncate: If True, negative importance values will be truncated (set to zero)
        compute_backend: The compute backend
        viz_backend: The visualization backend

        **kwargs: Other arguments to be passed to the preprocess function

    Returns:
        Matplotlib figure
    """
    importance_values, idx, cols = _get_compute_backend(
        compute_backend, data
    ).compute_importance(data, target, preprocess_func, estimator, truncate, **kwargs)

    if return_values:
        return importance_values
    else:
        return _get_viz_backend(viz_backend).viz_importance(
            importance_values, idx, cols
        )


def _pandas_compute_importance(
    data,
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
