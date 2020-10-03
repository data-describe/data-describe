import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt

from data_describe.config._config import get_option
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
        return _get_viz_backend(viz_backend)._seaborn_viz_importance(
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


def _seaborn_viz_importance(importance_values, idx, cols):
    """Plot feature importances.

    Args:
        importance_values: The importances
        idx: The sorted indices
        cols: The columns

    Returns:
        fig: The figure
    """
    plt.figure(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )
    plt.xlabel("Permutation Importance Value")
    plt.ylabel("Features")

    fig = sns.barplot(
        y=cols[idx], x=importance_values[idx], palette="Blues_d"
    ).set_title("Feature Importance")
    return fig
