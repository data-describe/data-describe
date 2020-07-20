import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

from data_describe.utilities.preprocessing import preprocess


def importance(
    data,
    target,
    preprocess_func=None,
    estimator=None,
    return_values=False,
    truncate=True,
    **kwargs
):
    """ Variable importance chart

    Uses Random Forest Classifier by default

    Args:
        data: A Pandas data frame
        target: Name of the response column, as a string
        preprocess_func: A custom preprocessing function that takes a Pandas dataframe and the target/response column
        as a string. Returns X and y as tuple
        estimator: A custom sklearn estimator. Default is Random Forest Classifier
        return_values: If True, only the importance values as a numpy array
        truncate: If True, negative importance values will be truncated (set to zero)
        **kwargs: Other arguments to be passed to the preprocess function

    Returns:
        Matplotlib figure
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

    # plt.figure(figsize=(context.fig_width.fig_height)) # TODO (haishiro): Replace with get_option
    plt.xlabel("Permutation Importance Value")
    plt.ylabel("Features")

    fig = sns.barplot(
        y=X.columns[idx], x=importance_values[idx], palette="Blues_d"
    ).set_title("Feature Importance")

    if return_values:
        return importance_values
    else:
        return fig
