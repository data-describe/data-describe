import warnings
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Using or importing the ABCs from 'collections",
)
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="eli5",
        message=r"inspect.getargspec()",
    )
    from eli5.sklearn import PermutationImportance
    
from mwdata.utilities.preprocessing import preprocess
from mwdata.utilities.contextmanager import _context_manager


@_context_manager
def importance(
    data,
    target,
    preprocess_func=None,
    estimator=RandomForestClassifier(random_state=1),
    return_values=False,
    truncate=True,
    context=None,
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
        context: The context
        **kwargs: Other arguments to be passed to the preprocess function

    Returns:
        Matplotlib figure
    """

    if preprocess_func is None:
        X, y = preprocess(data, target, **kwargs)
    else:
        X, y = preprocess_func(data, target, **kwargs)

    pi = PermutationImportance(estimator, cv=5, random_state=1)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r"The default value of n_estimators",
        )
        pi.fit(X, y)

    importance_values = np.array(
        [max(0, x) if truncate else x for x in pi.feature_importances_]
    )

    idx = importance_values.argsort()[::-1]

    plt.figure(figsize=(context.fig_width, context.fig_height))
    plt.xlabel("Permutation Importance Value")
    plt.ylabel("Features")

    fig = sns.barplot(
        y=X.columns[idx], x=importance_values[idx], palette="Blues_d"
    ).set_title("Feature Importance")

    if return_values:
        return importance_values
    else:
        return fig
