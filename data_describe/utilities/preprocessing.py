import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def preprocess(data, target, impute="simple", encode="label"):
<<<<<<< HEAD
    """Simple preprocessing pipeline for ML.
=======
    """Simple preprocessing pipeline for ML
>>>>>>> Clean up (Fixes #151) (#152)

    Args:
        data: A Pandas dataframe
        target: Name of the target feature
        impute: Method to use for imputing numeric variables. Only 'simple' is implemented.
        encode: Method to use for encoding categorical variables. Only 'label' is implemented.

    Returns:
        (X, y) tuple of numpy arrays
    """
    y = data[target]
    data = data.drop(target, axis=1)

    data = data.dropna(axis=1, how="all")

    # Process numeric features
    num = data.select_dtypes(["number"])
    if num.shape[1] > 0:
        if impute == "simple":
            imp = SimpleImputer(missing_values=np.nan, strategy="mean")
            x_num = imp.fit_transform(num)
        else:
            raise NotImplementedError("Unknown imputation method: {}".format(impute))

    # Ordinal encode everything else
    # TODO: Address date and text columns
    cat = data[[c for c in data.columns if c not in num.columns]]
    if cat.shape[1] > 0:
        cat = cat.astype(str)
        cat = cat.fillna("")
        if encode == "label":
            x_cat = cat.apply(LabelEncoder().fit_transform)
        else:
            raise NotImplementedError("Unknown encoding method: {}".format(encode))

    if num.shape[1] > 0 and cat.shape[1] > 0:
        X = pd.DataFrame(
            np.concatenate([x_num, x_cat], axis=1),
            columns=list(num.columns.values) + list(cat.columns),
        )
    elif num.shape[1] > 0:
        X = pd.DataFrame(x_num, columns=num.columns)
    elif cat.shape[1] > 0:
        X = pd.DataFrame(x_cat, columns=x_cat.columns)
    else:
        raise ValueError("No numeric or categorical columns were found.")

    return X, y
