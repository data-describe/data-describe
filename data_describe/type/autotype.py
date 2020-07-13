import warnings
import inspect

import pandas as pd
import numpy as np

import data_describe.type.dtypes
from data_describe.type.dtypes import (
    BaseType,
    CategoryType,
    IntegerType,
    ReferenceType,
)


def guess_dtypes(df, strict=True, sample_size=100, random_state=1, types=None):
    """Use heuristics to determine the column data types.

    Args:
        df: The data, as a Pandas data frame
        strict: If True, will require a rule to pass for every sampled record in order to assign a data type
        sample_size: The number of records to sample for inspection. If less than 1, will be interpreted as a fraction of the data.
        random_state: The random seed
        types: The type classes

    Returns:
        Dictionary containing data types for each column
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a Pandas Data Frame.")

    if not types:
        types = inspect.getmembers(data_describe.type.dtypes, _class_in_dtypes_module)
        types = [t() for name, t in types]

    return {
        col: guess_series_dtypes(
            df[col],
            strict=strict,
            sample_size=sample_size,
            random_state=random_state,
            types=types,
        )
        for col in df.columns
    }


def guess_series_dtypes(
    series, strict=True, sample_size=100, random_state=1, types=None
):
    """Use heuristics to determine the column data type for a pandas series.


    Args:
        series: A Pandas series
        strict: If True, will require a rule to pass for every sampled record in order to assign a data type
        sample_size: The number of records to sample for inspection. If less than 1, will be interpreted as a fraction of the data.
        random_state: The random seed
        types: The type classes

    Returns:
        The name of the type as a string
    """
    if not types:
        types = inspect.getmembers(data_describe.type.dtypes, _class_in_dtypes_module)
        types = [t() for name, t in types]

    if sample_size < 1:  # Fractional sample size
        sample_size = int(sample_size * series.shape[0])

    if series.shape[0] > sample_size:
        series = series[series.notnull()]
        if series.shape[0] < 1:
            return "None"
        elif series.shape[0] <= sample_size:
            warnings.warn(
                "Non-null values for {} are less than the specified sample size.".format(
                    series.name
                )
            )
            sampled_data = series
            pass
        else:
            sampled_data = sample_pandas_series(
                series, sample_size=sample_size, random_state=random_state
            )
    else:
        sampled_data = series

    guesses = dtype_heuristics(sampled_data, strict=strict, types=types)

    # Special handling for Reference type
    guess = max(guesses, key=guesses.get)
    if guess in [CategoryType.name, IntegerType.name]:
        if ReferenceType().test_meta(meta_features(series)) == 1:
            guess = ReferenceType.name

    return guess


def sample_pandas_series(series, sample_size=100, random_state=1):
    """Re-implementation of sampling from a Pandas series without shuffling.

    This is much faster than Pandas' implementation of .sample()

    Args:
        series: A Pandas series
        sample_size: Number of records to sample
        random_state: The random state

    Returns:
        A sampled Pandas series
    """
    np.random.seed(random_state)
    locs = np.random.randint(0, len(series), sample_size * 2)
    locs = np.unique(locs)[:sample_size]
    return series.take(locs)


def dtype_heuristics(series, strict=True, types=None):
    """Guess the data type for one Pandas series by testing each value.

    Args:
        series: A Pandas series
        strict: If True, will enforce rules strictly
        types: The type instances

    Returns:
        A type name
    """
    if not types:
        types = inspect.getmembers(data_describe.type.dtypes, _class_in_dtypes_module)
        types = [t() for name, t in types]

    guesses = {}
    for this_type in types:
        test_results = [this_type.test(x) for x in series if x is not None]
        if strict:
            guesses[this_type.name] = this_type.weight * all(
                [x >= 0 for x in test_results]
            ) * (sum(test_results)) + sum([x == 0 for x in test_results])
        else:
            guesses[this_type.name] = this_type.weight * sum(test_results)
    return guesses


def get_class_instance_by_name(name, types=None):
    """Return the type class instance by its name property.

    Args:
        name: The name
        types: A list of type classes

    Returns:
        A type instance
    """
    if name == "None":
        return BaseType()

    if not types:
        types = inspect.getmembers(data_describe.type.dtypes, _class_in_dtypes_module)
        types = [t() for name, t in types]

    instance = [t for t in types if t.name == name]
    if len(instance) > 0:
        return instance[0]
    else:
        raise ValueError("Unrecognized type name: {}".format(name))


def meta_features(series):
    """Calculate column meta features.

    Args:
        series: A Pandas series

    Returns:
        A dictionary with the meta features
    """
    if isinstance(series, pd.Series):
        size = len(series)
        cardinality = series.nunique()

        meta = {"size": size, "cardinality": cardinality}
        return meta


def select_dtypes(df, types, omit=False, dtypes=None):
    """Select columns in a dataframe by data type.

    Args:
        df: A Pandas Data frame
        types: A list of type strings to select; i.e. one or more of ['String','Category','Date','Integer','Decimal','Boolean','Reference']
        omit: If True, omit rather than select the columns of specified type
        dtypes: A dictionary with column names as keys and the data type as values. Missing columns are filled in using `guess_dtypes`.

    Returns:
        A Pandas dataframe
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a Pandas Data Frame.")

    if dtypes is None:
        dtypes = guess_dtypes(df)
    if len(dtypes.keys()) < df.shape[1]:
        default_dtypes = guess_dtypes(df)
        default_dtypes.update(dtypes)
        dtypes = default_dtypes

    if "None" in dtypes.values():
        warnings.warn("A column with None-type is present in the data.")

    if isinstance(types, str):
        types = [types]
    elif isinstance(types, list):
        pass
    else:
        raise ValueError("types must be a string or list of strings.")

    if omit:
        selected_columns = [k for k, v in dtypes.items() if v not in types]
    else:
        selected_columns = [k for k, v in dtypes.items() if v in types]

    return df.loc[:, selected_columns]


def cast_dtypes(df, dtypes=None, exclude=None):
    """Convert the data frame column types.

    Args:
        df: A Pandas data frame
        dtypes: A dictionary with column names as keys and the data type as values. Missing columns are filled in using `guess_dtypes`.
        exclude: A list of column names to exclude


    Returns:
        A Pandas data frame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a Pandas Data Frame.")

    if dtypes is None:
        dtypes = guess_dtypes(df)
    if len(dtypes.keys()) < df.shape[1]:
        default_dtypes = guess_dtypes(df)
        default_dtypes.update(dtypes)
        dtypes = default_dtypes

    for column, dtype in dtypes.items():
        if exclude:
            if column in exclude:
                continue

        dtype = get_class_instance_by_name(dtype)
        try:
            dtype = dtype.result_type[0]
        except IndexError:
            raise ValueError(
                "Could not determine data type ({}) to cast feature {}".format(
                    dtype, column
                )
            )

        if not isinstance(dtype, type(None)):  # TODO: Check for NoneType
            try:
                df[column] = df[column].astype(dtype)
            except TypeError as e:
                if "Int64" in str(dtype):
                    df[column] = (
                        df[column].astype(float).astype(dtype())
                    )  # Workaround for pandas' nullable int dtype
                else:
                    raise e
            except ValueError:
                warnings.warn(
                    "Failed to cast '{}' as a {}. Data type was kept as a string.".format(
                        column, dtype
                    )
                )

    return df


def _class_in_dtypes_module(member):
    """Predicate function to find type classes defined in data_describe.type.dtypes.

    Args:
        member: A member of the current module

    Returns:
        True if it is a class in data_describe.type.dtypes
    """
    if inspect.isclass(member):
        module = getattr(member, "__module__", None)
        if module:
            if module == "data_describe.type.dtypes":
                return True
    return False
