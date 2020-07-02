Compatibility
=============

This document describes the code patterns for managing optional dependencies used in Data Describe.

# setup.py
Any optional dependencies should be specified in the `EXTRAS` dictionary section of `setup.py`

```python
EXTRAS = {
    "nlp": ["nltk>=3.4", "pyldavis>=2.1.2", "gensim>=3.4.0"],
    "gcp": ["gcsfs>=0.2.1", "google-cloud-storage>=1.18.0"],
}
```
These optional dependencies may be installed by the end user e.g. using `pip install data_describe[nlp]`.

## Group Naming Convention
Optional dependencies should be grouped by the type of feature and the group name is preferred to be <= 5 characters.

# Implementation
## Module Imports
Data Describe modules utilizing optional dependencies should defer package import to `data_describe/compat.py`. Import statements should be wrapped in a `try except` block and update the `_PACKAGE_INSTALLED` dictionary on success or failure:
```python
try:
    import optional_package

    _PACKAGE_INSTALLED["optional_package"] = True
except ImportError:
    _PACKAGE_INSTALLED["optional_package"] = False
```

The feature implementation should import `compat`:

`from data_describe import compat`

## Function Requirements
A specific function may be marked as requiring a specific (optional) dependency by using the `@requires()` decorator from `compat.py`:
```python
@requires("optional_package")
def function_that_requires_optional_package():
    return
```

## Pandas Handling
Functions that accept Pandas-like inputs should avoid using any references to the package or instantiation of new objects, and should instead prefer using methods on the DataFrame object itself.
```python
# Preferred, uses a df function .agg()
df = df.agg(min) 

# Avoid, may not work with different backend implementations since it explicitly calls the pandas (pd) module
df = pd.DataFrame()
```

Currently, this only applies to Pandas and Modin, however this may expand to other frameworks in the future. `compat.py` contains a `_DATAFRAME_TYPE` that may be used to check for a *Pandas Dataframe-like* object, e.g.:
```python
if isinstance(df, _DATAFRAME_TYPE):
    ...
```