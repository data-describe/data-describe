Backends
========

This document describes the code patterns for implementing features in Data Describe while using the backend system.

# Definitions
- **widget**: A single feature in Data Describe, such as `correlation_matrix` or `data_summary`
- **backend**: A Python module or framework with a specific implementation. This term is also used interchangeably with the input data type from which an implementation approach is determined. For example, for any input Pandas DataFrame, the computation is expected to execute in memory. Backends for Data Describe are further subdivided into two groups:
    - **viz**: A visualization backend (framework). This is the plotting or visualization library (e.g. seaborn, Plotly) used to display results
    - **compute**: A computational backend (framework). This handles the preprocessing and computation for a feature. Pandas (and by extension, Modin) are intended to be the primary backend supported by Data Describe
- **plugin**: A Python package that provides the implementation details for a particular backend.

# Interface
Widgets should implement the following interface:

```python
from data_describe.backends import _get_compute_backend, _get_viz_backend

def my_widget([<arg1>, <arg2>], compute_backend=None, viz_backend=None, **kwargs):
    data = _get_compute_backend(compute_backend, data).compute_my_widget(<arg1>, **kwargs)
    return _get_viz_backend(viz_backend, data).viz_my_widget(data, <arg1>, **kwargs)
```

## Arguments
- `<arg1>, <arg2>`: Positional arguments should be reserved for options that affect the functionality and are not specific to a backend
- `compute_backend`, `viz_backend`: These two keyword arguments must be included to allow the user to easily specify the backend to be used
- `**kwargs`: Any other arguments should be handled by the `**kwargs` argument

## Backend Precendence
Data Describe chooses the first valid backend to use using the following order:

### Compute
1. The backend specified by the user in `compute_backend`
2. Determined from the input data and mapping provided in `_compat.py`
3. The backend set in Data Describe configuration options

### Viz
1. The backend specified by the user in `viz_backend`
2. The backend set in Data Describe configuration options

## Plugin Interface
- The `_get_compute_backend()` function should be used to load the appropriate backend that implements the data preprocessing or computation. The name of the compute function should be `compute_<widget_name>`. Any Python package that implements a `compute_<widget_name>` function could be used as a backend for Data Describe.
- The `_get_viz_backend()` function should be used to load the appropriate backend that implements the visualization. The name of the compute function should be `viz_<widget_name>`. Any Python package that implements a `viz_<widget_name>` function could be used as a backend for Data Describe.
- Since `**kwargs` may be shared across *compute* and *viz* interfaces, the implementation is expected to safely handle unknown keyword arguments, e.g.:
```python
def compute_my_widget(data, **kwargs):
    """This function will not fail if an unknown kwarg is passed in
    """
    kw1 = kwargs.get(kw, None)
    ...
    return
```
- For a package to be recognized as a plugin package for Data Describe, its `setup.py` must have `entry_points` defined with the following pattern:
```python
from setuptools import setup

# Credit: Pandas plotting backend
# https://github.com/pandas-dev/pandas/blob/master/setup.py
setup(
    ...,
    entry_points={
        "data_describe_viz_backends": [
            "<backend_name> = <package_name>:<path.to.implementation>",
        ],
        "data_describe_compute_backends": [
            "<backend_name> = <package_name>:<path.to.implementation>",
        ]
    }
)
)

```
- Data Describe's internal implementations also use this plugin system, e.g.:
```python
entry_points={
        "data_describe_viz_backends": [
            "seaborn = data_describe:backends.viz._seaborn",
            "plotly = data_describe:backends.viz._plotly"
        ],
        "data_describe_compute_backends": [
            "pandas = data_describe:backends.compute._pandas",
        ]
    },
```

# Dependencies (Internal)
Optional dependencies used by backends implemented in Data Describe should follow the patterns outlined in _compatIBILITY.md

# Contributor Checklist
- [ ] Add to `entry_points` in setup.py
- [ ] Main widget function does not contain any logic that belongs in the *compute* or *viz* backend
- [ ] Add the *compute* implementation under `data_describe/backends/compute/_[BACKEND_NAME].py` with the function name `compute_<widget_name>`
- [ ] Add the *viz* implementation under `data_describe/backends/viz/_[BACKEND_NAME].py` with the function name `viz_<widget_name>`
