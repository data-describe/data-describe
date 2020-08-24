"""data-describe backends.

This subpackage implements a pluggable architecture for the data-describe package. This allows third-party libraries to provide alternative implementation details for data-describe, such as implementing Bokeh visualizations for data-describe features. Default implementations in data-describe are also exposed as plugins to this backend system.
"""
from data_describe.backends._backends import (  # noqa
    _get_viz_backend,
    _get_compute_backend,
)
