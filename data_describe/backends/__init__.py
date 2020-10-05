"""data-describe backends.

This subpackage is typically not used by end users of data-describe.

`backends` implements a pluggable architecture for the data-describe package.
This allows third-party libraries to provide alternative implementation details
for data-describe: for example, implementing Bokeh plot (visualizations, abbreviated
as `viz`) or implementing calculations on compute clusters (`compute`).

Default implementations in data-describe are also exposed as plugins to this backend system.
"""
from data_describe.backends._backends import (  # noqa
    _get_viz_backend,
    _get_compute_backend,
)
