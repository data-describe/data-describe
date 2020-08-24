"""data-describe configuration.

This module provides package-level configuration for the data-describe package.

Examples:
    Assignment of configuration items is heavily inspired by Pandas' options system that uses a "dotted-style"::

        import data_describe as dd
        dd.options.backends.compute = "modin"

    To see all configuration options, simply print the options object::

        print(dd.options) # Or simply dd.options as the last line in a Jupyter notebook cell

    A context manager is also provided to allow for temporary re-assignment of configurations that only affect a block of code::

        with dd.config.update_context("display.fig_height", 20):
            dd.correlation_matrix(df)
"""
from data_describe.config._config import update_context  # noqa: F401
