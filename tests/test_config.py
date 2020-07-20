import data_describe as dd


def test_option_context_nargs():
    assert dd.options.backends.viz == "seaborn", "Unexpected default for viz backend"
    with dd.config.update_context("backends.viz", "mylib"):
        assert dd.options.backends.viz == "mylib", "Config context failed to update"
    assert (
        dd.options.backends.viz == "seaborn"
    ), "Config context exit did not restore options"


def test_option_context_dict():
    new_config = {"backends": {"viz": "mylib"}}
    assert dd.options.backends.viz == "seaborn", "Unexpected default for viz backend"
    with dd.config.update_context(new_config):
        assert dd.options.backends.viz == "mylib", "Config context failed to update"
    assert (
        dd.options.backends.viz == "seaborn"
    ), "Config context exit did not restore options"


def test_module_style_option():
    assert (
        dd.options.backends.compute == "pandas"
    ), "Unexpected default for compute backend"
    with dd.config.update_context(
        "backends.viz", ""
    ):  # Use context to prevent affecting other tests
        dd.options.backends.compute = "modin"
        assert (
            dd.options.backends.compute == "modin"
        ), "Module-style configuration set failed"
