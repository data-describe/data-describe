import logging

import pytest

from data_describe.misc.logging import OutputLogger


@pytest.mark.base
def test_output_logger(caplog):
    caplog.set_level(logging.INFO)
    with OutputLogger("log", "INFO"):
        print("test")
        assert "test" in caplog.text, "Print statement was not redirected to logging"


@pytest.mark.base
def test_output_logger_captured(caplog):
    caplog.set_level(logging.WARN)
    with OutputLogger("log", "INFO"):
        print("test")
        assert caplog.text == "", "Context failed to log at the correct level"
