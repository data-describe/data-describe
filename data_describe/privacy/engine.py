import contextlib
import logging

from data_describe.compat import presidio_analyzer
from data_describe.config._config import set_config, get_option


set_config({"sensitive_data.sample_size": 100, "sensitive_data.score_threshold": 0.2})
_DEFAULT_SCORE_THRESHOLD = get_option("sensitive_data.score_threshold")

logger = logging.getLogger("presidio")
logger.setLevel(logging.ERROR)


class OutputLogger:
    """Redirect logs.

    https://johnpaton.net/posts/redirect-logging/
    """

    def __init__(self, name="root", level="INFO"):
        self.logger = logging.getLogger(name)
        self.name = self.logger.name
        self.level = getattr(logging, level)
        self._redirector = contextlib.redirect_stdout(self)

    def write(self, msg):
        """Log messages."""
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def __enter__(self):
        self._redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Contextlib exception handling."""
        self._redirector.__exit__(exc_type, exc_value, traceback)


with OutputLogger("presidio", "WARN") as redirector:
    engine = presidio_analyzer.AnalyzerEngine(
        default_score_threshold=_DEFAULT_SCORE_THRESHOLD, enable_trace_pii=True
    )
