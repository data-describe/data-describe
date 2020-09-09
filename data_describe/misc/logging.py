import logging
import contextlib


class OutputLogger:
    """Redirect stdout to logging.

    https://johnpaton.net/posts/redirect-logging/
    """

    def __init__(self, name="root", level="INFO"):
        self.logger = logging.getLogger(name)
        self.source = self.logger.name
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
