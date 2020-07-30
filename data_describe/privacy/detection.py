from typing import Optional
import contextlib
import logging

from data_describe.backends import _get_compute_backend
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.config._config import set_config, get_option
from data_describe.compat import presidio_analyzer

# from data_describe.config._config import set_config, get_option

# from data_describe.privacy.engine import engine
# import contextlib
# import logging

# from data_describe.compat import presidio_analyzer
# from data_describe.config._config import set_config, get_option


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

set_config({"sensitive_data.engine": engine})

_DEFAULT_SCORE_THRESHOLD = get_option("sensitive_data.score_threshold")
_SAMPLE_SIZE = get_option("sensitive_data.sample_size")


def sensitive_data(
    df,
    redact: bool = True,
    encrypt: bool = False,
    detect_infotypes: bool = False,
    cols: Optional[list] = None,
    score_threshold=_DEFAULT_SCORE_THRESHOLD,
    sample_size=_SAMPLE_SIZE,
    compute_backend: Optional[str] = None,
    **kwargs,
):
    """Identifies, redacts, and encrypts PII data.

    Note: sensitive_data uses Microsoft's Presidio in the backend. Presidio can be help identify sensitive data.
    However, because Presidio uses trained ML models, there is no guarantee that Presidio will find all sensitive information.

    Args:
        df: The dataframe
        redact: If True, redact sensitive data
        encrypt: If True, anonymize data. Redact must be set to False
        detect_infotypes: If True, identifies infotypes for each column. Redact must be set to False
        cols: List of columns. Defaults to None
        score_threshold: Minimum confidence value for detected entities to be returned
        sample_size: Number of sampled rows used for identifying column infotypes
        compute_backend: Select compute backend
        **kwargs: Keyword arguments

    Returns:
        A dataframe if redact or anonymize is True.
        Dictionary of column infotypes if detect_infotypes is True
    """
    if not isinstance(df, _DATAFRAME_TYPE):
        raise TypeError("Pandas data frame or modin data frame required")
    if cols:
        if not isinstance(cols, list):
            raise TypeError("cols must be type list")
    if (encrypt or detect_infotypes) and redact:
        raise ValueError("Set redact=False to encrypt or detect_infotypes")

    df = _get_compute_backend(compute_backend, df).compute_sensitive_data(
        df=df,
        redact=redact,
        encrypt=encrypt,
        detect_infotypes=detect_infotypes,
        cols=cols,
        score_threshold=score_threshold,
        sample_size=sample_size,
        **kwargs,
    )

    return df
