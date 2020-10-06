import logging

from data_describe.misc.logging import OutputLogger
from data_describe.compat import _compat, _requires
from data_describe.config._config import get_option, _set_config


_set_config({"sensitive_data.sample_size": 100, "sensitive_data.score_threshold": 0.2})
_DEFAULT_SCORE_THRESHOLD = get_option("sensitive_data.score_threshold")

logger = logging.getLogger("presidio")
logger.setLevel(logging.WARNING)


@_requires("presidio_analyzer")
@_requires("spacy")
def presidio_engine():
    """Initialize presidio engine.

    Returns:
        Presidio engine
    """
    with OutputLogger("presidio", "INFO") as redirector:  # noqa: F841
        engine = _compat["presidio_analyzer"].AnalyzerEngine(
            default_score_threshold=_DEFAULT_SCORE_THRESHOLD, enable_trace_pii=True
        )
    return engine


engine = presidio_engine()
