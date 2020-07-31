from typing import Optional

from data_describe.backends import _get_compute_backend
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.config._config import get_option

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
